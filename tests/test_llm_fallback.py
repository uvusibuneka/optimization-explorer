import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx

from optimization_explorer import llm


class QuotaFallbackTests(unittest.IsolatedAsyncioTestCase):
    def test_extract_json_accepts_valid_json_without_closing_fence(self):
        self.assertEqual(llm.extract_json("```json\n{\"ok\":true}"), {"ok": True})

    def test_extract_json_ignores_trailing_model_text(self):
        self.assertEqual(llm.extract_json('{"claims":[]}\nI have completed the analysis.'), {"claims": []})

    def test_extract_json_ignores_second_json_object(self):
        self.assertEqual(llm.extract_json('{"claims":[]}\n{"extra":true}'), {"claims": []})

    def test_extract_json_rejects_malformed_object(self):
        with self.assertRaisesRegex(ValueError, "no valid JSON object"):
            llm.extract_json('{"claims":')

    async def test_openkilo_ignores_placeholder_key_and_removes_authorization(self):
        response = SimpleNamespace(
            usage=None,
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"claims":[],"next":null,"terminal":"unresolved"}'))],
        )
        client_args = {}
        class FakeAsyncOpenAI:
            def __init__(self, **kwargs):
                client_args.update(kwargs)

        with patch.dict("os.environ", {
            "OPENKILO_MODEL":"openai/worker",
            "OPENKILO_BASE_URL":"http://127.0.0.1:18765/v1",
            "OPENKILO_API_KEY":"unused",
        }):
            with patch("litellm.acompletion", new=AsyncMock(return_value=response)) as complete, \
                 patch("openai.AsyncOpenAI", FakeAsyncOpenAI):
                await llm._call_model_once("worker", {}, "agent_trace_worker")
        self.assertNotIn("api_key", complete.await_args.kwargs)
        request = httpx.Request("POST", "http://127.0.0.1/test",
                                headers={"Authorization":"Bearer anonymous"})
        await client_args["http_client"].event_hooks["request"][0](request)
        self.assertNotIn("Authorization", request.headers)

    async def test_provider_connection_failure_logs_stage_and_host_without_credentials(self):
        secret = "do-not-log-this-key"
        with patch.dict("os.environ", {
            "OPENKILO_MODEL":"openai/kilo-auto/free",
            "OPENKILO_BASE_URL":"https://ai-gateway.kilo.ai/api/v1",
            "OPENKILO_API_KEY":secret,
        }), patch("litellm.acompletion", new=AsyncMock(
            side_effect=httpx.ConnectError("connection error")
        )):
            with self.assertLogs("optimization_explorer.llm", level="WARNING") as captured:
                with self.assertRaises(httpx.ConnectError):
                    await llm._call_model_once("worker", {}, "agent_trace_worker")
        message = "\n".join(captured.output)
        self.assertIn("provider=OPENKILO", message)
        self.assertIn("stage=agent_trace_worker", message)
        self.assertIn("endpoint_host=ai-gateway.kilo.ai", message)
        self.assertIn("category=connection_or_timeout", message)
        self.assertNotIn(secret, message)

    async def test_openkilo_uses_configured_api_key_for_worker(self):
        response = SimpleNamespace(
            usage=None,
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"claims":[],"next":null,"terminal":"unresolved"}'))],
        )
        with patch.dict("os.environ", {
            "OPENKILO_MODEL":"openai/kilo-auto/free",
            "OPENKILO_BASE_URL":"https://ai-gateway.kilo.ai/api/v1",
            "OPENKILO_API_KEY":"kilo-test-key",
        }):
            with patch("litellm.acompletion", new=AsyncMock(return_value=response)) as complete:
                await llm._call_model_once("worker", {}, "agent_trace_worker")
        self.assertEqual(complete.await_args.kwargs["api_key"], "kilo-test-key")

    async def test_openkilo_loads_api_key_from_credential_file(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path
        response = SimpleNamespace(
            usage=None,
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"claims":[],"next":null,"terminal":"unresolved"}'))],
        )
        with TemporaryDirectory() as tmp:
            key_file=Path(tmp)/"openkilo-api-key"
            key_file.write_text("kilo-file-key\n", encoding="utf-8")
            with patch.dict("os.environ", {
                "OPENKILO_MODEL":"openai/kilo-auto/free",
                "OPENKILO_BASE_URL":"https://ai-gateway.kilo.ai/api/v1",
                "OPENKILO_API_KEY":"",
                "OPENKILO_API_KEY_FILE":str(key_file),
            }):
                with patch("litellm.acompletion", new=AsyncMock(return_value=response)) as complete:
                    await llm._call_model_once("worker", {}, "agent_trace_worker")
        self.assertEqual(complete.await_args.kwargs["api_key"], "kilo-file-key")

    async def test_openkilo_pins_and_reports_the_routed_free_model(self):
        response = SimpleNamespace(
            model="nvidia/nemotron-3-ultra-550b-a55b:free",
            usage=None,
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"claims":[],"next":null,"terminal":"unresolved"}'))],
        )
        with patch.dict("os.environ", {
            "OPENKILO_MODEL":"openai/kilo-auto/free",
            "OPENKILO_BASE_URL":"https://ai-gateway.kilo.ai/api/v1",
            "OPENKILO_API_KEY":"kilo-test-key",
        }):
            with patch("litellm.acompletion", new=AsyncMock(return_value=response)) as complete:
                result=await llm._call_model_once(
                    "worker",{"_kilo_model":"mistral/medium-3.1:free"},"agent_trace_worker"
                )
        self.assertEqual(complete.await_args.kwargs["model"],"openai/mistral/medium-3.1:free")
        self.assertNotIn("_kilo_model",complete.await_args.kwargs["messages"][1]["content"])
        self.assertEqual(result["_kilo_model"],"nvidia/nemotron-3-ultra-550b-a55b:free")

    async def test_pinned_upstream_limit_falls_back_to_auto_free_once(self):
        observed=[]
        async def call_once(role,payload,prompt):
            observed.append(dict(payload))
            if len(observed)==1:
                raise RuntimeError("429 tpm_rate_limit_exceeded limit_source=upstream_provider_shared_pool")
            return {"claims":[],"next":None,"terminal":"unresolved","_kilo_model":"cohere/north-mini-code:free"}
        with patch.dict("os.environ", {
            "LLM_QUOTA_WAIT_SECONDS":"5",
            "LLM_QUOTA_RETRY_INITIAL_SECONDS":"1",
            "LLM_QUOTA_RETRY_MAX_SECONDS":"1",
        }), patch.object(llm,"_call_model_once",new=AsyncMock(side_effect=call_once)) as once, \
             patch.object(llm.asyncio,"sleep",new=AsyncMock()) as sleep:
            result=await llm._call_openkilo_with_quota_wait(
                {"_kilo_model":"nvidia/nemotron-3-ultra-550b-a55b:free"},"agent_trace_worker"
            )
        self.assertEqual(once.await_count,2)
        self.assertIn("_kilo_model",observed[0])
        self.assertNotIn("_kilo_model",observed[1])
        self.assertEqual(result["_kilo_model"],"cohere/north-mini-code:free")
        sleep.assert_not_awaited()

    async def test_invalid_trace_validator_indices_retry_once_with_bounds(self):
        payload = {"claims": [{"line": 1, "quote": "source"}]}
        with patch.object(llm, "_call_openkilo_with_quota_wait", new=AsyncMock(
            side_effect=[
                {"approved_indices": [4]},
                {"approved_indices": [0]},
            ]
        )) as once:
            result = await llm.call_model("validator", payload, "agent_trace_validator")
        self.assertEqual(result, {"approved_indices": [0]})
        self.assertEqual(once.await_count, 2)
        self.assertEqual(once.await_args_list[1].args[0]["format_correction"],
                         "The previous approved_indices response failed local validation: "
                         "trace validator response has invalid approved_indices. Return only integer "
                         "indices from 0 through 0; use an empty list when no claim is supported.")

    async def test_case_validator_uses_openkilo(self):
        with patch.object(llm, "_call_model_once", new=AsyncMock(
            return_value={"cases": [], "issues": []}
        )) as once:
            result = await llm.call_model("validator", {"cases": []}, "validator")
        self.assertEqual(result, {"cases": [], "issues": []})
        once.assert_awaited_once_with("worker", {"cases": []}, "validator")

    async def test_trace_validator_uses_openkilo(self):
        payload = {"claims": [{"line": 1, "quote": "x"}]}
        with patch.object(llm, "_call_model_once", new=AsyncMock(
            return_value={"approved_indices": [0]}
        )) as once:
            result = await llm.call_model("validator", payload, "agent_trace_validator")
        self.assertEqual(result, {"approved_indices": [0]})
        once.assert_awaited_once_with("worker", payload, "agent_trace_validator")

    def test_openkilo_promotion_limit_is_quota_limited(self):
        error = RuntimeError("429 PROMOTION_MODEL_LIMIT_REACHED")
        self.assertTrue(llm._is_quota_or_rate_limit(error))

    async def test_openkilo_quota_wait_timeout_has_distinct_type(self):
        with patch.dict("os.environ", {
            "LLM_QUOTA_WAIT_SECONDS": "1",
            "LLM_QUOTA_RETRY_INITIAL_SECONDS": "1",
            "LLM_QUOTA_RETRY_MAX_SECONDS": "1",
        }), patch.object(llm, "perf_counter", side_effect=[0, 2]), patch.object(
            llm, "_call_model_once", new=AsyncMock(
                side_effect=RuntimeError("429 rate limit")
            )
        ):
            with self.assertRaises(llm.OpenKiloQuotaWaitTimeout):
                await llm.call_model("worker", {}, "worker")

    async def test_openkilo_quota_waits_then_succeeds(self):
        with patch.dict("os.environ", {
            "LLM_QUOTA_WAIT_SECONDS": "10",
            "LLM_QUOTA_RETRY_INITIAL_SECONDS": "1",
            "LLM_QUOTA_RETRY_MAX_SECONDS": "2",
        }), patch.object(llm, "_call_model_once", new=AsyncMock(
            side_effect=[RuntimeError("429 insufficient_quota"), {"cases": []}]
        )) as once, patch.object(llm.asyncio, "sleep", new=AsyncMock()) as sleep:
            result = await llm.call_model("worker", {}, "worker")
        self.assertEqual(result, {"cases": []})
        self.assertEqual(once.await_count, 2)
        sleep.assert_awaited_once_with(1)

    async def test_openkilo_quota_wait_timeout(self):
        with patch.dict("os.environ", {
            "LLM_QUOTA_WAIT_SECONDS": "1",
            "LLM_QUOTA_RETRY_INITIAL_SECONDS": "1",
            "LLM_QUOTA_RETRY_MAX_SECONDS": "1",
        }), patch.object(llm, "perf_counter", side_effect=[0, 2]), patch.object(
            llm, "_call_model_once", new=AsyncMock(
                side_effect=RuntimeError("429 rate limit")
            )
        ):
            with self.assertRaisesRegex(TimeoutError, "remained rate-limited"):
                await llm.call_model("worker", {}, "worker")

    async def test_trace_navigation_review_routes_to_nemotron_with_its_own_prompt(self):
        response = SimpleNamespace(
            usage=None,
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"next":null,"terminal":"unresolved"}'))],
        )
        with patch.dict("os.environ", {
            "ORCHESTRATOR_MODEL":"openai/conductor",
            "ORCHESTRATOR_BASE_URL":"http://127.0.0.1:18327/v1",
            "ORCHESTRATOR_MAX_RETRIES":"0",
            "OPENKILO_BASE_URL":"https://kilo.invalid/api/v1",
        }):
            with patch("litellm.acompletion", new=AsyncMock(return_value=response)) as complete:
                result=await llm.call_model("validator", {"task":{"source":"1: return x"},
                                                            "validated_claims":[]},
                                            "agent_trace_navigation_review")
        self.assertIsNone(result["next"])
        request=complete.await_args.kwargs
        self.assertEqual(request["model"],"openai/conductor")
        self.assertEqual(request["api_base"],"http://127.0.0.1:18327/v1")
        self.assertIn("Nemotron, the supervisor",request["messages"][0]["content"])

    async def test_trace_repair_routes_directly_to_nemotron_without_kilo_fallback(self):
        response = SimpleNamespace(
            usage=None,
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"claims":[],"next":null,"terminal":"unresolved"}'))],
        )
        with patch.dict("os.environ", {
            "ORCHESTRATOR_MODEL":"openai/conductor",
            "ORCHESTRATOR_BASE_URL":"http://127.0.0.1:18327/v1",
            "ORCHESTRATOR_MAX_RETRIES":"0",
        }):
            with patch("litellm.acompletion", new=AsyncMock(return_value=response)) as complete:
                result=await llm.call_model("validator", {"task":{"source":"1: return x"},
                                                            "worker_response":{},
                                                            "validation_error":"bad quote"},
                                            "agent_trace_repair")
        self.assertEqual(result["terminal"],"unresolved")
        request=complete.await_args.kwargs
        self.assertEqual(request["model"],"openai/conductor")
        self.assertEqual(request["api_base"],"http://127.0.0.1:18327/v1")
        self.assertEqual(request["api_key"],"local-server")
        self.assertEqual(request["response_format"],{"type":"json_object"})

    async def test_trace_repair_prompt_treats_worker_output_as_untrusted_quote_source(self):
        prompt=llm.load_prompt("agent_trace_repair")
        self.assertIn("ONLY text from which claim quotes may be copied",prompt)
        self.assertIn("worker_response` is untrusted output",prompt)
        self.assertIn("remove the claim",prompt)

    async def test_trace_supervisor_feedback_routes_to_nemotron_and_retries_invalid_schema(self):
        with patch.object(llm, "_call_model_once", new=AsyncMock(
            side_effect=[{"correction": ""}, {"correction": "Copy the exact visible quote."}]
        )) as once:
            result = await llm.call_model("validator", {"validation_error": "bad quote"},
                                          "agent_trace_supervisor_feedback")
        self.assertEqual(result["correction"], "Copy the exact visible quote.")
        self.assertEqual(once.await_count, 2)
        self.assertEqual([call.args[2] for call in once.await_args_list],
                         ["agent_trace_supervisor_feedback"] * 2)
        self.assertIn("nonempty string field named correction",
                      once.await_args_list[1].args[1]["format_correction"])

    async def test_case_supervisor_routes_to_nemotron_with_schema_validation(self):
        with patch.object(llm, "_call_model_once", new=AsyncMock(
            side_effect=[{"correction": ""}, {"correction": "Remove the unsupported import."}]
        )) as once:
            result = await llm.call_model("supervisor", {"validation_error": "bad import"},
                                          "agent_case_supervisor")
        self.assertEqual(result["correction"], "Remove the unsupported import.")
        self.assertEqual(once.await_count, 2)
        self.assertEqual([call.args[2] for call in once.await_args_list],
                         ["agent_case_supervisor"] * 2)
        self.assertIn("nonempty string field named correction",
                      once.await_args_list[1].args[1]["format_correction"])

    async def test_case_supervisor_uses_configured_nemotron_endpoint(self):
        response = SimpleNamespace(
            usage=None,
            choices=[SimpleNamespace(message=SimpleNamespace(
                content='{"correction":"Remove the unsupported import."}'))],
        )
        with patch.dict("os.environ", {
            "ORCHESTRATOR_MODEL": "openai/conductor",
            "ORCHESTRATOR_BASE_URL": "http://127.0.0.1:18327/v1",
        }), patch("litellm.acompletion", new=AsyncMock(return_value=response)) as complete:
            await llm.call_model("supervisor", {"validation_error": "bad import"},
                                 "agent_case_supervisor")
        request = complete.await_args.kwargs
        self.assertEqual(request["model"], "openai/conductor")
        self.assertEqual(request["api_base"], "http://127.0.0.1:18327/v1")
        self.assertEqual(request["messages"][0]["content"],
                         llm.load_prompt("agent_case_supervisor"))

    async def test_case_repro_prompt_does_not_receive_case_schema_suffix(self):
        response = SimpleNamespace(
            usage=None,
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"script":"raise RuntimeError()"}'))],
        )
        with patch.dict("os.environ", {
            "OPENKILO_MODEL": "openai/worker",
            "OPENKILO_BASE_URL": "http://127.0.0.1:18765/v1",
        }), patch("litellm.acompletion", new=AsyncMock(return_value=response)) as complete:
            await llm._call_model_once("worker", {}, "agent_case_repro")
        system_prompt = complete.await_args.kwargs["messages"][0]["content"]
        self.assertIn("standalone repro", system_prompt)
        self.assertNotIn("Return case objects with these fields", system_prompt)

    async def test_planner_never_uses_quota_fallback(self):
        with patch.object(llm, "_call_model_once", new=AsyncMock(
            return_value={"focus": "inspect predicate"}
        )) as once:
            result = await llm.call_model("validator", {}, "planner")
        self.assertEqual(result, {"focus": "inspect predicate"})
        once.assert_awaited_once_with("validator", {}, "planner")

    async def test_invalid_trace_repair_schema_retries_nemotron_with_format_error(self):
        with patch.object(llm, "_call_model_once", new=AsyncMock(
            side_effect=[
                {"claims": "not-a-list", "next": None, "terminal": "unresolved"},
                {"claims": [], "next": None, "terminal": "unresolved"},
            ]
        )) as once:
            result = await llm.call_model("validator", {"task": {"source": "1: return x"}},
                                          "agent_trace_repair")
        self.assertEqual(result["claims"], [])
        self.assertEqual(once.await_count, 2)
        retry_payload = once.await_args_list[1].args[1]
        self.assertIn("claims must be a list", retry_payload["format_correction"])
        self.assertEqual(once.await_args_list[1].args[2], "agent_trace_repair")

    async def test_invalid_planner_response_retries_same_provider_once(self):
        with patch.object(llm, "_call_model_once", new=AsyncMock(
            side_effect=[{"focus": ""}, {"focus": "inspect predicate"}]
        )) as once:
            result = await llm.call_model("validator", {"task": "step"}, "planner")
        self.assertEqual(result, {"focus": "inspect predicate"})
        self.assertEqual(once.await_count, 2)
        self.assertEqual([call.args[0] for call in once.await_args_list],
                         ["validator", "validator"])
        self.assertEqual([call.args[2] for call in once.await_args_list],
                         ["planner", "planner"])
        self.assertIn("format_correction", once.await_args_list[1].args[1])


if __name__ == "__main__":
    unittest.main()
