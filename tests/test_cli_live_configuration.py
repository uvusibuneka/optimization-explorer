import unittest
from optimization_explorer.cli import missing_live_configuration


class LiveConfigurationTests(unittest.TestCase):
    def test_local_validator_proxy_does_not_require_remote_groq_key(self):
        env = {
            "OPENKILO_MODEL": "openai/worker",
            "OPENKILO_BASE_URL": "http://127.0.0.1:18765/v1",
            "GROQ_MODEL": "openai/validator",
            "GROQ_BASE_URL": "http://127.0.0.1:18766/v1",
            "ORCHESTRATOR_BASE_URL": "http://127.0.0.1:18327/v1",
        }
        self.assertEqual(missing_live_configuration(env, "conductor"), [])

    def test_direct_groq_endpoint_still_requires_key(self):
        env = {
            "OPENKILO_MODEL": "openai/worker",
            "OPENKILO_BASE_URL": "http://127.0.0.1:18765/v1",
            "GROQ_MODEL": "groq/openai/gpt-oss-120b",
            "GROQ_BASE_URL": "https://api.groq.com/openai/v1",
        }
        self.assertEqual(missing_live_configuration(env, "fixed"), ["GROQ_API_KEY"])


if __name__ == "__main__":
    unittest.main()
