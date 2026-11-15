import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from optimization_explorer.llm import call_model, load_prompt


class ConductorConfigurationTests(unittest.IsolatedAsyncioTestCase):
    async def test_local_planner_defaults(self):
        self.assertIn('"focus"', load_prompt('planner'))
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=json.dumps({'focus': 'Inspect dispatch predicates'})))], usage=None)
        completion = AsyncMock(return_value=response)
        with patch.dict('sys.modules', {'litellm': SimpleNamespace(acompletion=completion)}):
            with patch.dict('os.environ', {'ORCHESTRATOR_BASE_URL': 'http://127.0.0.1:18327/v1'}, clear=True):
                plan = await call_model('validator', {'task': {'context': 'if fuse: return fast_path'}}, 'planner')
        self.assertEqual(plan['focus'], 'Inspect dispatch predicates')
        kwargs = completion.await_args.kwargs
        self.assertEqual(kwargs['model'], 'openai/conductor')
        self.assertEqual(kwargs['api_base'], 'http://127.0.0.1:18327/v1')
        self.assertEqual(kwargs['max_tokens'], 512)
        self.assertEqual(kwargs['api_key'], 'local-server')
        self.assertEqual(kwargs['response_format'], {'type': 'json_object'})
        self.assertEqual(kwargs['extra_body'], {'chat_template_kwargs': {'enable_thinking': False}})


if __name__ == '__main__':
    unittest.main()
