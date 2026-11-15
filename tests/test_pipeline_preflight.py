import importlib.util
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "pipeline_preflight.py"
spec = importlib.util.spec_from_file_location("pipeline_preflight", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PreflightTests(unittest.TestCase):
    def test_missing_inputs_and_secrets_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = dict((label, status) for status, label in module.check(Path(tmp), {}))
        self.assertEqual(results["results/candidate_snippets.jsonl"], "MISSING")
        self.assertEqual(results["GROQ_API_KEY"], "UNSET")

    def test_present_inputs_and_env_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tools").mkdir()
            (root / "tools/first_cases.py").touch()
            (root / "results").mkdir()
            (root / "results/candidate_snippets.jsonl").touch()
            (root / "decompiled").mkdir()
            (root / "binaries").mkdir()
            env = {key: "configured" for key in (
                "OPENKILO_API_KEY", "OPENKILO_BASE_URL", "OPENKILO_MODEL",
                "GROQ_API_KEY", "GROQ_MODEL")}
            results = dict((label, status) for status, label in module.check(root, env))
        self.assertEqual(results["OPENKILO_API_KEY"], "OK")
        self.assertEqual(results["decompiled/"], "OK")


if __name__ == "__main__":
    unittest.main()
