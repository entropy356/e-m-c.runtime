"""CLI tests: configuration validation (T15) and end-to-end smoke (T17)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from .helpers import REPO_ROOT, TempDirTestCase, write_json

ENV = {**os.environ, "PYTHONPATH": f"{REPO_ROOT / 'src'}{os.pathsep}{REPO_ROOT}"}


def run_cli(argv, cwd=REPO_ROOT):
    return subprocess.run(
        [sys.executable, "-m", "emc_runtime", *argv],
        capture_output=True,
        text=True,
        env=ENV,
        cwd=str(cwd),
        timeout=60,
    )


class ConfigValidationTest(TempDirTestCase):
    """T15: broken configurations fail at startup with readable errors."""

    def _config(self, **overrides):
        data = {
            "schema_version": 1,
            "database": str(self.tmp / "agent.sqlite3"),
            "provider": {"kind": "offline"},
            "cognition": {"kind": "direct"},
            "modules": [],
        }
        data.update(overrides)
        return write_json(self.tmp / "config.json", data)

    def _run(self, **overrides):
        return run_cli(["--config", str(self._config(**overrides)), "turn", "hi"])

    def test_bad_schema_version(self):
        result = self._run(schema_version=99)
        self.assertEqual(result.returncode, 1)
        self.assertIn("Error:", result.stderr)
        self.assertIn("schema_version", result.stderr)

    def test_duplicate_module_ids(self):
        module = {
            "id": "counter",
            "entrypoint": "examples.counter_module:CounterModule",
            "scope": "agent",
            "config": {},
        }
        result = self._run(modules=[module, module])
        self.assertEqual(result.returncode, 1)
        self.assertIn("counter", result.stderr)

    def test_broken_entrypoint_format(self):
        result = self._run(
            modules=[
                {"id": "bad", "entrypoint": "no-colon-here", "scope": "agent", "config": {}}
            ]
        )
        self.assertEqual(result.returncode, 1)

    def test_unknown_provider_kind(self):
        result = self._run(provider={"kind": "telepathy"})
        self.assertEqual(result.returncode, 1)
        self.assertIn("telepathy", result.stderr)

    def test_unknown_module_class(self):
        result = self._run(
            modules=[
                {
                    "id": "ghost",
                    "entrypoint": "examples.counter_module:Missing",
                    "scope": "agent",
                    "config": {},
                }
            ]
        )
        self.assertEqual(result.returncode, 1)

    def test_scope_mismatch(self):
        result = self._run(
            modules=[
                {
                    "id": "fixed",
                    "entrypoint": "tests.helpers:FixedAgentScopeModule",
                    "scope": "conversation",  # module declares agent
                    "config": {},
                }
            ]
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("scope", result.stderr)


class CliSmokeTest(TempDirTestCase):
    """T17: the documented commands work end-to-end (offline provider)."""

    def setUp(self):
        super().setUp()
        self.config_path = self.tmp / "counter.json"
        data = json.loads((REPO_ROOT / "examples" / "counter.json").read_text("utf-8"))
        data["database"] = str(self.tmp / "agent.sqlite3")
        write_json(self.config_path, data)
        self.scenario_path = write_json(
            self.tmp / "scenario.json",
            {
                "name": "cli-smoke",
                "conversation_id": "default",
                "events": [
                    {"kind": "user_message", "content": "场景一"},
                    {"kind": "internal_trigger", "metadata": {"reason": "smoke"}},
                    {
                        "kind": "user_message",
                        "content": "另一个会话",
                        "conversation_id": "second",
                    },
                ],
            },
        )

    def test_full_cli_flow(self):
        cfg = ["--config", str(self.config_path)]

        first = run_cli(cfg + ["turn", "你好", "--conversation", "c1"])
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertIn("[offline]", first.stdout)

        second = run_cli(["--config", str(self.config_path), "--json", "turn", "再来一句", "--conversation", "c1"])
        self.assertEqual(second.returncode, 0, second.stderr)
        payload = json.loads(second.stdout)
        self.assertEqual(payload["event_kind"], "user_message")
        self.assertEqual(payload["module_traces"]["counter"]["prepare"]["count_after_increment"], 2)

        history = run_cli(["--config", str(self.config_path), "--json", "history", "--conversation", "c1"])
        self.assertEqual(history.returncode, 0, history.stderr)
        messages = json.loads(history.stdout)
        self.assertEqual(
            [m["role"] for m in messages], ["user", "assistant", "user", "assistant"]
        )

        inspect = run_cli(["--config", str(self.config_path), "--json", "inspect", "--conversation", "c1"])
        self.assertEqual(inspect.returncode, 0, inspect.stderr)
        report = json.loads(inspect.stdout)
        self.assertEqual(report["revision"], 2)
        counter_view = next(
            m for m in report["modules"] if m["module_id"] == "counter"
        )
        self.assertEqual(counter_view["state"], {"count": 2})
        self.assertEqual(len(report["recent_turns"]), 2)

        trigger = run_cli(cfg + ["trigger", "--conversation", "c1", "--reason", "tick"])
        self.assertEqual(trigger.returncode, 0, trigger.stderr)
        # the trigger turn must not add a user message
        history2 = json.loads(
            run_cli(["--config", str(self.config_path), "--json", "history", "--conversation", "c1"]).stdout
        )
        self.assertEqual(
            [m["role"] for m in history2],
            ["user", "assistant", "user", "assistant", "assistant"],
        )

        scenario = run_cli(cfg + ["scenario", str(self.scenario_path)])
        self.assertEqual(scenario.returncode, 0, scenario.stderr)
        self.assertIn("cli-smoke", scenario.stdout)

        final = json.loads(run_cli(["--config", str(self.config_path), "--json", "inspect", "--conversation", "default"]).stdout)
        # scenario: 2 turns in default (incl. 1 trigger), 1 turn in second
        final_counter = next(m for m in final["modules"] if m["module_id"] == "counter")
        self.assertEqual(final_counter["state"], {"count": 6})
        self.assertEqual(len(final["recent_turns"]), 2)

    def test_offline_installation_of_examples(self):
        result = run_cli(["--config", str(REPO_ROOT / "examples" / "basic.json"), "turn", "hi", "--conversation", "smoke"])
        # database would land in repo data/; point it back by running with cwd
        # elsewhere is impossible for relative paths, so accept the repo data
        # directory but ensure the command itself succeeds.
        self.assertIn("[offline]", result.stdout)
        data_dir = REPO_ROOT / "data"
        if data_dir.exists():
            for item in data_dir.glob("basic.sqlite3*"):
                item.unlink()
            if not any(data_dir.iterdir()):
                data_dir.rmdir()


if __name__ == "__main__":
    unittest.main()
