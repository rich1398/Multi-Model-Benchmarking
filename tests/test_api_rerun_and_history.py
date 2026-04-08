import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

import app as app_module
from models import AppConfig, JudgeConfig, ProviderConfig


class DummyClient:
    async def preflight_check(self, **kwargs):
        return {"ok": True, "errors": [], "checks": [{"scope": "base_model", "ok": True}]}

    async def list_ollama_model_details(self):
        return [{"name": "llama3.2", "chat_capable": True, "reason": ""}]


class ApiRerunAndHistoryTests(unittest.TestCase):
    def setUp(self) -> None:
        app_module._config = AppConfig(
            models=(
                ProviderConfig(
                    name="llama3.2",
                    base_url="http://localhost:11434",
                    provider="ollama",
                ),
            ),
            judge=JudgeConfig(backend="ollama", model="llama3.2"),
            tasks_file="tasks/core_tasks.json",
        )
        app_module._client = DummyClient()
        app_module._active_runs.clear()
        app_module._run_queues.clear()
        app_module._cancel_events.clear()
        self.client = TestClient(app_module.app)

    def tearDown(self) -> None:
        self.client.close()
        app_module._active_runs.clear()
        app_module._run_queues.clear()
        app_module._cancel_events.clear()

    def test_rerun_failed_creates_new_run_for_only_failed_cells(self):
        app_module._active_runs["orig1234"] = {
            "status": "completed_with_failures",
            "started_at": 1.0,
            "total": 2,
            "completed": 2,
            "results": [
                {
                    "task_id": "T1",
                    "task_prompt": "Broken prompt",
                    "task_rubric": "Must work",
                    "task_category": "testing",
                    "task_difficulty": "medium",
                    "pipeline_id": "single",
                    "ok": False,
                    "error": "timeout",
                },
                {
                    "task_id": "T2",
                    "task_prompt": "Good prompt",
                    "task_rubric": "Must work",
                    "task_category": "testing",
                    "task_difficulty": "medium",
                    "pipeline_id": "single",
                    "ok": True,
                    "score": 9,
                },
            ],
            "summary": {
                "total_results": 2,
                "success_count": 1,
                "failure_count": 1,
                "judge_failure_count": 0,
            },
            "config": {
                "model": "llama3.2",
                "task_suite": "core_tasks.json",
                "role_models": {},
                "judge_model": "llama3.2",
                "judge_backend": "ollama",
                "tasks": [
                    {
                        "id": "T1",
                        "prompt": "Broken prompt",
                        "rubric": "Must work",
                        "category": "testing",
                        "difficulty": "medium",
                    },
                    {
                        "id": "T2",
                        "prompt": "Good prompt",
                        "rubric": "Must work",
                        "category": "testing",
                        "difficulty": "medium",
                    },
                ],
            },
        }

        scheduled = []

        def fake_create_task(coro):
            scheduled.append(coro)
            coro.close()
            return None

        with patch("app.asyncio.create_task", side_effect=fake_create_task):
            response = self.client.post("/api/rerun-failed/orig1234")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["total"], 1)
        self.assertTrue(scheduled)

        new_run = app_module._active_runs[payload["run_id"]]
        self.assertEqual(new_run["total"], 1)
        self.assertTrue(new_run["config"]["rerun_failed_only"])
        self.assertEqual(new_run["config"]["rerun_of"], "orig1234")
        self.assertEqual(new_run["config"]["tasks"][0]["id"], "T1")

    def test_results_list_fallback_exposes_summary_metrics(self):
        app_module._active_runs["run5678"] = {
            "status": "completed_with_failures",
            "started_at": 10.0,
            "total": 2,
            "completed": 2,
            "results": [
                {"task_id": "T1", "pipeline_id": "single", "ok": True, "score": 8, "judge_ok": True},
                {"task_id": "T2", "pipeline_id": "single", "ok": False, "score": 0, "judge_ok": True},
            ],
        }

        with patch("db.list_runs", new=AsyncMock(side_effect=RuntimeError("db unavailable"))):
            response = self.client.get("/api/results")

        self.assertEqual(response.status_code, 200)
        run = response.json()["runs"][0]
        self.assertEqual(run["average_score_all_tasks"], 4.0)
        self.assertEqual(run["success_rate"], 0.5)
        self.assertEqual(run["failure_count"], 1)

    def test_export_uses_historical_run_when_not_active(self):
        historical = {
            "run_id": "hist1234",
            "status": "completed",
            "config": {"model": "llama3.2"},
            "summary": {"average_score_all_tasks": 8.0},
            "results": [
                {
                    "task_id": "T1",
                    "pipeline_id": "single",
                    "ok": True,
                    "score": 8,
                    "judge_ok": True,
                    "judge_backend": "ollama",
                    "judge_model": "llama3.2",
                    "judge_reasoning": "Good",
                    "error": "",
                    "judge_error": "",
                    "wall_ms": 120,
                    "llm_calls": 1,
                    "total_tokens": 42,
                }
            ],
        }

        with patch("db.get_run_results", new=AsyncMock(return_value=historical)):
            response = self.client.get("/api/export/hist1234?format=json")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["run_id"], "hist1234")
        self.assertEqual(payload["summary"]["average_score_all_tasks"], 8.0)
        self.assertEqual(payload["results"][0]["judge_model"], "llama3.2")
