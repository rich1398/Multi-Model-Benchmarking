import unittest

from app import _benchmark_mode, _final_status, _summarize_results
from db import _normalized_status


class MetricsTests(unittest.TestCase):
    def test_summary_counts_failures_and_judge_failures_honestly(self):
        results = [
            {"ok": True, "score": 8, "judge_ok": True},
            {"ok": False, "score": 0, "judge_ok": True},
            {"ok": True, "score": 6, "judge_ok": False},
        ]

        summary = _summarize_results(results)

        self.assertEqual(summary["total_results"], 3)
        self.assertEqual(summary["success_count"], 2)
        self.assertEqual(summary["failure_count"], 1)
        self.assertEqual(summary["judge_failure_count"], 1)
        self.assertEqual(summary["average_score_all_tasks"], 4.67)
        self.assertEqual(summary["average_score_success_only"], 7.0)
        self.assertFalse(summary["valid_for_thesis"])
        self.assertEqual(_final_status(results), "completed_with_failures")

    def test_embedding_run_is_reclassified_as_invalid_config(self):
        summary = {
            "success_count": 0,
            "failure_count": 2,
            "judge_failure_count": 0,
        }
        results = [
            {"ok": False, "error": "model does not support chat"},
            {"ok": False, "error": "model does not support chat"},
        ]

        status, reason = _normalized_status(
            "completed",
            {"model": "nomic-embed-text:latest"},
            summary,
            results,
        )

        self.assertEqual(status, "invalid_config")
        self.assertIn("not suitable for chat benchmarks", reason)

    def test_benchmark_mode_marks_role_overrides_as_multi_model(self):
        self.assertEqual(
            _benchmark_mode(
                "llama3.2",
                {},
                "ollama",
                "llama3.2",
                "ollama",
            ),
            "single_model_orchestration",
        )
        self.assertEqual(
            _benchmark_mode(
                "llama3.2",
                {"critic": "mistral"},
                "ollama",
                "llama3.2",
                "ollama",
            ),
            "multi_model",
        )
