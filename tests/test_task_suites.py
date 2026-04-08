import json
import unittest
from pathlib import Path


class TaskSuitesTests(unittest.TestCase):
    def test_benchmark_suites_are_structurally_valid(self):
        base = Path("tasks")
        suites = ["smoke_tasks.json", "core_tasks.json", "stress_tasks.json"]

        for filename in suites:
            with self.subTest(filename=filename):
                data = json.loads((base / filename).read_text(encoding="utf-8"))
                self.assertIsInstance(data, list)
                self.assertTrue(data)

                ids = []
                for task in data:
                    self.assertIn("id", task)
                    self.assertIn("prompt", task)
                    self.assertIn("rubric", task)
                    self.assertIn("category", task)
                    self.assertIn("difficulty", task)
                    self.assertTrue(str(task["id"]).strip())
                    self.assertTrue(str(task["prompt"]).strip())
                    self.assertTrue(str(task["rubric"]).strip())
                    self.assertTrue(str(task["category"]).strip())
                    self.assertTrue(str(task["difficulty"]).strip())
                    ids.append(task["id"])

                self.assertEqual(len(ids), len(set(ids)))
