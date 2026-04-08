import unittest

from role_assigner import auto_assign_roles


class RoleAssignerTests(unittest.TestCase):
    def test_gemini_is_not_default_reviewer_when_openai_or_claude_are_available(self):
        assignment = auto_assign_roles([
            {"provider": "anthropic", "model": "claude-sonnet-4-20250514"},
            {"provider": "openai", "model": "gpt-4o"},
            {"provider": "gemini", "model": "gemini-2.5-flash"},
        ])

        self.assertEqual(assignment["primary_model"], "claude-sonnet-4-20250514")
        self.assertEqual(assignment["role_models"]["generator"], "claude-sonnet-4-20250514")
        self.assertEqual(assignment["role_models"]["synthesizer"], "claude-sonnet-4-20250514")
        self.assertEqual(assignment["role_models"]["reviewer"], "gpt-4o")
        self.assertEqual(assignment["role_models"]["critic"], "gemini-2.5-flash")

    def test_gemini_is_last_resort_for_reviewer_role(self):
        assignment = auto_assign_roles([
            {"provider": "openai", "model": "gpt-4o"},
            {"provider": "gemini", "model": "gemini-2.5-flash"},
        ])

        self.assertEqual(assignment["primary_model"], "gpt-4o")
        self.assertEqual(assignment["role_models"]["reviewer"], "gpt-4o")
        self.assertEqual(assignment["role_models"]["critic"], "gemini-2.5-flash")


if __name__ == "__main__":
    unittest.main()
