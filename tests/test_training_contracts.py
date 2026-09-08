"""Tests for the model-completion to intent contract."""
from __future__ import annotations

import unittest

from training.contracts import IntentParseError, parse_intent_completion


class ParseIntentCompletionTests(unittest.TestCase):
    def test_parses_reasoning_followed_by_intent_tag(self):
        result = parse_intent_completion(
            "<reasoning>The notch is too low.</reasoning>\n"
            '<intent>{"ro":"decrease_strong"}</intent>'
        )
        self.assertEqual(result.reasoning, "The notch is too low.")
        self.assertEqual(result.intent, {"ro": "decrease_strong"})

    def test_parses_fenced_json(self):
        result = parse_intent_completion(
            "Move the resonance upward.\n```json\n"
            '{"ro": "decrease_slight", "alpha": "hold"}\n```'
        )
        self.assertEqual(
            result.intent,
            {"ro": "decrease_slight", "alpha": "hold"},
        )

    def test_parses_bare_json(self):
        result = parse_intent_completion('{"Wf":"increase"}')
        self.assertEqual(result.intent, {"Wf": "increase"})

    def test_rejects_unknown_variable(self):
        with self.assertRaisesRegex(IntentParseError, "unknown variable"):
            parse_intent_completion('{"radius":"decrease"}')

    def test_rejects_unknown_token(self):
        with self.assertRaisesRegex(IntentParseError, "invalid token"):
            parse_intent_completion('{"ro":"smaller"}')

    def test_rejects_numeric_parameter_value(self):
        with self.assertRaisesRegex(IntentParseError, "invalid token"):
            parse_intent_completion('{"ro":5.3}')

    def test_rejects_empty_object(self):
        with self.assertRaisesRegex(IntentParseError, "at least one"):
            parse_intent_completion("{}")

    def test_rejects_all_hold_action(self):
        with self.assertRaisesRegex(IntentParseError, "all-hold"):
            parse_intent_completion('{"ro":"hold","alpha":"hold"}')

    def test_rejects_more_than_two_active_variables(self):
        with self.assertRaisesRegex(IntentParseError, "at most two"):
            parse_intent_completion(
                '{"ro":"decrease","ri":"increase","alpha":"decrease"}'
            )

    def test_rejects_text_without_json(self):
        with self.assertRaisesRegex(IntentParseError, "JSON"):
            parse_intent_completion("Decrease the radius strongly.")


if __name__ == "__main__":
    unittest.main()
