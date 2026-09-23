"""Проверки аудита на маленьких примерах с вручную известными ответами."""

import json
import unittest

import pandas as pd

from plan_validation import audit_plan


class PlanValidationTests(unittest.TestCase):
    def setUp(self):
        self.profile = pd.DataFrame({
            "ID_NUMBER": [1, 2, 3, 4],
            "current_tariff": ["tariff_1", "tariff_1", "tariff_2", "tariff_2"],
            "arpu_segment": ["LOW", "LOW", "MID", "HIGH"],
            "data_segment": ["LITE", "HEAVY", "LITE", "NON_USER"],
            "call_segment": ["LOW", "MEDIUM", "HIGH", "LOW"],
            "predicted_arpu": [500, 800, 2000, 6000],
        })
        self.tariffs = pd.DataFrame({"tariff_plan_code": ["tariff_1", "tariff_2", "tariff_3"]})
        self.channels = {
            "push": {"cost_per_contact": 0, "conversion_multiplier": 0.5},
            "sms": {"cost_per_contact": 4, "conversion_multiplier": 0.65},
        }

    def audit(self, campaigns, budget=100, contacts=100, profile=None):
        return audit_plan(campaigns, self.profile if profile is None else profile,
                          self.tariffs, self.channels, budget, contacts)

    def campaign(self, **updates):
        result = {"target_tariff": "tariff_3", "channel": "sms", "filter_arpu_segment": "LOW"}
        result.update(updates)
        return result

    def test_actual_audience_and_cost(self):
        result = self.audit([self.campaign(filter_data_segment="HEAVY", filter_call_segment="MEDIUM")])
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(result["total_contacts"], 1)
        self.assertEqual(result["unique_contacts"], 1)
        self.assertEqual(result["communication_cost"], 4.0)

    def test_remaining_limits_after_pilots_are_used(self):
        for budget, contacts, error_fragment in [(7, 2, "бюджета после пилотов"), (8, 1, "контактов")]:
            with self.subTest(budget=budget, contacts=contacts):
                result = self.audit([self.campaign()], budget=budget, contacts=contacts)
                self.assertFalse(result["valid"])
                self.assertTrue(any(error_fragment in error for error in result["errors"]))
        self.assertTrue(self.audit([self.campaign()], budget=8, contacts=2)["valid"])

    def test_overlapping_campaigns_spend_contacts_twice(self):
        result = self.audit([self.campaign(), self.campaign()], budget=16, contacts=3)
        self.assertFalse(result["valid"])
        self.assertEqual(result["total_contacts"], 4)
        self.assertEqual(result["unique_contacts"], 2)
        self.assertEqual(result["communication_cost"], 16.0)

    def test_invalid_tariff_channel_and_filters(self):
        bad_fields = [
            {"target_tariff": "missing"},
            {"channel": "email"},
            {"filter_arpu_segment": "AVERAGE"},
            {"filter_data_segment": "UNKNOWN"},
            {"filter_call_segment": "MID"},
            {"filter_current_tariff": "tariff_1;missing"},
            {"filter_arpu_segment": "LOW;MID"},
            {"filter_arpu_segment": ["LOW"]},
        ]
        for update in bad_fields:
            with self.subTest(update=update):
                self.assertFalse(self.audit([self.campaign(**update)])["valid"])

    def test_multiple_current_tariffs_and_missing_filters(self):
        for value in [None, float("nan")]:
            with self.subTest(value=value):
                result = self.audit([self.campaign(filter_arpu_segment=value,
                                                   filter_current_tariff="tariff_1; tariff_2")])
                self.assertTrue(result["valid"], result["errors"])
                self.assertEqual(result["total_contacts"], 4)
                json.dumps(result, allow_nan=False)
        no_filter = {"target_tariff": "tariff_3", "channel": "push"}
        result = self.audit([no_filter], budget=0)
        self.assertTrue(result["valid"])
        self.assertEqual(result["total_contacts"], 4)
        self.assertEqual(result["communication_cost"], 0.0)

    def test_empty_strings_and_empty_audiences_are_errors(self):
        for update in [{"filter_arpu_segment": ""}, {"filter_current_tariff": " "},
                       {"filter_current_tariff": "tariff_1;"}, {"filter_data_segment": "NON_USER"}]:
            with self.subTest(update=update):
                self.assertFalse(self.audit([self.campaign(**update)])["valid"])

    def test_unknown_fields_and_explicit_ids_are_rejected(self):
        for update in [{"explicit_ids": [1]}, {"notes": "ignored?"}]:
            with self.subTest(update=update):
                result = self.audit([self.campaign(**update)])
                self.assertFalse(result["valid"])
                self.assertIn("Недопустимые поля", result["errors"][0])

    def test_per_campaign_boundary_5000(self):
        for size, valid in [(5000, True), (5001, False)]:
            with self.subTest(size=size):
                profile = pd.concat([self.profile.iloc[[0]]] * size, ignore_index=True)
                profile["ID_NUMBER"] = range(size)
                result = self.audit([self.campaign(channel="push")], contacts=10000, profile=profile)
                self.assertEqual(result["valid"], valid, result["errors"])
                self.assertEqual(result["total_contacts"], size)

    def test_plan_shape_and_campaign_count(self):
        for campaigns in [None, {}, [], [None], [self.campaign()] * 11]:
            with self.subTest(campaigns=campaigns):
                self.assertFalse(self.audit(campaigns)["valid"])
        self.assertTrue(self.audit([self.campaign()] * 10)["valid"])

    def test_invalid_context_is_reported_without_nonfinite_json(self):
        for budget, contacts in [(float("nan"), 5), (float("inf"), 5), (100, -1), (100, 1.5)]:
            result = self.audit([self.campaign()], budget=budget, contacts=contacts)
            self.assertFalse(result["valid"])
            json.dumps(result, allow_nan=False)

    def test_extreme_costs_stay_json_safe(self):
        self.channels["sms"]["cost_per_contact"] = 1e308
        result = self.audit([self.campaign()])
        self.assertFalse(result["valid"])
        self.assertIsNone(result["communication_cost"])
        json.dumps(result, allow_nan=False)

    def test_inputs_are_not_mutated(self):
        original = self.profile.copy(deep=True)
        campaign = self.campaign()
        copy = dict(campaign)
        self.audit([campaign])
        pd.testing.assert_frame_equal(self.profile, original)
        self.assertEqual(campaign, copy)


if __name__ == "__main__":
    unittest.main()
