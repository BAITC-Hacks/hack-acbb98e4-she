"""Behavioral checks against a small independent public-API environment."""

import unittest
from unittest.mock import patch

import pandas as pd

from agent import Agent
from plan_validation import audit_plan


class PublicTestEnvironment:
    """No organizer effect tables; explicit counterfactuals for behavioral tests."""

    def __init__(self, winner="offer_b", budget=100000, contacts=15000):
        self.customer_profile = pd.DataFrame({
            "ID_NUMBER": range(600),
            "current_tariff": ["original"] * 600,
            "arpu_segment": ["HIGH"] * 600,
            "data_segment": ["HEAVY"] * 600,
            "call_segment": ["HIGH"] * 600,
            "predicted_arpu": [8000.0] * 600,
        })
        self.tariffs = pd.DataFrame({
            "tariff_plan_code": ["original", "offer_b", "offer_c"],
            "price_tariff": [4000, 6000, 7000],
        })
        self.channels = {
            "push": {"cost_per_contact": 0, "conversion_multiplier": 0.50},
            "sms": {"cost_per_contact": 4, "conversion_multiplier": 0.65},
            "digital_ads": {"cost_per_contact": 22, "conversion_multiplier": 0.85},
            "call": {"cost_per_contact": 160, "conversion_multiplier": 1.20},
        }
        self.remaining_budget = budget
        self.remaining_contacts = contacts
        self.pilots_left = 20
        self.pilot_history = []
        self.winner = winner

    def run_pilot(self, target_tariff, channel, n_customers=100, **filters):
        assert self.pilots_left > 0
        assert 10 <= n_customers <= 200
        assert target_tariff in set(self.tariffs.tariff_plan_code)
        frame = self.customer_profile
        for key, value in filters.items():
            if value is None:
                continue
            field = key.removeprefix("filter_")
            frame = frame[frame[field].isin(str(value).split(";"))]
        n = min(n_customers, len(frame))
        assert n >= 10
        cost = n * self.channels[channel]["cost_per_contact"]
        assert cost <= self.remaining_budget
        assert n <= self.remaining_contacts
        self.remaining_budget -= cost
        self.remaining_contacts -= n
        self.pilots_left -= 1
        lift = (0.5 if target_tariff == self.winner else -0.5)
        lift *= self.channels[channel]["conversion_multiplier"]
        result = {
            "pilot": len(self.pilot_history) + 1,
            "target_tariff": target_tariff,
            "channel": channel,
            "observed_lift_ratio": lift,
            "observed_lift_total": lift * n * 8000,
            "n_customers": n,
            "cost": cost,
            "remaining_budget": self.remaining_budget,
            "remaining_contacts": self.remaining_contacts,
        }
        self.pilot_history.append(result)
        return result


class AgentBehaviorTests(unittest.TestCase):
    def assert_valid(self, env, campaigns):
        report = audit_plan(campaigns, env.customer_profile, env.tariffs, env.channels,
                            env.remaining_budget, env.remaining_contacts)
        self.assertTrue(report["valid"], report["errors"])
        self.assertGreater(len(env.pilot_history), 0)
        self.assertLessEqual(len(env.pilot_history), 20)

    def test_pilot_outcome_changes_target(self):
        for winner in ("offer_b", "offer_c"):
            with self.subTest(winner=winner):
                env = PublicTestEnvironment(winner=winner)
                campaigns = Agent().act(env)
                self.assert_valid(env, campaigns)
                self.assertTrue(any(c["target_tariff"] == winner for c in campaigns))
                self.assertFalse(any(c["target_tariff"] != winner for c in campaigns))

    def test_zero_money_can_use_free_channel(self):
        env = PublicTestEnvironment(budget=0)
        campaigns = Agent().act(env)
        self.assert_valid(env, campaigns)
        self.assertTrue(all(c["channel"] == "push" for c in campaigns))

    def test_same_observations_produce_same_plan(self):
        self.assertEqual(Agent().act(PublicTestEnvironment()),
                         Agent().act(PublicTestEnvironment()))

    def test_revenue_loss_is_not_disguised_by_small_customer_growth(self):
        # Nine small gains (+5400 total) cannot offset one large loss (-50000).
        # The unweighted average relative change is nevertheless positive.
        history = pd.DataFrame({
            "AVG_ARPU_PREV_3M": [6000] * 9 + [100000],
            "AVG_ARPU_NEXT_3M": [6600] * 9 + [50000],
            "tariff_plan_code_from": ["original"] * 10,
            "tariff_plan_code_to": ["offer_b"] * 10,
        })
        with patch("agent.pd.read_csv", return_value=history):
            pairs, _ = Agent()._historical_ranks(["original", "offer_b"])
        self.assertLess(pairs[("original", "HIGH", "offer_b")][0], 0)

    def test_bad_pilots_stop_without_spending_all_experiments(self):
        env = PublicTestEnvironment(winner="no_profitable_offer")
        agent = Agent()
        campaigns = agent.act(env)
        self.assert_valid(env, campaigns)
        self.assertLess(len(env.pilot_history), 20)
        self.assertTrue(agent.last_report["decisions"][0]["unverified_fallback"])


if __name__ == "__main__":
    unittest.main()
