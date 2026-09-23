"""Independent resilience scenarios using only the participant-facing API.

These synthetic observations test behavior, not hidden judging effects or a
future score. No organizer modules, agent methods, or plan validator are patched.
Run: python -m unittest discover -s tests -p test_agent_resilience.py -v
"""

import math
import unittest

import pandas as pd

from agent import Agent


class ResilienceEnvironment:
    """One homogeneous 600-person audience with explicit scripted observations."""

    ORIGINAL = "resilience_original"
    OFFER_A = "resilience_offer_a"
    OFFER_B = "resilience_offer_b"
    SIZE = 600
    ARPU = 8000.0
    BUDGET = 100000
    CONTACTS = 15000
    PILOTS = 20
    FILTERS = {
        "filter_current_tariff": "current_tariff",
        "filter_arpu_segment": "arpu_segment",
        "filter_data_segment": "data_segment",
        "filter_call_segment": "call_segment",
    }

    def __init__(self, *, fail_second=False, expensive=False, repeat_mode=None):
        self.customer_profile = pd.DataFrame({
            "ID_NUMBER": range(self.SIZE),
            "current_tariff": [self.ORIGINAL] * self.SIZE,
            "arpu_segment": ["HIGH"] * self.SIZE,
            "data_segment": ["HEAVY"] * self.SIZE,
            "call_segment": ["HIGH"] * self.SIZE,
            "predicted_arpu": [self.ARPU] * self.SIZE,
        })
        self.tariffs = pd.DataFrame({
            "tariff_plan_code": [self.ORIGINAL, self.OFFER_A, self.OFFER_B],
            "price_tariff": [4000, 6000, 7000],
        })
        costs = (200, 500, 2000) if expensive else (4, 22, 160)
        self.channels = {
            "push": {"cost_per_contact": 0, "conversion_multiplier": 0.50},
            "sms": {"cost_per_contact": costs[0], "conversion_multiplier": 0.65},
            "digital_ads": {"cost_per_contact": costs[1], "conversion_multiplier": 0.85},
            "call": {"cost_per_contact": costs[2], "conversion_multiplier": 1.20},
        }
        if repeat_mode is not None:
            # Isolate contradictory evidence from paid channel upgrades.
            self.channels = {key: self.channels[key] for key in ("push", "sms")}
        self.remaining_budget = self.BUDGET
        self.remaining_contacts = self.CONTACTS
        self.pilots_left = self.PILOTS
        self.pilot_history = []
        self.attempts = []
        self.failures = []
        self.fail_second = fail_second
        self.repeat_mode = repeat_mode
        self.observation_counts = {}

    def run_pilot(self, target_tariff, channel, n_customers=100, **filters):
        if self.pilots_left <= 0:
            raise AssertionError("Agent exceeded the pilot allowance")
        if target_tariff not in set(self.tariffs["tariff_plan_code"]):
            raise AssertionError("Unknown target tariff")
        if channel not in self.channels:
            raise AssertionError("Unknown communication channel")
        if not isinstance(n_customers, int) or not 10 <= n_customers <= 200:
            raise AssertionError("Pilot size must be an integer from 10 to 200")

        selected = self.customer_profile
        for field, value in filters.items():
            if field not in self.FILTERS:
                raise AssertionError("Unsupported pilot filter: " + field)
            if value is not None:
                values = value.split(";") if field == "filter_current_tariff" else [value]
                selected = selected[selected[self.FILTERS[field]].isin(values)]
        n = min(n_customers, len(selected))
        cost = n * self.channels[channel]["cost_per_contact"]
        if n < 10 or n > self.remaining_contacts or cost > self.remaining_budget:
            raise AssertionError("Agent requested an unaffordable or empty pilot")

        attempt = {"target_tariff": target_tariff, "channel": channel,
                   "n_customers": n, "filters": dict(filters)}
        self.attempts.append(attempt)
        if self.fail_second and len(self.attempts) == 2:
            # A rejected call makes no contacts and consumes no pilot allowance.
            self.failures.append(attempt)
            raise RuntimeError("synthetic transient pilot failure")

        key = (target_tariff, channel)
        self.observation_counts[key] = self.observation_counts.get(key, 0) + 1
        if self.repeat_mode is None:
            ratio = (0.50 if target_tariff == self.OFFER_A else -0.50)
            ratio *= self.channels[channel]["conversion_multiplier"]
        elif target_tariff != self.OFFER_A:
            ratio = -0.20
        elif self.repeat_mode == "conflicting" and self.observation_counts[key] > 1:
            ratio = -0.065
        else:
            ratio = 0.065

        self.remaining_budget -= cost
        self.remaining_contacts -= n
        self.pilots_left -= 1
        result = {
            **attempt,
            "pilot": len(self.pilot_history) + 1,
            "observed_lift_ratio": ratio,
            "observed_lift_total": ratio * n * self.ARPU,
            "cost": cost,
            "remaining_budget": self.remaining_budget,
            "remaining_contacts": self.remaining_contacts,
        }
        self.pilot_history.append(result)
        return result


class AgentResilienceTests(unittest.TestCase):
    def assert_resources_and_plan(self, env, campaigns):
        """Hand-count this fixture: each nonempty final filter selects all 600."""
        self.assertIsInstance(campaigns, list)
        self.assertEqual(len(campaigns), 1, "One audience must not be contacted twice")
        self.assertGreater(len(env.pilot_history), 0)
        self.assertLessEqual(len(env.pilot_history), 20)
        self.assertLessEqual(len(env.attempts), 20)

        pilot_contacts = sum(item["n_customers"] for item in env.pilot_history)
        pilot_cost = sum(item["n_customers"] *
                         env.channels[item["channel"]]["cost_per_contact"]
                         for item in env.pilot_history)
        self.assertEqual(env.remaining_budget, 100000 - pilot_cost)
        self.assertEqual(env.remaining_contacts, 15000 - pilot_contacts)
        self.assertEqual(env.pilots_left, 20 - len(env.pilot_history))

        campaign = campaigns[0]
        self.assertIsInstance(campaign, dict)
        self.assertIn(campaign.get("target_tariff"), (env.OFFER_A, env.OFFER_B))
        self.assertIn(campaign.get("channel"), env.channels)
        allowed = {"campaign_name", "target_tariff", "channel"} | set(env.FILTERS)
        self.assertFalse(set(campaign) - allowed)
        expected = {"filter_current_tariff": env.ORIGINAL,
                    "filter_arpu_segment": "HIGH", "filter_data_segment": "HEAVY",
                    "filter_call_segment": "HIGH"}
        for field, value in expected.items():
            if campaign.get(field) is not None:
                self.assertEqual(campaign[field], value)
        final_cost = 600 * env.channels[campaign["channel"]]["cost_per_contact"]
        self.assertLessEqual(pilot_cost + final_cost, 100000)
        self.assertLessEqual(pilot_contacts + 600, 15000)

    def assert_direct_evidence(self, env, campaign):
        matching = [item for item in env.pilot_history
                    if item["target_tariff"] == campaign["target_tariff"]
                    and item["channel"] == campaign["channel"]
                    and item["filters"] == {key: campaign[key] for key in env.FILTERS
                                           if key in campaign}]
        self.assertTrue(matching, "Selected campaign must have a successful exact pilot")

    def test_one_failed_pilot_preserves_successful_evidence_and_resources(self):
        """Один сбой не теряет успешные пилоты и не ломает итоговый план."""
        env = ResilienceEnvironment(fail_second=True)
        agent = Agent()
        campaigns = agent.act(env)

        self.assert_resources_and_plan(env, campaigns)
        self.assertEqual(len(env.failures), 1)
        self.assertEqual(len(env.attempts), len(env.pilot_history) + 1)
        self.assertGreaterEqual(len(env.pilot_history), 2, "Research should continue after recovery")
        first = env.pilot_history[0]
        self.assertEqual((first["target_tariff"], first["channel"]), (env.OFFER_A, "sms"))
        self.assertEqual(agent.last_report["pilots"][0]["observed_lift_ratio"], 0.325)
        self.assertEqual(len(agent.last_report["pilots"]), len(env.pilot_history))
        self.assertTrue(any("synthetic transient pilot failure" in text
                            for text in agent.last_report["warnings"]))
        self.assertEqual(campaigns[0]["target_tariff"], env.OFFER_A)
        self.assertFalse(agent.last_report["decisions"][0].get("unverified_fallback", False))
        self.assert_direct_evidence(env, campaigns[0])

    def test_higher_paid_costs_switch_to_a_directly_tested_affordable_channel(self):
        """Если платные каналы не помещаются в бюджет, проверяется бесплатный."""
        baseline = ResilienceEnvironment()
        baseline_plan = Agent().act(baseline)
        self.assert_resources_and_plan(baseline, baseline_plan)
        self.assertNotEqual(baseline_plan[0]["channel"], "push")

        expensive = ResilienceEnvironment(expensive=True)
        agent = Agent()
        campaigns = agent.act(expensive)
        self.assert_resources_and_plan(expensive, campaigns)
        # Even the cheapest paid final contact plan costs 600 * 200 = 120000.
        self.assertEqual(600 * expensive.channels["sms"]["cost_per_contact"], 120000)
        self.assertEqual(campaigns[0]["channel"], "push")
        self.assertEqual(campaigns[0]["target_tariff"], expensive.OFFER_A)
        self.assertTrue(all(item["channel"] == "push" for item in expensive.pilot_history))
        self.assertEqual(expensive.remaining_budget, 100000)
        self.assertFalse(agent.last_report["decisions"][0].get("unverified_fallback", False))
        self.assert_direct_evidence(expensive, campaigns[0])

    def test_contradictory_repeats_increase_uncertainty_and_downgrade_plan(self):
        """Противоречие пилотов увеличивает неопределённость, а не уверенность."""
        consistent = ResilienceEnvironment(repeat_mode="consistent")
        consistent_agent = Agent()
        consistent_plan = consistent_agent.act(consistent)
        self.assert_resources_and_plan(consistent, consistent_plan)
        self.assertEqual(consistent_plan[0]["channel"], "sms")
        self.assertFalse(consistent_agent.last_report["decisions"][0].get("unverified_fallback", False))

        conflicting = ResilienceEnvironment(repeat_mode="conflicting")
        agent = Agent()
        campaigns = agent.act(conflicting)
        self.assert_resources_and_plan(conflicting, campaigns)
        repeated = [item for item in agent.last_report["pilots"]
                    if item["target_tariff"] == conflicting.OFFER_A and item["channel"] == "sms"]
        self.assertEqual([item["observed_lift_ratio"] for item in repeated], [0.065, -0.065])
        observations = [item for item in conflicting.pilot_history
                        if item["target_tariff"] == conflicting.OFFER_A and item["channel"] == "sms"]
        counts = [item["n_customers"] for item in observations]
        self.assertEqual([item["actual_n"] for item in repeated], counts)
        self.assertEqual([item["action"] for item in repeated], ["explore", "confirm"])
        # Extra observations must not conceal the disagreement between batches.
        self.assertGreater(repeated[-1]["standard_error"], repeated[0]["standard_error"])
        self.assertGreater(repeated[-1]["standard_error"], 0.80 / math.sqrt(sum(counts)))
        # Count the public observations independently; do not pin the agent's
        # adaptive repeat size to an implementation-specific constant.
        expected_mean = (counts[0] * 0.065 - counts[1] * 0.065) / sum(counts)
        self.assertAlmostEqual(repeated[-1]["pooled_lift_ratio"], expected_mean)
        self.assertEqual(campaigns[0]["channel"], "push")
        self.assertTrue(agent.last_report["decisions"][0]["unverified_fallback"])
        self.assertEqual(agent.last_report["decisions"][0]["pilot_n"], 0)


if __name__ == "__main__":
    unittest.main()
