"""Adaptive, offline tariff campaign planner using only the participant API.

The public example states that a 150-person pilot has noise std about .07.
We use .80 / sqrt(n) as a configurable working standard error. Intervals are
model-based risk estimates, not distribution-free confidence guarantees.
Historical tariff changes rank hypotheses only: they are not causal estimates
for the new audience. Final profitable campaigns require direct pilot evidence
for their exact segment, target and channel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd


@dataclass
class Cell:
    key: tuple
    filters: dict
    n: int
    arpu: float
    current: str
    segment: str


@dataclass
class Evidence:
    cell: Cell
    target: str
    channel: str
    batches: list = field(default_factory=list)

    @property
    def n(self):
        return sum(batch[0] for batch in self.batches)

    @property
    def mean(self):
        return sum(n * ratio for n, ratio in self.batches) / self.n

    def se(self, noise_std):
        # Repeat-pilot disagreement can INCREASE uncertainty. It must never
        # spuriously make the documented noise assumption more optimistic.
        base = noise_std / math.sqrt(self.n)
        if len(self.batches) < 2:
            return base
        means = np.array([ratio for _, ratio in self.batches], dtype=float)
        counts = np.array([n for n, _ in self.batches], dtype=float)
        weights = counts / counts.sum()
        denom = 1.0 - float(np.dot(weights, weights))
        if denom <= 0:
            return base
        batch_var = float(np.dot(weights, (means - self.mean) ** 2)) / denom
        disagreement = math.sqrt(batch_var * float(np.dot(weights, weights)))
        return max(base, disagreement)


class Agent:
    """Budgeted sequential experiments followed by a disjoint portfolio.

    Policy parameters express risk/resource preferences, not fitted mock values:
    * noise_std: public approximate individual pilot noise;
    * risk_aversion: number of working standard errors subtracted from uplift;
    * pilot_budget_fraction: reserve most money for final implementation;
    * max_seconds: local soft deadline, below both published time limits.
    """

    def __init__(self, noise_std=0.80, risk_aversion=1.28,
                 pilot_budget_fraction=0.20, max_seconds=240):
        self.noise_std = max(0.01, float(noise_std))
        self.risk_aversion = max(0.0, float(risk_aversion))
        self.pilot_budget_fraction = min(1.0, max(0.0, float(pilot_budget_fraction)))
        self.max_seconds = max(1.0, float(max_seconds))
        self.last_report = {}

    @staticmethod
    def _number(value, default=0.0):
        try:
            result = float(value)
            return result if math.isfinite(result) else default
        except (TypeError, ValueError, OverflowError):
            return default

    def _cells(self, profile):
        """A partition: never choose overlapping final campaigns by construction."""
        result = []
        required = ["current_tariff", "arpu_segment"]
        valid = profile.dropna(subset=required).copy()
        valid["predicted_arpu"] = pd.to_numeric(
            valid["predicted_arpu"], errors="coerce"
        ).replace([np.inf, -np.inf], 0).fillna(0).clip(lower=0)
        valid = valid[valid["current_tariff"].astype(str).str.len() > 0]

        def add_or_split(part, filters, key, remaining):
            if len(part) > 5000 and remaining:
                column = remaining[0]
                for value, sub in part.groupby(column, observed=True, sort=True):
                    if pd.notna(value):
                        add_or_split(sub, {**filters, f"filter_{column}": str(value)},
                                     (*key, str(value)), remaining[1:])
                return
            if 0 < len(part) <= 5000:
                result.append(Cell(
                    key=key, filters=filters, n=int(len(part)),
                    arpu=float(part["predicted_arpu"].sum()),
                    current=filters["filter_current_tariff"],
                    segment=filters["filter_arpu_segment"],
                ))

        extra = [x for x in ["data_segment", "call_segment"] if x in valid.columns]
        for (current, segment), part in valid.groupby(required, observed=True, sort=True):
            add_or_split(part, {"filter_current_tariff": str(current),
                               "filter_arpu_segment": str(segment)},
                         (str(current), str(segment)), extra)
        return sorted(result, key=lambda cell: (-cell.arpu, cell.key))

    def _historical_ranks(self, tariffs):
        """Read the explicitly supplied historical CSV, never environment files."""
        path = Path(__file__).resolve().parent / "data" / "change_tariff.csv"
        columns = ["AVG_ARPU_PREV_3M", "AVG_ARPU_NEXT_3M",
                   "tariff_plan_code_from", "tariff_plan_code_to"]
        try:
            history = pd.read_csv(path, usecols=columns)
            previous = pd.to_numeric(history[columns[0]], errors="coerce")
            following = pd.to_numeric(history[columns[1]], errors="coerce")
            keep = (previous > 0) & following.notna() & np.isfinite(previous) & np.isfinite(following)
            history = history.loc[keep].copy()
            previous, following = previous.loc[keep], following.loc[keep]
            # Aggregate dollars before taking a ratio. Averaging individual
            # percentage changes can reverse the sign of actual ARPU change.
            history["historical_delta"] = following - previous
            history["historical_baseline"] = previous
            history["segment"] = np.where(previous < 1000, "LOW",
                                           np.where(previous > 5000, "HIGH", "MID"))
            history = history[history["tariff_plan_code_to"].isin(tariffs)]
            # Condition the fallback on pre-transition ARPU too. Pooling LOW
            # and HIGH would recommend destinations popular among tiny-ARPU
            # subscribers to high-ARPU users (a compositional confounder).
            def summaries(keys):
                grouped = history.groupby(keys).agg(
                    delta=("historical_delta", "sum"),
                    baseline=("historical_baseline", "sum"),
                    count=("historical_delta", "size"))
                # Apply the bounded ranking transform AFTER aggregation, so
                # it preserves the financial sign and damps tiny denominators.
                grouped["mean"] = np.tanh(grouped["delta"] / grouped["baseline"])
                return grouped

            target = summaries(["segment", "tariff_plan_code_to"])
            target_stats = {tuple(map(str, k)): (float(v["mean"]), int(v["count"]))
                            for k, v in target.iterrows()}
            segment_stats = summaries("segment")
            target_stats.update({(str(k), "__segment_average__"):
                                 (float(v["mean"]), int(v["count"]))
                                 for k, v in segment_stats.iterrows()})
            pairs = summaries(["tariff_plan_code_from", "segment", "tariff_plan_code_to"])
            pair_stats = {tuple(map(str, k)): (float(v["mean"]), int(v["count"]))
                          for k, v in pairs.iterrows()}
            self.last_report["history"] = {"rows_used": int(len(history)),
                                            "role": "hypothesis_ranking_only"}
            return pair_stats, target_stats
        except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
            self.last_report["history"] = {"rows_used": 0,
                                            "role": "optional_history_unavailable",
                                            "reason": str(exc)[:160]}
            return {}, {}

    def _hypotheses(self, cells, tariffs):
        pairs, targets = self._historical_ranks(tariffs)
        hypotheses = []
        for cell in cells:
            choices = []
            for target in tariffs:
                if target == cell.current:
                    continue
                fallback = targets.get((cell.segment, "__segment_average__"), (0.0, 0))
                global_mean, global_n = targets.get((cell.segment, target), fallback)
                pair_mean, pair_n = pairs.get((cell.current, cell.segment, target), (global_mean, 0))
                # Shrink small historical groups towards a broad target ranking.
                weight = pair_n / (pair_n + math.sqrt(max(global_n, 1)))
                score = weight * pair_mean + (1 - weight) * global_mean
                choices.append((score, target))
            choices.sort(key=lambda item: (-item[0], item[1]))
            for position, (score, target) in enumerate(choices):
                rank = 1.0 - position / max(1, len(choices) - 1)
                hypotheses.append((cell, target, rank, score))
        return hypotheses

    def _limits(self, env):
        return (max(0.0, self._number(env.remaining_budget)),
                max(0, int(self._number(env.remaining_contacts))),
                max(0, int(self._number(env.pilots_left))))

    def _sample_size(self, cell, evidence, max_arpu, available, cost, pilot_money):
        if evidence is None:
            # Spend more observations where a wrong decision affects more ARPU.
            n = math.ceil(50 + 150 * math.sqrt(cell.arpu / max(max_arpu, 1)))
        else:
            break_even = cost * cell.n / max(cell.arpu, 1)
            margin = abs(evidence.mean - break_even)
            target_total = math.ceil((self.risk_aversion * self.noise_std /
                                      max(margin, self.noise_std / math.sqrt(400))) ** 2)
            n = max(30, target_total - evidence.n)
        n = min(200, cell.n, available, n)
        if cost > 0:
            n = min(n, int(max(0.0, pilot_money) // cost))
        return int(n) if n >= 10 else 0

    def _actions(self, env, cells, hypotheses, evidence, channels,
                 initial_budget, initial_contacts, initial_pilots, attempted):
        budget, contacts, pilots_left = self._limits(env)
        spent = max(0.0, initial_budget - budget)
        pilot_money = min(budget, max(0.0, initial_budget * self.pilot_budget_fraction - spent))
        # Reserve one complete deployable segment and most contact capacity.
        reserve = min((cell.n for cell in cells), default=0)
        pilot_contacts = min(max(0, contacts - reserve),
                             max(0, initial_contacts // 3 - (initial_contacts - contacts)))
        if pilots_left <= 0 or pilot_contacts < 10:
            return []
        max_arpu = max((cell.arpu for cell in cells), default=1)
        per_cell = {}
        for item in evidence.values():
            per_cell.setdefault(item.cell.key, []).append(item)
        # SMS is an inexpensive exploration channel if it can be afforded at
        # implementation scale. Otherwise use the cheapest available channel.
        cheap = sorted(channels, key=lambda c: (channels[c]["cost"], c))
        cheapest = cheap[0]
        base = "sms" if "sms" in channels else cheapest
        remaining_fraction = pilots_left / max(1, initial_pilots)
        uncertainty_scale = self.noise_std / math.sqrt(100)
        observed_segments = {item.cell.segment for item in evidence.values()}
        eligible_segments = {cell.segment for cell in cells if cell.n >= 10 and cell.arpu > 0}
        uncovered_segments = eligible_segments - observed_segments
        observations = sum(len(item.batches) for item in evidence.values())
        history_available = self.last_report.get("history", {}).get("rows_used", 0) > 0
        # A limited exploration component remains even when the observational
        # history is unhelpful, but raw audience size cannot dominate it.
        max_historical_value = max((cell.arpu * max(0.0, signal)
                                    for cell, _, _, signal in hypotheses), default=0)
        actions = []
        for cell, target, rank, historical_signal in hypotheses:
            if cell.n < 10 or cell.arpu <= 0:
                continue
            cell_items = per_cell.get(cell.key, [])
            base_channel = base
            if channels[base]["cost"] * cell.n > budget or pilot_money < 10 * channels[base]["cost"]:
                base_channel = cheapest
            key = (cell.key, target, base_channel)
            if key in evidence or key in attempted:
                continue
            if uncovered_segments and cell.segment not in uncovered_segments:
                continue
            # Stop exploring a new target in the last slots when an existing
            # promising result needs confirmation. Poor early results instead
            # keep the search open.
            promising = any(x.mean > 0 for x in evidence.values())
            if pilots_left <= 2 and promising:
                continue
            incumbent = max((x.mean - self.risk_aversion * x.se(self.noise_std)
                             for x in cell_items), default=0.0)
            tested_targets = len({x.target for x in cell_items})
            diversification = 1.0 / (1 + tested_targets) ** 1.5
            if history_available and max_historical_value > 0:
                # Weak absolute historical signal: rank-within-cell alone
                # would incorrectly give a losing large cell the same prior
                # as a clearly improving smaller cell. This utility is only
                # used for research allocation, never final revenue estimates.
                historical_value = cell.arpu * max(0.0, historical_signal)
                exploratory_bonus = 0.05 * max_historical_value * math.sqrt(
                    cell.arpu / max(max_arpu, 1)) * rank
                priority = 0.25 * max(0.0, historical_value - max(0, incumbent) * cell.arpu)
                priority += exploratory_bonus
                # After covering ARPU strata, unsupported/negative historical
                # hypotheses receive a bounded share of the remaining search.
                if not uncovered_segments and historical_signal <= 0 and observations % 4 != 0:
                    continue
            else:
                opportunity = max(0.0, uncertainty_scale * (0.5 + rank) - max(0, incumbent))
                priority = cell.arpu * opportunity
            priority *= diversification * (0.35 + 0.65 * remaining_fraction)
            n = self._sample_size(cell, None, max_arpu, pilot_contacts,
                                  channels[base_channel]["cost"], pilot_money)
            if n and priority > max(0.0, n * channels[base_channel]["cost"]):
                actions.append((priority, "explore", cell, target, base_channel, n,
                                "signed financial history, ARPU coverage and audience value"))

        for key, item in evidence.items():
            cell = item.cell
            se = item.se(self.noise_std)
            cost = channels[item.channel]["cost"]
            break_even = cost * cell.n / max(cell.arpu, 1)
            low = item.mean - self.risk_aversion * se
            high = item.mean + self.risk_aversion * se
            if high <= break_even:
                continue
            # Confirmation is valuable near zero and when uncertainty affects
            # ranking against another measured alternative in the same cell.
            other = [x for x in per_cell[cell.key] if x is not item]
            competitor = max((x.mean - channels[x.channel]["cost"] * cell.n /
                              max(cell.arpu, 1) for x in other), default=0.0)
            decision_gap = min(abs(item.mean - break_even),
                               abs(item.mean - break_even - competitor))
            ambiguity = math.exp(-0.5 * (decision_gap / max(se, 1e-9)) ** 2)
            positive_margin = item.mean - break_even
            # Repeating a near-zero mean requires unbounded sample size. Only
            # confirm plausible finalists whose decision can be resolved with
            # the remaining experiment capacity; cover ARPU strata first.
            needed_total = (self.risk_aversion * self.noise_std /
                            max(positive_margin, 1e-9)) ** 2
            resolvable = needed_total <= item.n + 200 * max(1, pilots_left // 2)
            if (not uncovered_segments and len(item.batches) < 4 and
                    positive_margin >= 0.5 * se and resolvable and
                    (low <= break_even or ambiguity > 0.05)):
                n = self._sample_size(cell, item, max_arpu, pilot_contacts, cost, pilot_money)
                if n:
                    se_reduction = max(0.0, se - self.noise_std / math.sqrt(item.n + n))
                    priority = cell.arpu * self.risk_aversion * se_reduction * ambiguity
                    priority *= 1.5 - 0.5 * remaining_fraction
                    actions.append((priority, "confirm", cell, item.target, item.channel, n,
                                    "uncertainty can change profitability or alternative ranking"))

            if low <= break_even or item.mean <= 0 or uncovered_segments:
                continue
            for channel, config in channels.items():
                new_key = (cell.key, item.target, channel)
                if new_key in evidence or new_key in attempted or channel == item.channel:
                    continue
                final_cost = config["cost"] * cell.n
                if final_cost > budget:
                    continue
                # Multiplier scaling is a SEARCH heuristic only; saturation
                # can invalidate it. An upgrade never enters the final plan
                # without its own real pilot.
                factor = config["multiplier"] / max(channels[item.channel]["multiplier"], 1e-9)
                estimated = item.mean * factor
                improvement = cell.arpu * (estimated - item.mean) - (config["cost"] - cost) * cell.n
                cheaper_hedge = config["cost"] < cost and low <= break_even
                if improvement <= 0 and not cheaper_hedge:
                    continue
                n = self._sample_size(cell, None, max_arpu, pilot_contacts, config["cost"],
                                      min(pilot_money, max(0, budget - final_cost)))
                if not n:
                    continue
                # A tiny, noisy expensive pilot rarely settles a channel
                # upgrade; discount its expected decision value accordingly.
                reliability = math.sqrt(n / 200)
                priority = max(improvement, cell.arpu * se * 0.25) * reliability
                priority /= 1 + config["cost"] * n / max(pilot_money, 1)
                actions.append((priority, "channel_check", cell, item.target, channel, n,
                                "marginal channel economics; direct pilot required"))
        return sorted(actions, key=lambda a: (-a[0], a[1], a[2].key, a[3], a[4]))

    def _portfolio(self, evidence, channels, budget, contacts):
        """Small beam search over measured choices, at most one per cell."""
        grouped = {}
        for item in evidence.values():
            cost = channels[item.channel]["cost"] * item.cell.n
            lower = item.mean - self.risk_aversion * item.se(self.noise_std)
            score = lower * item.cell.arpu - cost
            if score > 0 and cost <= budget and item.cell.n <= contacts:
                grouped.setdefault(item.cell.key, []).append((item, cost, score))
        # Each state: conservative value, spend, contacts, chosen evidence.
        states = [(0.0, 0.0, 0, ())]
        for key in sorted(grouped):
            new = list(states)
            for value, spend, count, chosen in states:
                if len(chosen) >= 10:
                    continue
                for item, cost, score in grouped[key]:
                    if spend + cost <= budget + 1e-8 and count + item.cell.n <= contacts:
                        new.append((value + score, spend + cost,
                                    count + item.cell.n, (*chosen, item)))
            if len(new) > 1200:
                # Retain high-value and resource-efficient partial portfolios.
                indices = set()
                for metric in [lambda s: s[0],
                               lambda s: s[0] / (1 + s[1] / max(budget, 1)),
                               lambda s: s[0] / (1 + s[2] / max(contacts, 1))]:
                    indices.update(sorted(range(len(new)), key=lambda i: metric(new[i]),
                                          reverse=True)[:400])
                states = [new[i] for i in sorted(indices)]
            else:
                states = new
        return list(max(states, key=lambda s: (s[0], -s[1], -s[2]))[3])

    def _campaign(self, item, index):
        return {"campaign_name": f"Evidence_{index:02d}_{item.target}_{item.channel}",
                **item.cell.filters, "target_tariff": item.target, "channel": item.channel}

    def act(self, env) -> list[dict]:
        started = time.monotonic()
        initial_budget, initial_contacts, initial_pilots = self._limits(env)
        self.last_report = {
            "strategy": "adaptive_evidence_portfolio",
            "assumptions": {
                "noise_std": self.noise_std,
                "risk_aversion": self.risk_aversion,
                "uncertainty": "working normal-noise estimate; no coverage guarantee",
                "history": "ranking only; different cohort, non-causal observations",
                "channel_transfer": "search heuristic only; final channels directly piloted",
                "pilot_overlap": "sample IDs unavailable; pilot/final overlap is not exactly measurable",
            },
            "initial_resources": {"budget": initial_budget, "contacts": initial_contacts,
                                  "pilots_left": initial_pilots},
            "pilots": [], "decisions": [], "warnings": [],
        }
        profile = env.customer_profile
        tariffs = sorted(set(str(x) for x in env.tariffs["tariff_plan_code"].dropna()))
        channels = {
            str(name): {"cost": max(0.0, self._number(config["cost_per_contact"])),
                        "multiplier": max(0.0, self._number(config["conversion_multiplier"]))}
            for name, config in env.channels.items()
        }
        if not tariffs or not channels:
            self.last_report["warnings"].append("No valid tariffs/channels in public input")
            return []
        cells = self._cells(profile)
        hypotheses = self._hypotheses(cells, tariffs)
        evidence = {}
        attempted = set()
        # A finite bound also protects against an API failure that does not
        # decrement pilots_left. Successful calls use only public results.
        attempts = 0
        while attempts < min(20, initial_pilots) and time.monotonic() - started < self.max_seconds:
            if evidence and attempts >= max(3, math.ceil(initial_pilots / 2)):
                plausible = any(
                    item.mean - channels[item.channel]["cost"] * item.cell.n /
                    max(item.cell.arpu, 1) >= 0.75 * item.se(self.noise_std)
                    for item in evidence.values())
                if not plausible:
                    self.last_report["stop_reason"] = "half the pilot allowance used; no plausible positive finalist"
                    break
            actions = self._actions(env, cells, hypotheses, evidence, channels,
                                    initial_budget, initial_contacts, initial_pilots, attempted)
            if not actions:
                self.last_report["stop_reason"] = "no feasible experiment with sufficient decision value"
                break
            priority, kind, cell, target, channel, n, reason = actions[0]
            key = (cell.key, target, channel)
            attempted.add(key)
            attempts += 1
            try:
                result = env.run_pilot(target_tariff=target, channel=channel,
                                       n_customers=n, **cell.filters)
                actual_n = int(self._number(result.get("n_customers"), n))
                ratio = self._number(result.get("observed_lift_ratio"), float("nan"))
                if actual_n < 1 or not math.isfinite(ratio):
                    self.last_report["warnings"].append("Pilot returned invalid aggregate evidence")
                    continue
                if key not in evidence:
                    evidence[key] = Evidence(cell, target, channel)
                evidence[key].batches.append((actual_n, ratio))
                current = evidence[key]
                self.last_report["pilots"].append({
                    "step": len(self.last_report["pilots"]) + 1,
                    "action": kind, "filters": dict(cell.filters), "target_tariff": target,
                    "channel": channel, "requested_n": n, "actual_n": actual_n,
                    "observed_lift_ratio": ratio, "pooled_lift_ratio": current.mean,
                    "standard_error": current.se(self.noise_std),
                    "cost": self._number(result.get("cost"), actual_n * channels[channel]["cost"]),
                    "reason": reason, "priority": float(priority),
                })
            except (RuntimeError, ValueError, KeyError, TypeError) as exc:
                self.last_report["warnings"].append(f"Pilot unavailable: {type(exc).__name__}: {str(exc)[:160]}")
                # Keep already acquired evidence and produce a feasible plan.
                continue

        budget, contacts, pilots_left = self._limits(env)
        selected = self._portfolio(evidence, channels, budget, contacts)
        selected.sort(key=lambda item: (-(item.mean - self.risk_aversion * item.se(self.noise_std)) *
                                         item.cell.arpu + channels[item.channel]["cost"] * item.cell.n,
                                        item.cell.key))
        campaigns = []
        for index, item in enumerate(selected, 1):
            campaign = self._campaign(item, index)
            campaigns.append(campaign)
            lower = item.mean - self.risk_aversion * item.se(self.noise_std)
            cost = item.cell.n * channels[item.channel]["cost"]
            self.last_report["decisions"].append({
                **campaign, "n_customers": item.cell.n, "predicted_arpu_sum": item.cell.arpu,
                "pilot_n": item.n, "pilot_batches": len(item.batches),
                "lift_ratio_mean": item.mean, "standard_error": item.se(self.noise_std),
                "lift_ratio_lower": lower, "communication_cost": cost,
                "estimated_net": item.mean * item.cell.arpu - cost,
                "conservative_net": lower * item.cell.arpu - cost,
                "reason": "positive conservative value; exact segment/target/channel piloted",
            })
        if not campaigns:
            # The contract requires >=1 campaign even when all evidence is
            # negative. Explicitly label the minimum-exposure fallback.
            cheapest = min(channels, key=lambda name: (channels[name]["cost"], name))
            feasible = [cell for cell in cells if cell.n <= contacts and
                        cell.n * channels[cheapest]["cost"] <= budget and
                        any(t != cell.current for t in tariffs)]
            if feasible:
                cell = min(feasible, key=lambda item: (item.arpu, item.n, item.key))
                target = next(t for c, t, _, _ in hypotheses if c.key == cell.key)
                fallback = Evidence(cell, target, cheapest)
                campaign = self._campaign(fallback, 1)
                campaign["campaign_name"] = "Minimum_exposure_fallback"
                campaigns = [campaign]
                self.last_report["decisions"].append({
                    **campaign, "n_customers": cell.n,
                    "communication_cost": cell.n * channels[cheapest]["cost"],
                    "reason": "mandatory campaign; no positive conservative portfolio",
                    "pilot_n": 0, "unverified_fallback": True,
                })
                self.last_report["warnings"].append("Fallback has no guaranteed positive effect")
            else:
                self.last_report["warnings"].append("No campaign can satisfy remaining resources")
        self.last_report["remaining_resources_after_pilots"] = {
            "budget": budget, "contacts": contacts, "pilots_left": pilots_left}
        self.last_report["planned_contacts"] = sum(x["n_customers"] for x in self.last_report["decisions"])
        self.last_report["planned_cost"] = sum(x["communication_cost"] for x in self.last_report["decisions"])
        self.last_report["elapsed_seconds"] = time.monotonic() - started
        self.last_report["candidate_cells"] = len(cells)
        return campaigns
