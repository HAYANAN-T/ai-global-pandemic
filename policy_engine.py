"""Validated policy actions. The engine, never the advisor, decides what an action does."""
import numpy as np

ACTIONS = {
    #                      daily cost, delay (days), description
    "info_campaign":        dict(cost=0.4, delay=2,  desc="Public information campaign"),
    "testing_monitoring":   dict(cost=1.6, delay=5,  desc="Testing and voluntary health monitoring"),
    "isolation_guidance":   dict(cost=0.3, delay=1,  desc="Isolation guidance for detected cases"),
    "limit_gatherings":     dict(cost=0.3, delay=2,  desc="Temporary limits on large gatherings"),
    "remote_work":          dict(cost=0.2, delay=3,  desc="Remote work where possible"),
    "close_schools":        dict(cost=0.2, delay=2,  desc="Temporary school closures"),
    "expand_capacity":      dict(cost=2.2, delay=10, desc="Expand hospital capacity within resources"),
    "essential_supplies":   dict(cost=0.6, delay=3,  desc="Distribute essential supplies"),
    "financial_support":    dict(cost=2.5, delay=5,  desc="Financial support for affected households"),
    "protect_vulnerable":   dict(cost=0.7, delay=3,  desc="Measures to protect vulnerable people"),
    "trust_communication":  dict(cost=0.5, delay=2,  desc="Transparent communication and rumor response"),
}
RESTRICTIVE = {"limit_gatherings", "remote_work", "close_schools"}

class PolicyState:
    def __init__(self):
        self.active, self.pending, self.log = set(), [], []

    def apply_decision(self, world, decision, advisor):
        """Validate and schedule a decision. Returns (accepted, rejected_with_reasons)."""
        wanted = decision.get("actions", [])
        accepted, rejected = [], []
        if not isinstance(wanted, list):
            return [], [("<actions>", "actions must be a list")]
        interval = world.cfg["simulation"]["decision_interval"]
        projected = 0.0
        for a in wanted:
            if a not in ACTIONS:
                rejected.append((a, "unknown action")); continue
            projected += ACTIONS[a]["cost"] * interval
        if projected > world.budget_left + 1e-9:
            # keep actions in the advisor's stated order until the budget runs out
            keep, spent = [], 0.0
            for a in wanted:
                if a in ACTIONS:
                    c = ACTIONS[a]["cost"] * interval
                    if spent + c <= world.budget_left: keep.append(a); spent += c
                    else: rejected.append((a, "exceeds remaining budget"))
            wanted = keep
        for a in wanted:
            if a in ACTIONS and a not in [r[0] for r in rejected]: accepted.append(a)
        # schedule switches: newly added actions start after their delay, dropped actions stop now
        for a in list(self.active):
            if a not in accepted: self.active.discard(a)
        self.pending = [(d, a) for d, a in self.pending if a in accepted]
        for a in accepted:
            if a not in self.active and a not in [p[1] for p in self.pending]:
                self.pending.append((world.day + ACTIONS[a]["delay"], a))
        self.log.append(dict(day=world.day, advisor=advisor, accepted=accepted, rejected=rejected,
                             rationale=decision.get("rationale", "")[:500], uncertainty=decision.get("uncertainty", "")[:300]))
        return accepted, rejected

    def tick(self, world):
        for d, a in list(self.pending):
            if world.day >= d: self.active.add(a); self.pending.remove((d, a))
        cost = sum(ACTIONS[a]["cost"] for a in self.active)
        world.budget_left = max(0.0, world.budget_left - cost)
        if world.budget_left <= 0:                       # out of money: costly actions lapse
            self.active = {a for a in self.active if ACTIONS[a]["cost"] == 0}
        return cost

def compliance(world, active):
    c = 0.30 + 0.60 * world.trust
    if "financial_support" in active: c += 0.12           # people can afford to stay home
    if "info_campaign" in active: c += 0.05
    return float(np.clip(c, 0, 1))

def modifiers(world, active):
    comp = compliance(world, active)
    s = dict(home=1.0, school=1.0, work=1.0, community=1.0)
    if "limit_gatherings" in active: s["community"] *= 1 - 0.45 * comp
    if "remote_work" in active:      s["work"] *= 1 - 0.50 * comp
    if "close_schools" in active:    s["school"] *= 1 - 0.85 * comp
    iso = 0.0
    if "isolation_guidance" in active:
        iso = 0.30 * comp * (1.0 if "testing_monitoring" in active else 0.45)
    older = 1 - 0.45 * comp if "protect_vulnerable" in active else 1.0
    beds = 1.35 if "expand_capacity" in active else 1.0
    return dict(setting=s, isolation=iso, older_protection=older, bed_mult=beds), comp

def update_society(world, active, out, sick_share_adults):
    """Trust and economy respond to policies, rumors and hospital overload."""
    t = world.trust
    if "info_campaign" in active: t += 0.0025
    if "trust_communication" in active: t += 0.0035
    restrictive = len(active & RESTRICTIVE)
    if restrictive and "financial_support" not in active: t -= 0.0012 * restrictive    # hardship erodes trust
    overload = float((out["occupancy"] > out["beds"]).any())
    t -= 0.004 * overload
    if world.day in world.sc["rumors"]:
        t -= 0.12 * (0.45 if "trust_communication" in active else 1.0)
    t += 0.0008 * (0.6 - t)                                                   # slow drift toward normal
    world.trust = float(np.clip(t, 0.05, 0.95))
    e = world.econ_mult
    activity = 1.0 - e * (0.035 * ("limit_gatherings" in active) + 0.07 * ("remote_work" in active)
                          + 0.03 * ("close_schools" in active)) - 0.6 * sick_share_adults - 0.05 * overload
    activity = float(np.clip(activity, 0.3, 1.0))
    loss = 1 - activity
    support = "financial_support" in active
    income_loss_low = loss * 1.4 * (0.4 if support else 1.0) * (0.85 if "essential_supplies" in active else 1.0)
    income_loss_other = loss * 0.8
    return activity, income_loss_low, income_loss_other

def observe(world, pop, out, history, active):
    """The dashboard every advisor receives: noisy, delayed and (in scenario F) conflicting."""
    rng = world.rng
    recent = history[-7:] if history else []
    true_cases = sum(h["new_symptomatic"] for h in recent) / max(len(recent), 1)
    detect = 0.55 if "testing_monitoring" in active else 0.22
    reported = true_cases * detect * rng.uniform(0.85, 1.15)
    occ = float(out["occupancy"].sum()) if out else 0.0
    beds = float((world.beds * (1.35 if "expand_capacity" in active else 1.0)).sum())
    obs = dict(day=world.day, reported_daily_cases_7d=round(reported, 1),
               reported_case_trend=("rising" if len(history) > 14 and reported > 1.15 * history[-8].get("reported", reported) else "flat_or_falling"),
               hospital_occupancy_pct=round(100 * occ / beds, 1), hospital_beds_total=int(beds),
               public_trust_survey=round(float(np.clip(world.trust + rng.normal(0, 0.03), 0, 1)), 2),
               economic_activity_index=round(history[-1]["activity"] if history else 1.0, 3),
               budget_remaining=round(world.budget_left, 1), active_policies=sorted(active),
               rumor_detected_recently=bool(any(0 <= world.day - r <= 14 for r in world.sc["rumors"])),
               cumulative_reported_deaths=int(pop.X["D"].sum()))
    if world.sc["conflicting"]:
        obs["second_source_daily_cases"] = round(reported * rng.choice([0.4, 2.2]), 1)
        obs["note"] = "Two surveillance sources disagree; neither has been validated."
    for h in history[-1:]: h["reported"] = reported
    return obs
