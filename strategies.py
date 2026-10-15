"""Non-AI advisors: the predefined neutral baseline and transparent reference strategies."""

def baseline_neutral(obs, state):
    """The brief's neutral strategy: inform, advise isolation, and restrict only when hospitals are nearly full."""
    acts = ["info_campaign", "isolation_guidance"]
    if obs["hospital_occupancy_pct"] > 90: state["restrict"] = True
    if obs["hospital_occupancy_pct"] < 50: state["restrict"] = False
    if state.get("restrict"): acts += ["limit_gatherings", "remote_work"]
    return dict(actions=acts, rationale="predefined neutral rule", uncertainty="n/a")

def do_nothing(obs, state):
    return dict(actions=[], rationale="no intervention", uncertainty="n/a")

def early_proportionate(obs, state):
    acts = ["info_campaign", "testing_monitoring", "isolation_guidance", "protect_vulnerable", "trust_communication"]
    occ = obs["hospital_occupancy_pct"]
    if occ > 35 or obs["reported_daily_cases_7d"] > 150: state["level"] = 1
    if occ < 15 and obs["reported_daily_cases_7d"] < 60: state["level"] = 0
    if state.get("level"): acts += ["limit_gatherings", "remote_work", "financial_support"]
    if occ > 60: acts.append("expand_capacity")
    return dict(actions=acts, rationale="early, targeted measures; escalate on hospital pressure", uncertainty="thresholds fixed in advance")

def strict_lockdown(obs, state):
    if obs["reported_daily_cases_7d"] > 0 and "start" not in state: state["start"] = obs["day"]
    acts = ["info_campaign", "isolation_guidance"]
    if "start" in state and obs["day"] - state["start"] < 70:
        acts += ["limit_gatherings", "remote_work", "close_schools"]
    return dict(actions=acts, rationale="hard restrictions for 70 days from first detection", uncertainty="n/a")

def economy_first(obs, state):
    acts = ["info_campaign"]
    if obs["hospital_occupancy_pct"] > 50: acts.append("expand_capacity")
    return dict(actions=acts, rationale="keep the economy open; add beds when needed", uncertainty="n/a")

REFERENCE = {"Baseline (neutral rule)": baseline_neutral, "Do nothing": do_nothing,
             "Early proportionate": early_proportionate, "Strict lockdown": strict_lockdown,
             "Economy first": economy_first}
