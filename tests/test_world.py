import json, copy, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.chdir(os.path.dirname(os.path.dirname(__file__)))
import numpy as np, pytest
from world import load_config, World
from population import Population, STATES
from disease_model import step, contact_matrices
from policy_engine import PolicyState, ACTIONS, modifiers
from experiment_runner import run
from strategies import REFERENCE, baseline_neutral
import llm_adapters as L

CFG = load_config()

def test_population_is_conserved():
    w = World(CFG, "B_rapid_growth", 1); p = Population(w); C = contact_matrices(CFG)
    mods, _ = modifiers(w, set())
    for d in range(60):
        w.day = d; step(w, p, mods, C)
        assert (p.total() == w.N).all()
        assert all((p.X[s] >= 0).all() for s in STATES)

def test_same_seed_is_reproducible():
    a = run(baseline_neutral, "b", "E_misinformation", 7, CFG); b = run(baseline_neutral, "b", "E_misinformation", 7, CFG)
    assert a == b

def test_every_advisor_sees_identical_first_dashboard():
    hashes = {run(fn, n, "A_early_warning", 3, CFG)["obs_hashes"][0] for n, fn in REFERENCE.items()}
    assert len(hashes) == 1

def test_unknown_and_unaffordable_actions_are_rejected():
    w = World(CFG, "A_early_warning", 1); ps = PolicyState()
    acc, rej = ps.apply_decision(w, {"actions": ["teleport_everyone", "info_campaign"]}, "t")
    assert acc == ["info_campaign"] and rej[0][1] == "unknown action"
    w.budget_left = 1.0
    acc, rej = ps.apply_decision(w, {"actions": ["financial_support"]}, "t")
    assert acc == [] and rej[0][1] == "exceeds remaining budget"

def test_full_hospital_raises_fatality():
    w = World(CFG, "C_limited_capacity", 2); p = Population(w); C = contact_matrices(CFG)
    p.X["H"][:] = (w.N * 0.05).astype(int); p.X["S"] -= p.X["H"]
    mods, _ = modifiers(w, set()); step(w, p, mods, C)
    assert p.cum_deaths_no_bed > 0

def test_zero_population_city_does_not_crash():
    cfg = copy.deepcopy(CFG); cfg["cities"][3]["population"] = 0
    m = run(baseline_neutral, "b", "A_early_warning", 1, cfg); assert m["deaths"] >= 0

@pytest.mark.parametrize("text,ok", [('{"actions": ["info_campaign"], "rationale": "r", "uncertainty": "u"}', True),
                                     ('```json\n{"actions": [], "rationale": "r", "uncertainty": "u"}\n```', True),
                                     ('{"actions": "info_campaign", "rationale": "r", "uncertainty": "u"}', False),
                                     ('not json at all', False), ('{"actions": []}', False)])
def test_schema_validation(text, ok):
    if ok: assert isinstance(L.validate(text)["actions"], list)
    else:
        with pytest.raises(L.InvalidResponse): L.validate(text)

def test_missing_key_is_reported_not_faked(monkeypatch):
    for p in L.PROVIDERS:
        monkeypatch.delenv(L.PROVIDERS[p][0].env, raising=False)
        with pytest.raises(L.NotConfigured): L.make(p)

def test_api_failure_falls_back_safely(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy"); monkeypatch.setenv("ANTHROPIC_MODEL", "dummy-model")
    a = L.make("anthropic")
    def boom(prompt): raise TimeoutError("simulated timeout")
    a._call = boom; monkeypatch.setattr(L.time, "sleep", lambda s: None)
    d = a.decide({"day": 0}); assert d["actions"] == [] and d.get("failed")
