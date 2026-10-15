"""Run one advisor in one scenario with one seed. The same engine and the same observations for every advisor."""
import json, hashlib, numpy as np
from world import World, load_config, GROUPS
from population import Population
from disease_model import step, contact_matrices
from policy_engine import PolicyState, modifiers, update_society, observe
from metrics import summarise

def run(advisor_fn, advisor_name, scenario, seed, cfg=None, keep_history=False):
    cfg = cfg or load_config()
    world = World(cfg, scenario, seed); pop = Population(world); C = contact_matrices(cfg)
    policy = PolicyState(); state = {}; hist = []; obs_hashes = []
    days, interval = cfg["simulation"]["days"], cfg["simulation"]["decision_interval"]
    out = None
    for day in range(days):
        world.day = day
        if day % interval == 0:
            obs = observe(world, pop, out, hist, policy.active | {a for _, a in policy.pending})
            obs_hashes.append(hashlib.sha256(json.dumps(obs, sort_keys=True).encode()).hexdigest()[:12])
            decision = advisor_fn(obs, state)
            policy.apply_decision(world, decision, advisor_name)
        cost = policy.tick(world)
        mods, comp = modifiers(world, policy.active)
        out = step(world, pop, mods, C)
        adults = [GROUPS.index("adult_low"), GROUPS.index("adult_other")]
        sick = (pop.X["I"][:, adults].sum() + pop.X["H"][:, adults].sum()) / world.N[:, adults].sum()
        activity, il_low, il_other = update_society(world, policy.active, out, sick)
        hist.append(dict(day=day, new_symptomatic=out["new_symptomatic"], new_infections=out["new_infections"],
                         occupancy=out["occupancy"], beds=out["beds"], activity=activity, income_loss_low=il_low,
                         income_loss_other=il_other, cost=cost, trust=world.trust, compliance=comp,
                         active_infectious=int(pop.X["I"].sum()), active=sorted(policy.active),
                         city_infectious=(pop.X["I"].sum(1) / np.maximum(world.N.sum(1), 1)).tolist()))
    m = summarise(world, pop, hist, policy)
    m.update(advisor=advisor_name, scenario=scenario, seed=seed, obs_hashes=obs_hashes)
    return (m, hist, policy.log) if keep_history else m
