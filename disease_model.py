"""Abstract, fictional SEIR-type transitions with hospital capacity. Independent of any advisor."""
import numpy as np
from world import GROUPS

SETTINGS = ["home", "school", "work", "community"]

def contact_matrices(cfg):
    return {s: np.array([cfg["contacts"][s][g] for g in GROUPS]) for s in SETTINGS}

def step(world, pop, mods, C):
    """Advance one day. mods: setting multipliers, isolation strength, protection of older adults."""
    d = world.cfg["disease"]; rng = world.rng
    X = pop.X; N = world.N.astype(float)
    alive = np.maximum(N - X["D"], 1)
    prev = X["I"] / alive                                                    # infectious share by city x group
    mixed = world.travel @ prev                                              # travel mixes prevalence between linked cities
    Ceff = sum(mods["setting"][s] * C[s] for s in SETTINGS)                  # group x group daily contacts
    Ceff = Ceff.copy(); older = GROUPS.index("older")
    Ceff[older, :] *= mods["older_protection"]; Ceff[:, older] *= mods["older_protection"]
    lam = d["beta"] * world.sc["beta_mult"] * (mixed * (1 - mods["isolation"])) @ Ceff.T   # force of infection
    p_inf = 1 - np.exp(-lam)
    new_E = rng.binomial(X["S"], np.clip(p_inf, 0, 1))
    new_I = rng.binomial(X["E"], 1 / d["incubation_days"])
    leave_I = rng.binomial(X["I"], 1 / d["infectious_days"])
    hp = np.array([d["hosp_prob"][g] for g in GROUPS])[None, :]
    to_H = rng.binomial(leave_I, np.broadcast_to(hp, leave_I.shape))
    to_P_from_I = leave_I - to_H
    leave_H = rng.binomial(X["H"], 1 / d["hospital_days"])
    # beds: occupancy above capacity raises fatality for the overflow share in each city
    occ = X["H"].sum(1); beds = world.beds * mods["bed_mult"]
    overflow_share = np.clip((occ - beds) / np.maximum(occ, 1), 0, 1)[:, None]
    fat = d["fatality_with_bed"] * (1 - overflow_share) + d["fatality_without_bed"] * overflow_share
    die = rng.binomial(leave_H, np.clip(fat, 0, 1))
    pop.cum_deaths_no_bed += int((die * overflow_share).sum())
    to_P_from_H = leave_H - die
    wane = rng.binomial(X["P"], 1 / d["protection_days"])
    X["S"] += -new_E + wane
    X["E"] += new_E - new_I
    X["I"] += new_I - leave_I
    X["H"] += to_H - leave_H
    X["P"] += to_P_from_I + to_P_from_H - wane
    X["D"] += die
    pop.cum_infections += new_E
    return dict(new_infections=int(new_E.sum()), new_symptomatic=int(new_I.sum()), occupancy=occ, beds=beds,
                deaths=die, new_hosp=int(to_H.sum()))
