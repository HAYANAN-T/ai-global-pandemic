"""Population health states per city x group (compartmental, stochastic)."""
import numpy as np
from world import GROUPS

STATES = ["S", "E", "I", "H", "P", "D"]     # susceptible, exposed, infectious, hospitalised, temporarily protected, deceased

class Population:
    def __init__(self, world):
        n_c, n_g = world.n_c, world.n_g
        self.X = {s: np.zeros((n_c, n_g), int) for s in STATES}
        self.X["S"] = world.N.copy()
        seeds = world.sc["initial_infected"]
        adults = GROUPS.index("adult_other")
        alloc = world.rng.multinomial(seeds, world.N[:, adults] / world.N[:, adults].sum())
        self.X["S"][:, adults] -= alloc
        self.X["E"][:, adults] += alloc
        self.cum_infections = np.zeros((n_c, n_g), int)
        self.cum_infections[:, adults] += alloc
        self.cum_deaths_no_bed = 0

    def total(self):
        return sum(self.X[s] for s in STATES)
