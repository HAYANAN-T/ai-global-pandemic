"""World state: cities, groups, transport links and the simulation clock."""
import numpy as np, yaml

GROUPS = ["child", "adult_low", "adult_other", "older"]

def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

class World:
    def __init__(self, cfg, scenario_name, seed):
        self.cfg, self.sc_name = cfg, scenario_name
        self.sc = cfg["scenarios"][scenario_name]
        self.rng = np.random.default_rng(seed)
        self.day = 0
        self.cities = cfg["cities"]
        self.n_c, self.n_g = len(self.cities), len(GROUPS)
        shares = np.array([cfg["groups"][g] for g in GROUPS])
        pop = np.array([c["population"] for c in self.cities], float)
        self.N = np.round(np.outer(pop, shares)).astype(int)                       # people per city x group
        self.beds = np.array([c["beds_per_1000"] * c["population"] / 1000 for c in self.cities]) * self.sc["beds_mult"]
        self.travel = np.eye(self.n_c)
        for a, b, w in cfg["travel_links"]:
            self.travel[a, b] = self.travel[b, a] = w
        self.travel /= self.travel.sum(1, keepdims=True)
        self.trust = float(self.sc["trust0"])
        self.budget_left = float(cfg["simulation"]["budget"])
        self.econ_mult = float(self.sc["econ_mult"])
