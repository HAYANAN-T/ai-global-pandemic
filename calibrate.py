"""Set the fictional transmission scale so the basic reproduction number R0 equals a target (next-generation matrix)."""
import numpy as np, yaml
from world import load_config, GROUPS
from disease_model import contact_matrices, SETTINGS

def r0(cfg, beta=None):
    C = contact_matrices(cfg); Ceff = sum(C[s] for s in SETTINGS)
    beta = cfg["disease"]["beta"] if beta is None else beta
    # infections caused in group i by one infectious person in group j over their infectious period
    K = beta * cfg["disease"]["infectious_days"] * Ceff
    return float(max(abs(np.linalg.eigvals(K))))

if __name__ == "__main__":
    cfg = load_config(); target = 2.3
    beta = target / r0(cfg, 1.0)
    print("current R0", round(r0(cfg), 2), "-> beta for R0=2.3:", round(beta, 5))
    text = open("config.yaml").read()
    import re
    text = re.sub(r"beta: [0-9.]+ ", f"beta: {beta:.5f} ", text)
    text = text.replace("budget: 420 ", "budget: 650 ")
    open("config.yaml", "w").write(text)
    print("new R0", round(r0(load_config()), 2))
