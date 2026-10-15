"""Run the baseline and every reference strategy on every scenario and seed. Saves results/reference_runs.json."""
import json, time
from world import load_config
from experiment_runner import run
from strategies import REFERENCE

cfg = load_config(); rows = []; t0 = time.time()
for sc in cfg["scenarios"]:
    for name, fn in REFERENCE.items():
        for seed in cfg["simulation"]["seeds"]:
            rows.append(run(fn, name, sc, seed, cfg))
json.dump(dict(generated=time.strftime("%Y-%m-%d %H:%M:%S"), runs=rows), open("results/reference_runs.json", "w"), indent=0)
print(len(rows), "runs in", round(time.time() - t0, 1), "s")
