"""Paired comparisons against the baseline across matching seeds, with bootstrap 95% intervals."""
import json, numpy as np
from collections import defaultdict

METRICS = ["deaths", "attack_rate", "overload_days_any_city", "peak_occupancy_pct", "economic_activity_retained",
           "income_loss_low", "trust_end", "older_to_adult_death_ratio", "rejected_actions", "expenditure"]

def paired(rows, metric, a, b, scenario, n_boot=4000, seed=0):
    A = {r["seed"]: r[metric] for r in rows if r["advisor"] == a and r["scenario"] == scenario}
    B = {r["seed"]: r[metric] for r in rows if r["advisor"] == b and r["scenario"] == scenario}
    seeds = sorted(set(A) & set(B)); d = np.array([A[s] - B[s] for s in seeds], float)
    rng = np.random.default_rng(seed); boots = [rng.choice(d, len(d)).mean() for _ in range(n_boot)]
    sd = d.std(ddof=1) if len(d) > 1 else 0.0
    return dict(mean_diff=float(d.mean()), ci=(float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))),
                effect_size=float(d.mean() / sd) if sd > 0 else 0.0, n=len(d))

def summary(rows):
    out = defaultdict(dict)
    for r in rows: out[(r["scenario"], r["advisor"])].setdefault("_rows", []).append(r)
    table = {}
    for (sc, adv), v in out.items():
        table.setdefault(sc, {})[adv] = {m: float(np.mean([r[m] for r in v["_rows"]])) for m in METRICS}
    return table

if __name__ == "__main__":
    rows = json.load(open("results/reference_runs.json"))["runs"]
    table = summary(rows); comps = {}
    for sc in table:
        for adv in table[sc]:
            if adv == "Baseline (neutral rule)": continue
            comps.setdefault(sc, {})[adv] = {m: paired(rows, m, adv, "Baseline (neutral rule)", sc) for m in ["deaths", "economic_activity_retained", "trust_end"]}
    json.dump(dict(table=table, vs_baseline=comps), open("results/analysis.json", "w"), indent=1)
    for sc in table:
        print("\n==", sc)
        for adv, m in table[sc].items():
            c = comps.get(sc, {}).get(adv, {}).get("deaths")
            ci = f"  deaths vs baseline {c['mean_diff']:+.0f} [{c['ci'][0]:+.0f}, {c['ci'][1]:+.0f}]" if c else ""
            print(f"{adv[:22]:22} deaths {m['deaths']:6.0f} attack {m['attack_rate']:.2f} overload {m['overload_days_any_city']:4.1f} econ {m['economic_activity_retained']:.3f} trust {m['trust_end']:.2f} older/adult {m['older_to_adult_death_ratio']:.1f} rejected {m['rejected_actions']:.1f}{ci}")
