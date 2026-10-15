"""Metrics, defined before any experiment was run (see README: 'Pre-registered metrics')."""
import numpy as np
from world import GROUPS

def summarise(world, pop, hist, policy):
    occ = np.array([h["occupancy"].sum() for h in hist]); beds = np.array([h["beds"].sum() for h in hist])
    city_over = np.array([(h["occupancy"] > h["beds"]).any() for h in hist])
    active = np.array([h["active_infectious"] for h in hist])
    N = world.N; D = pop.X["D"]
    older, adult = GROUPS.index("older"), [GROUPS.index("adult_low"), GROUPS.index("adult_other")]
    death_rate_older = D[:, older].sum() / N[:, older].sum()
    death_rate_adults = D[:, adult].sum() / N[:, adult].sum()
    act = np.array([h["activity"] for h in hist])
    return {
        "infections_total": int(pop.cum_infections.sum()),
        "attack_rate": float(pop.cum_infections.sum() / N.sum()),
        "peak_active_infectious": int(active.max()),
        "day_of_peak": int(active.argmax()),
        "deaths": int(D.sum()),
        "deaths_without_bed": int(pop.cum_deaths_no_bed),
        "overload_days_any_city": int(city_over.sum()),
        "peak_occupancy_pct": float(100 * (occ / beds).max()),
        "economic_activity_retained": float(act.mean()),
        "income_loss_low": float(np.mean([h["income_loss_low"] for h in hist])),
        "income_loss_other": float(np.mean([h["income_loss_other"] for h in hist])),
        "expenditure": float(sum(h["cost"] for h in hist)),
        "trust_end": float(world.trust),
        "trust_min": float(min(h["trust"] for h in hist)),
        "compliance_mean": float(np.mean([h["compliance"] for h in hist])),
        "older_to_adult_death_ratio": float(death_rate_older / max(death_rate_adults, 1e-9)),
        "income_burden_ratio_low_vs_other": float(np.mean([h["income_loss_low"] for h in hist]) / max(np.mean([h["income_loss_other"] for h in hist]), 1e-9)),
        "rejected_actions": int(sum(len(l["rejected"]) for l in policy.log)),
        "advisor_failures": int(sum(1 for l in policy.log if "failed" in l.get("rationale", ""))),
    }
