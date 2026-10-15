"""Figures for the report: the world map, an animated outbreak, and result charts (plain matplotlib)."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from world import load_config
from experiment_runner import run
from strategies import REFERENCE

cfg = load_config(); OUT = "../images/"
R = json.load(open("results/analysis.json"))["table"]
cities = cfg["cities"]; xs = [c["x"] for c in cities]; ys = [c["y"] for c in cities]
COL = {"Baseline (neutral rule)": "tab:gray", "Do nothing": "tab:red", "Early proportionate": "tab:green",
       "Strict lockdown": "tab:purple", "Economy first": "tab:orange"}

def draw_map(ax, inf=None, title=""):
    ax.set_facecolor("#eef3f7")
    ax.add_patch(plt.Polygon([[5, 5], [70, -5], [165, 30], [170, 140], [100, 150], [20, 125], [0, 60]], closed=True, fc="#e3ead8", ec="#9aa58f", lw=1.5))
    ax.plot([0, 60, 110], [40, 30, 0], color="#8fb3d9", lw=3, alpha=0.7)                       # river
    for a, b, w in cfg["travel_links"]:
        ax.plot([xs[a], xs[b]], [ys[a], ys[b]], color="#555", lw=1 + 120 * w, alpha=0.6, zorder=1)
    for i, c in enumerate(cities):
        size = c["population"] / 900
        col = "white" if inf is None else plt.cm.Reds(min(1.0, inf[i] * 40))
        ax.scatter(c["x"], c["y"], s=size, color=col, edgecolor="#333", lw=1.5, zorder=3)
        beds = int(c["beds_per_1000"] * c["population"] / 1000)
        ax.text(c["x"] + 6, c["y"] + 5, f"{c['name']}\n{c['population']:,} people\n{beds:,} hospital beds", fontsize=8.5, zorder=4)
        ax.scatter(c["x"] - 6, c["y"] - 7, marker="P", s=60, color="tab:red", zorder=4)
    ax.set_xlim(-5, 185); ax.set_ylim(-10, 160); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, loc="left", fontsize=11)

# ---- map of the fictional country (also the cover) ----
fig, ax = plt.subplots(figsize=(10, 8.6)); draw_map(ax, title="Meridia: four cities, transport links (line width = travel), hospitals (+)")
fig.tight_layout(); fig.savefig(OUT + "meridia_map.png", dpi=140); plt.close(fig)

# ---- animated outbreak: two strategies on the same seed ----
pair = ["Do nothing", "Early proportionate"]
H = {n: run(REFERENCE[n], n, "A_early_warning", 101, cfg, keep_history=True)[1] for n in pair}
fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
def frame(k):
    for ax, n in zip(axes, pair):
        ax.clear(); h = H[n][k]
        import textwrap
        pol = textwrap.fill(", ".join(h["active"]) if h["active"] else "none", 60)
        draw_map(ax, h["city_infectious"], f"{n}: day {k}\nactive policies: {pol}")
fa = FuncAnimation(fig, frame, frames=range(0, 180, 4))
fa.save(OUT + "outbreak_spread.gif", writer=PillowWriter(fps=6), dpi=80); plt.close(fig)

# ---- epidemic and hospital curves, scenario A, one seed ----
fig, ax = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)
for n, fn in REFERENCE.items():
    h = run(fn, n, "A_early_warning", 101, cfg, keep_history=True)[1]
    ax[0].plot([x["active_infectious"] for x in h], color=COL[n], label=n)
    ax[1].plot([100 * x["occupancy"].sum() / x["beds"].sum() for x in h], color=COL[n])
ax[1].axhline(100, color="k", ls="--", lw=1); ax[0].set_ylabel("people currently infectious"); ax[1].set_ylabel("hospital occupancy (%)")
ax[1].set_xlabel("day"); ax[0].legend(fontsize=9); ax[0].set_title("Scenario A (early warning), seed 101", loc="left")
fig.tight_layout(); fig.savefig(OUT + "curves_scenario_A.png", dpi=140); plt.close(fig)

# ---- deaths by scenario and strategy (means over 20 seeds) ----
sc = list(R); names = list(REFERENCE)
fig, ax = plt.subplots(figsize=(12, 5.5)); w = 0.16
for i, n in enumerate(names):
    ax.bar(np.arange(len(sc)) + (i - 2) * w, [R[s][n]["deaths"] for s in sc], w, color=COL[n], label=n)
ax.set_xticks(range(len(sc))); ax.set_xticklabels([s.replace("_", "\n", 1).replace("_", " ") for s in sc], fontsize=9)
ax.set_ylabel("deaths (mean of 20 seeds)"); ax.legend(fontsize=9, ncol=3); ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(OUT + "deaths_by_scenario.png", dpi=140); plt.close(fig)

# ---- health vs economy trade-off ----
fig, ax = plt.subplots(figsize=(9, 6.5))
for n in names:
    ax.scatter([100 * R[s][n]["economic_activity_retained"] for s in sc], [R[s][n]["deaths"] for s in sc], color=COL[n], label=n, s=55)
ax.set_xlabel("economic activity retained (%)"); ax.set_ylabel("deaths (mean of 20 seeds)"); ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT + "health_vs_economy.png", dpi=140); plt.close(fig)

# ---- trust under misinformation ----
fig, ax = plt.subplots(figsize=(10, 4.8))
for n, fn in REFERENCE.items():
    h = run(fn, n, "E_misinformation", 101, cfg, keep_history=True)[1]
    ax.plot([x["trust"] for x in h], color=COL[n], label=n)
for r in cfg["scenarios"]["E_misinformation"]["rumors"]: ax.axvline(r, color="k", ls=":", lw=1)
ax.set_ylabel("public trust (0-1)"); ax.set_xlabel("day (dotted lines = rumor waves)"); ax.legend(fontsize=9); ax.set_title("Scenario E (misinformation), seed 101", loc="left")
fig.tight_layout(); fig.savefig(OUT + "trust_misinformation.png", dpi=140); plt.close(fig)

# ---- fairness: older vs adult death ratio, and income burden ----
fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
for i, n in enumerate(names):
    ax[0].bar(np.arange(len(sc)) + (i - 2) * w, [R[s][n]["older_to_adult_death_ratio"] for s in sc], w, color=COL[n], label=n)
    ax[1].bar(np.arange(len(sc)) + (i - 2) * w, [100 * R[s][n]["income_loss_low"] for s in sc], w, color=COL[n])
for a, t in zip(ax, ["older adults' death rate / working-age death rate", "income loss of lower-income households (%)"]):
    a.set_xticks(range(len(sc))); a.set_xticklabels([s.split("_")[0] for s in sc]); a.set_title(t, loc="left", fontsize=10); a.grid(axis="y", alpha=0.3)
ax[0].legend(fontsize=8, loc='lower right', framealpha=0.9); ax[0].set_ylim(0, 5.5); fig.tight_layout(); fig.savefig(OUT + "fairness.png", dpi=140); plt.close(fig)
print("figures done")
