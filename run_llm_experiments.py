"""Run ChatGPT, Claude and Gemini advisors on every scenario and seed, under identical conditions.

Usage:
  export OPENAI_API_KEY=...    OPENAI_MODEL=<explicit model id>
  export ANTHROPIC_API_KEY=... ANTHROPIC_MODEL=<explicit model id>
  export GEMINI_API_KEY=...    GEMINI_MODEL=<explicit model id>
  python run_llm_experiments.py [--seeds 5]

Providers without credentials are recorded as NOT RUN. No responses are ever simulated or invented.
Each run makes 13 decisions (one every 14 simulated days); 7 scenarios x 5 seeds = 455 calls per provider.
"""
import argparse, json, time
from world import load_config
from experiment_runner import run
import llm_adapters as L

ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=5); args = ap.parse_args()
cfg = load_config(); seeds = cfg["simulation"]["seeds"][: args.seeds]
status, rows, logs = {}, [], {}
for provider in L.PROVIDERS:
    try:
        adapter = L.make(provider)
    except L.NotConfigured as e:
        status[provider] = f"NOT RUN: {e}"; print(provider, status[provider]); continue
    status[provider] = f"RUN with model {adapter.model} at {time.strftime('%Y-%m-%d %H:%M:%S')}"
    for sc in cfg["scenarios"]:
        for seed in seeds:
            m, hist, log = run(lambda obs, st: adapter.decide(obs), adapter.name, sc, seed, cfg, keep_history=True)
            m["model_id"] = adapter.model; rows.append(m); logs[f"{provider}|{sc}|{seed}"] = log
    status[provider] += f"; calls={adapter.calls}, failures={adapter.failures}, mean latency={sum(adapter.latency)/max(len(adapter.latency),1):.2f}s"
json.dump(dict(status=status, runs=rows, decision_logs=logs), open("results/llm_runs.json", "w"), indent=1)
print(json.dumps(status, indent=1))
