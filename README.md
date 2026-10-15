# Meridia: a simulated pandemic world for testing AI advisors

A fictional country, a fictional disease ("Velora fever"), and a policy engine that decides what every recommendation actually does.
ChatGPT, Claude and Gemini plug in as advisors through `llm_adapters.py`. Nothing here models a real pathogen.

## Install
```
pip install numpy pyyaml matplotlib pillow pytest
```

## Reproduce everything that has been run
```
python calibrate.py                   # sets the fictional transmission scale so R0 = 2.3 (already applied)
python run_reference_experiments.py   # 7 scenarios x 5 non-AI strategies x 20 seeds = 700 runs (~30 s)
python results_analysis.py            # paired comparisons vs the baseline, bootstrap 95% intervals
python make_figures.py                # map, outbreak GIF and charts -> ../images/
python dashboard.py                   # interactive dashboard -> results/dashboard.html
python -m pytest -q tests             # 13 tests
```

## Run the AI advisors (requires your own API keys)
```
export OPENAI_API_KEY=...    OPENAI_MODEL=<explicit model id>
export ANTHROPIC_API_KEY=... ANTHROPIC_MODEL=<explicit model id>
export GEMINI_API_KEY=...    GEMINI_MODEL=<explicit model id>
python run_llm_experiments.py --seeds 5     # 7 scenarios x 5 seeds x 13 decisions = 455 calls per provider
```
Providers without credentials are recorded as `NOT RUN` in `results/llm_runs.json`. Responses are never simulated.
Every response is validated against a strict schema; invalid or unaffordable actions are rejected and logged.
Afterwards, re-run `results_analysis.py` with `results/llm_runs.json` added to compare the models with the baseline on matching seeds.

## Status of this release
| component | status |
|---|---|
| World, disease model, policy engine, metrics | implemented and tested |
| Baseline + 4 reference strategies, 7 scenarios, 20 seeds | **run** (results/reference_runs.json) |
| OpenAI, Anthropic, Gemini adapters | implemented; schema, failure and missing-key paths unit-tested |
| ChatGPT, Claude, Gemini experiments | **NOT RUN** in this release: no API credentials were available, and the OpenAI and Google APIs were unreachable from the build environment |
| Multi-model ensemble (combining recommendations) | designed, not run (needs the model runs) |

## Pre-registered metrics (defined in `metrics.py` before any experiment)
Health: total infections, attack rate (infections per person, including re-infections), peak active infectious, day of peak, deaths, deaths without a bed, days any city's hospitals exceeded capacity, peak occupancy.
Economy: mean economic activity retained, mean income loss (lower-income and other households), public expenditure.
Social: final and minimum public trust, mean compliance, older-to-working-age death-rate ratio, lower-income income-burden ratio.
Decision quality: rejected (invalid or unaffordable) actions, advisor failures. For LLM runs also: calls, failures, latency.
No composite score is used; strategies are compared metric by metric.

## Files
config.yaml, world.py, population.py, disease_model.py, policy_engine.py, strategies.py, llm_adapters.py,
experiment_runner.py, metrics.py, results_analysis.py, run_reference_experiments.py, run_llm_experiments.py,
make_figures.py, dashboard.py, calibrate.py, tests/.

## Limitations
The world's outcomes follow rules written by the author (for example, how much transparent communication restores trust).
Results show how strategies behave under these assumptions, not how real policies work.
