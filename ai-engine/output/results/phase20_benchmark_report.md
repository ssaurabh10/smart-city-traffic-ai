# Phase 20 Benchmark Results

Benchmark configuration:

- Simulation: `sumo/dhanbad.sumocfg`
- Warmup: 200 steps
- Measurement: 1000 steps
- Seed: 2026
- RL model: `ai-engine/output/models/ppo_traffic_model.zip`
- Scope: network-wide active vehicle metrics

| Strategy | Waiting time | Queue length | CO2 emissions | Throughput |
|---|---:|---:|---:|---:|
| PPO RL Multi-Agent | 203.152 s | 3.900 vehicles | 798.170 g/s | 594 vehicles |
| Webster Fixed-Cycle | 194.827 s | 3.485 vehicles | 798.442 g/s | 594 vehicles |
| Static Timing | 203.152 s | 3.900 vehicles | 798.170 g/s | 594 vehicles |

## RL vs Webster

| Metric | Result |
|---|---:|
| Waiting time improvement | -4.27% |
| Queue length improvement | -11.91% |
| Emissions improvement | +0.03% |
| Throughput improvement | +0.00% |

In this run, Webster fixed-cycle timing produced lower average waiting time and queue length than the current PPO policy. PPO and static timing were effectively identical on the measured episode, which suggests the deployed PPO policy is not yet changing enough phases to outperform the default SUMO program under this demand pattern.

## RL vs Static Timing

| Metric | Result |
|---|---:|
| Waiting time improvement | +0.00% |
| Queue length improvement | +0.00% |
| Emissions improvement | +0.00% |
| Throughput improvement | +0.00% |

## Artifacts

- Raw JSON: `ai-engine/output/results/phase20_benchmark.json`
- Plot: `ai-engine/output/plots/phase20_benchmark_bar.png`
