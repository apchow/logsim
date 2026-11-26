# Demo Walkthrough (`logisticsbt`)

## 1. Generate & Validate Synthetic Reverse-Logistics Dataset

```bash
logisticsbt ingest --synthetic --seed 42 --orders-per-day 110 --output data/synthetic/default
logisticsbt validate --dataset data/synthetic/default
```

## 2. Train the Compact PPO Policy

```bash
logisticsbt train --timesteps 25000 --seed 42 --dataset data/synthetic/default
```

## 3. Run Multi-Seed Paired Backtest (5 Seeds x 3 Scenarios x Policies)

```bash
logisticsbt backtest \
  --policies current_replay,fifo_edf,nearest_resource,fixed_batch,greedy_cost,ortools_optimizer,ml_risk_scorer,ppo_agent \
  --scenarios baseline_normal,peak_surge_breakdown,storm_tight_sla \
  --seeds 101,202,303,404,505 \
  --experiment reverse_mvp_demo \
  --baseline current_replay
```

## 4. Generate Reports & Exports

```bash
logisticsbt report --experiment reverse_mvp_demo --baseline current_replay
logisticsbt export --experiment reverse_mvp_demo --format parquet
logisticsbt export --experiment reverse_mvp_demo --format csv
```

## 5. Launch Dashboard or API

```bash
python -m app.dashboard.gradio_app
streamlit run app/dashboard/main.py
uvicorn app.api.main:app --port 8000
```
