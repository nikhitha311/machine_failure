# Robust Machine Failure Prediction Under Operating Condition Shift
## SDG 9 Alignment: Industry, Innovation & Infrastructure

## Problem Statement

Industrial machines generate continuous sensor data — air temperature,
process temperature, rotational speed (RPM), torque, and tool wear.
Unexpected machine failures cause:

- Production downtime (₹10+ lakhs per hour for large factories)
- Emergency maintenance costs (3x planned maintenance)
- Safety risks for operators
- Material waste from aborted production runs

The core challenge: Standard ML models predict failure well on training
data but degrade severely when deployment operating conditions differ from
training. A model trained on summer data fails silently when winter shifts
temperature by 5-10%.

Our solution: Predict machine failure AND measure robustness under
operating condition shift — a critical gap in real-world predictive maintenance.
Impact:
- Prevents unexpected machine failures in manufacturing
- Reduces production downtime and maintenance costs
- Extends equipment lifespan through early intervention
- Open-source implementation for community adoption

**A factory with 100 machines can prevent ~30 failures/year, saving ~₹30 lakhs
in downtime and repairs — while extending equipment life.**

![CI](https://github.com/nikhitha311/machine_failure/actions/workflows/ci.yml/badge.svg)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

**OptiForge 2026** · Team **code crafters** (OPT-26-3407) · Track: ML & AI

> Predict machine failure from sensor data **before** breakdown — and stay reliable when operating conditions shift from training distribution.

## Live Demo

- **API Health:** https://machine-failure-api-5ev9.onrender.com/health
- **Swagger Docs:** https://machine-failure-api-5ev9.onrender.com/docs
- **Web UI:** https://machine-failure-api-5ev9.onrender.com/static/index.html

## Innovation: Robustness Under Distribution Shift

| Test | F1 Score | Change |
|------|----------|--------|
| Normal (Baseline) | 0.68 | — |
| Shifted (Baseline) | 0.34 | **-50%** |
| Shifted (Robust) | 0.54 | **+59%** |

We simulate realistic deployment shifts:
- Air temperature +5-10%
- Process temperature +4-6%
- Torque +8-12%
- RPM -3-7%
- Tool wear +8-12%

Our robust training recovers shifted F1 from 0.34 to 0.54 — a 59% relative improvement.

## Results

- Precision: 0.90
- Recall: 0.54
- F1 Score: 0.68 (normal)
- ROC-AUC: 0.97
- Inference Time: ~62 ms
- Unit Tests: 7/7 passing

## Why F1/Recall > Accuracy?

Failure is a 3.4% minority class. An "always normal" classifier gets 96.6% accuracy but F1 = 0. Missing a failure costs more than a false alarm.

## Quick Start

```bash
git clone https://github.com/nikhitha311/machine_failure.git
cd machine_failure
pip install -r requirements.txt
python main.py                    # CLI demo
uvicorn api.main:app --reload     # API server
pytest                            # Run tests
