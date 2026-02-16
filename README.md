# Credit Cycles and Capital Misallocation
Empirical macro-financial pipeline combining econometrics and machine learning to detect regime shifts and intertemporal distortions associated with credit expansions.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](#)
[![Reproducible](https://img.shields.io/badge/Reproducible-pipeline-success)](#)
[![Data](https://img.shields.io/badge/Data-FRED%2C%20Yahoo%2C%20Shiller-lightgrey)](#)

## At a glance
This repository builds a monthly macro-financial dataset (2000–2025) and prepares it for econometric and ML models aimed at characterizing credit-cycle dynamics and market regimes.

Research question. Can monetary and credit conditions generate measurable distortions that are detectable through time-series econometrics and supervised or unsupervised learning?

Core methods. VAR and VECM for dynamics and long-run relationships, GARCH-type volatility modeling, regime identification and classification, and predictive models (tree ensembles and sequence models).

Key output. A unified monthly panel with levels and transformations, saved as `data/processed/monthly_data.csv` and `data/processed/monthly_data.pkl`.

## Repository structure
The project is organized as a reproducible pipeline. Raw data are never modified, processed datasets are generated deterministically by scripts.

