# Kaggle Playground Series S6E2 - Heart Disease Prediction

This repository contains the ongoing work, scripts, and MLOps automation pipeline for the [Kaggle Playground Series - Season 6, Episode 2](https://www.kaggle.com/competitions/playground-series-s6e2) competition.

**Status**: Post-Deadline Study & Portfolio Building
**Highest Private Score**: 0.95506 (vs #1 Score 0.95535)

## Directory Structure
- `docs/`: Contains dynamic performance summaries and analysis (see [Portfolio Summary](docs/portfolio_summary.md)). Old static reports have been moved to `docs/archive/`.
- `src/`: Python source code for data engineering, modeling, and ensemble methods. (Currently undergoing refactoring for modularity).
- `notebooks/`: Kaggle kernel metadata and runner scripts for CI/CD Kaggle Cloud Execution.

## Focus
We are actively building an end-to-end pipeline connecting GitHub with Kaggle Notebooks using the `kaggle` CLI to automate submissions, evaluate Late Submissions, and reproduce top-tier Kaggle strategies.
