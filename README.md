# shodh-ai-loan

# Lending Club Loan Approval: DL & Offline RL Project

This repository contains all code, data processing steps, and models for predicting loan defaults and learning optimal loan approval policies using both Deep Learning (DL) and Offline Reinforcement Learning (RL).

---

## Project Overview

The goal of this project is to:

1. Predict loan defaults using a supervised Deep Learning model.
2. Learn an optimal loan approval policy using an offline RL agent (CQL).
3. Compare the performance of DL and RL policies in terms of financial returns and risk.

---

## Folder Structure

Project Insights

DL Model: Evaluates prediction accuracy for defaults (AUC/F1-score).

RL Model: Optimizes expected financial returns (Estimated Policy Value).

Policy Differences: RL may approve borderline high-risk loans if expected reward outweighs potential loss.

Limitations & Future Work

One-step episode assumption for RL ignores long-term repayment behavior.

Limited features may miss borrower behavior signals.

Future work: hybrid RL+DL models, more offline RL algorithms (SAC, IQL), threshold tuning for DL, additional features.
