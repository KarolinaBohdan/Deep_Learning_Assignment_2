# Deep Learning Assignment 2 — Green Patent Detection with Human-in-the-Loop

A Human-in-the-Loop (HITL) active learning pipeline for classifying green patents using [PatentSBERTa](https://huggingface.co/AI-Growth-Lab/PatentSBERTa) and LLM-assisted labeling.

---

## Overview

This project builds a classifier that identifies *green* patents — patents related to renewable energy, energy efficiency, or carbon reduction (Y02 CPC classification codes). The workflow follows four stages:

| Part | Description |
|------|-------------|
| A | Baseline classifier using frozen PatentSBERTa embeddings + Logistic Regression |
| B | Uncertainty sampling to find the 100 most ambiguous examples from the unlabeled pool |
| C | LLM auto-labeling (Ollama/Qwen3) followed by human review (HITL) |
| D | Full fine-tuning of PatentSBERTa on the training set augmented with gold labels |

---

## Dataset

- **50k balanced patent dataset** — 25k green, 25k non-green  
- **Split**: 35k train · 7.5k eval · 7.5k unlabeled pool  
- Green patents are identified by Y02 CPC classification codes

---

## Methodology

### Part A — Baseline
Frozen PatentSBERTa embeddings (768-dim) were extracted and fed into a Logistic Regression classifier.

### Part B — Uncertainty Sampling
The 100 most uncertain predictions from the unlabeled pool were selected using entropy scoring:

```
u = 1 − 2 · |p − 0.5|
```

where `p` is the model's predicted probability. Scores closest to 1.0 indicate maximum uncertainty.

### Part C — LLM Labeling + Human-in-the-Loop

The 100 uncertain examples were auto-labeled by an LLM (Ollama/Qwen3). A human reviewer then inspected and corrected the labels, creating a *gold* set of 100 verified examples.

**Human corrections found 5 labeling errors by the LLM**, for example:

> **doc_id 9638427** — *"Apparatus for substantially blocking flames and spreading heated gases from a broiler flue…"*  
> LLM predicted: `0.0 (NOT_GREEN)` | Correct label: `1.0 (GREEN)`

> **doc_id 8948831** — *"Transmission system comprising: a superconductive cable having three phase conductors; and a cryostat surrounding the phase conductors…"*  
> LLM predicted: `0.0 (NOT_GREEN)` | Correct label: `1.0 (GREEN)`

Note: only 5 out of the 100 uncertain samples were actually green (heavily imbalanced), and 46 disagreed with the original silver labels.

### Part D — Fine-tuning

PatentSBERTa was fine-tuned end-to-end on the full training set, with gold labels substituted where available (1 epoch, lr=2e-5, batch size=16). The model was trained on the AI-lab GPU cluster.

---

## Results

| Stage | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Before fine-tuning (Part A) | 0.7916 | 0.7726 | 0.7820 |
| After fine-tuning (Part D) | 0.8341 | 0.7971 | 0.8153 |

Fine-tuning improved F1 by **+3.3 percentage points**.

---

## Repository Structure

```
├── assignment_2.ipynb          # Main notebook (Parts A–D)
├── train_part_d.py             # Standalone fine-tuning script for Part D
├── patents_50k_green.parquet   # Full dataset (50k patents)
├── hitl_green_100.csv          # 100 uncertain samples (silver labels)
├── hitl_green_100_label_only.csv
├── hitl_green_100_with_llm.csv # LLM-labeled version
├── hitl_green_100_gold.csv     # Gold labels after human review
├── X_eval.npy                  # Eval embeddings
├── y_eval.npy                  # Eval labels
└── y_train.npy                 # Train labels
```

---

## Requirements

- Python 3.8+
- `transformers`, `torch`, `sentence-transformers`
- `scikit-learn`, `pandas`, `numpy`
- Ollama (for LLM labeling in Part C)
 
