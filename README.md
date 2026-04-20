# NeuroSearch

**Automated Neural Architecture Search Engine**

A NAS framework that automatically discovers high-performing image classification architectures from a modular DAG-based search space. Combines evolutionary algorithms with Bayesian optimization and uses a surrogate model to estimate architecture performance without full training, significantly reducing search time.

---

## Results

| Metric | Value |
|--------|-------|
| Accuracy on MNIST | 90%+ |
| Surrogate speedup | 5-15x vs full training per candidate |
| Search strategy | Evolutionary + Bayesian hybrid |

---

## Overview

NeuroSearch treats architecture design as an optimization problem over a directed acyclic graph search space. Two complementary strategies drive the search. Evolutionary algorithms explore broadly through mutation and selection. Bayesian optimization focuses the search on high-promising regions using a probabilistic model of performance. A surrogate predictor estimates architecture quality without full training, making the search feasible on standard hardware.

---

## Architecture

### Search Space

Architectures are represented as directed acyclic graphs where each node is a computational operation such as convolution, pooling, batch normalization, or skip connection. Edges define data flow between operations.

### Search Pipeline

```
DAG Search Space
      |
      v
Evolutionary Algorithm + Bayesian Optimization
      |
      v
Surrogate Predictor (performance estimation without full training)
      |
      v
Evaluation Pipeline (training + benchmarking)
      |
      v
Visualization + Model Export
```

---

## Quick Start

```bash
git clone https://github.com/Jaismali/NeuroSearch
cd NeuroSearch
pip install -r requirements.txt

python main.py
```

---

## Tech Stack

- Python 3.8+
- PyTorch / TensorFlow
- Evolutionary algorithms
- Bayesian optimization
- Matplotlib

---

## Platform

Python 3.8+, Windows / macOS / Linux. GPU optional. Surrogate-accelerated search runs feasibly on CPU.
