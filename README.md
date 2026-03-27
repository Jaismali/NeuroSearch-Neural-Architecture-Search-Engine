## 🚀 NeuroSearch: Neural Architecture Search Engine

NeuroSearch is an automated Neural Architecture Search (NAS) framework designed to discover high-performing image classification models with minimal manual intervention.

### 🧠 Overview
This project implements an intelligent NAS system that explores a modular, DAG-based search space to automatically design efficient neural network architectures. By combining evolutionary algorithms and Bayesian optimization, NeuroSearch identifies high-quality models while balancing performance and computational cost.

### ⚙️ Key Features
- 🔍 **Automated Architecture Design**: Generates image-classification models from a modular DAG-based search space.
- 🧬 **Hybrid Search Strategy**: Combines evolutionary algorithms with Bayesian optimisation for efficient exploration.
- ⚡ **Surrogate Model Acceleration**: Uses a predictive model to estimate performance and significantly reduce training time.
- 📊 **High Performance**: Achieves **90%+ accuracy on MNIST** with optimized architectures.
- 📈 **Visualisation Tools**: Provides clear visual insights into architecture evolution and search progress.
- 📦 **Model Export**: Supports exporting best-performing architectures for deployment or further training.

### 🏗️ Architecture
- Search Space: Directed Acyclic Graph (DAG)-based modular architecture design
- Search Engine: Evolutionary + Bayesian optimisation
- Surrogate Predictor: Fast performance estimation
- Evaluation Pipeline: Training and benchmarking on datasets (e.g., MNIST)
- Visualisation Layer: Tracks and displays search progress

### 🚀 Use Cases
- Automated model design for image classification
- Research in Neural Architecture Search (NAS)
- Efficient experimentation in deep learning
- Rapid prototyping of neural networks

### 🔧 Tech Stack
- Python
- PyTorch / TensorFlow
- Optimisation Algorithms (Evolutionary + Bayesian)
- Visualization Libraries

### 📈 Results
- ✅ Achieved **90%+ accuracy on MNIST**
- ⚡ Reduced search time using surrogate-based evaluation
- 📊 Improved interpretability with visualised search dynamics

### 🤝 Contribution
Contributions, ideas, and feedback are welcome!

---
