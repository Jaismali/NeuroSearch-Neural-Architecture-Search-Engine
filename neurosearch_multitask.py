"""
NeuroSearch - Multi-Task Neural Architecture Search Engine
===========================================================
Automatically designs the best neural network for YOUR chosen task.

Just run:  python neurosearch_multitask.py
An interactive menu will appear and guide you through everything.

Requirements:
    pip install torch torchvision numpy matplotlib scikit-learn scipy pandas

Supported Tasks:
    1. Handwritten Digit Recognition (MNIST)
    2. Handwritten Letter Recognition (A-Z)
    3. Spam vs. Not Spam Classification
    4. Disease Prediction (Diabetes)
    5. Object Image Classification (CIFAR-10)
"""

# ==============================================================================
# IMPORTS
# ==============================================================================
import json
import os
import random
import sys
import time
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset

import torchvision
import torchvision.transforms as transforms

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# ==============================================================================
# GLOBAL CONFIG
# ==============================================================================
DEVICE = torch.device('cpu')
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Search space operations
OPS_IMAGE  = ['conv3x3', 'conv5x5', 'maxpool', 'avgpool', 'identity']
OPS_TABULAR = ['linear_big', 'linear_small', 'dropout', 'batchnorm', 'identity']
NUM_NODES  = 5
CHANNELS   = 16

# ==============================================================================
# TASK DEFINITIONS
# ==============================================================================

TASKS = {
    1: {
        'name': 'Handwritten Digit Recognition',
        'description': 'Recognize digits 0-9 from handwritten images (MNIST)',
        'type': 'image',
        'num_classes': 10,
        'input_channels': 1,
        'emoji': '✏️',
        'train_size': 6000,
        'val_size': 2000,
    },
    2: {
        'name': 'Handwritten Letter Recognition',
        'description': 'Recognize letters A-Z from handwritten images (EMNIST)',
        'type': 'image',
        'num_classes': 26,
        'input_channels': 1,
        'emoji': '🔤',
        'train_size': 6000,
        'val_size': 2000,
    },
    3: {
        'name': 'Spam Detection',
        'description': 'Classify emails/messages as Spam or Not Spam',
        'type': 'tabular',
        'num_classes': 2,
        'emoji': '📧',
        'train_size': 4000,
        'val_size': 1000,
    },
    4: {
        'name': 'Diabetes Risk Prediction',
        'description': 'Predict diabetes risk from patient health data',
        'type': 'tabular',
        'num_classes': 2,
        'emoji': '🏥',
        'train_size': 500,
        'val_size': 150,
    },
    5: {
        'name': 'Object Image Classification',
        'description': 'Classify objects: planes, cars, birds, cats, and more (CIFAR-10)',
        'type': 'image',
        'num_classes': 10,
        'input_channels': 3,
        'emoji': '🖼️',
        'train_size': 5000,
        'val_size': 1000,
    },
}

# ==============================================================================
# INTERACTIVE MENU
# ==============================================================================

def clear_line():
    print('\r' + ' ' * 80 + '\r', end='')

def print_banner():
    banner = """
╔══════════════════════════════════════════════════════════════╗
║          🧠  NeuroSearch — Neural Architecture Search         ║
║        Automatically designs the best AI for your task        ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def show_task_menu() -> int:
    """Show interactive task selection menu. Returns task number."""
    print("  ┌─────────────────────────────────────────────────────┐")
    print("  │                  CHOOSE YOUR TASK                    │")
    print("  └─────────────────────────────────────────────────────┘")
    print()
    for num, task in TASKS.items():
        print(f"   [{num}]  {task['emoji']}  {task['name']}")
        print(f"        {task['description']}")
        print()

    while True:
        try:
            choice = input("  Enter task number (1-5): ").strip()
            choice = int(choice)
            if choice in TASKS:
                return choice
            print("  ❌ Please enter a number between 1 and 5.")
        except (ValueError, KeyboardInterrupt):
            print("  ❌ Please enter a valid number.")


def show_strategy_menu() -> str:
    """Show search strategy selection. Returns strategy name."""
    print()
    print("  ┌─────────────────────────────────────────────────────┐")
    print("  │              CHOOSE SEARCH STRATEGY                  │")
    print("  └─────────────────────────────────────────────────────┘")
    print()
    strategies = {
        '1': ('random',      '🎲', 'Random Search',     'Tries random network designs. Simple but effective baseline.'),
        '2': ('evolutionary','🧬', 'Evolutionary',      'Mimics natural selection. Breeds best designs together.'),
        '3': ('bayesian',    '📊', 'Bayesian',          'Learns from results to make smarter guesses over time.'),
    }
    for key, (_, emoji, name, desc) in strategies.items():
        print(f"   [{key}]  {emoji}  {name}")
        print(f"        {desc}")
        print()

    while True:
        try:
            choice = input("  Enter strategy number (1-3): ").strip()
            if choice in strategies:
                return strategies[choice][0]
            print("  ❌ Please enter 1, 2, or 3.")
        except KeyboardInterrupt:
            sys.exit(0)


def confirm_and_run(task_num: int, strategy: str) -> bool:
    """Show summary and ask for confirmation."""
    task = TASKS[task_num]
    print()
    print("  ┌─────────────────────────────────────────────────────┐")
    print("  │                   READY TO RUN                       │")
    print("  └─────────────────────────────────────────────────────┘")
    print()
    print(f"   Task:      {task['emoji']}  {task['name']}")
    print(f"   Strategy:  {strategy.capitalize()}")
    print(f"   Est. time: 5–10 minutes")
    print()
    ans = input("  Start search? (y/n): ").strip().lower()
    return ans in ('y', 'yes', '')


# ==============================================================================
# DATA LOADERS FOR ALL 5 TASKS
# ==============================================================================

def load_task_data(task_num: int) -> Tuple[DataLoader, DataLoader, dict]:
    """Load data for the selected task. Returns train_loader, val_loader, task_info."""
    task = TASKS[task_num]
    print(f"\n  📥 Loading data for: {task['name']}...")

    if task_num == 1:
        return _load_mnist(task)

    elif task_num == 2:
        return _load_emnist(task)

    elif task_num == 3:
        return _load_spam(task)

    elif task_num == 4:
        return _load_diabetes(task)

    elif task_num == 5:
        return _load_cifar10(task)


def _load_mnist(task):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_ds = torchvision.datasets.MNIST('./data', train=True,  download=True, transform=transform)
    val_ds   = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(Subset(train_ds, range(task['train_size'])), batch_size=128, shuffle=True)
    val_loader   = DataLoader(Subset(val_ds,   range(task['val_size'])),   batch_size=128, shuffle=False)
    print(f"  ✓ MNIST loaded — {task['train_size']} train / {task['val_size']} val samples")
    return train_loader, val_loader, task


def _load_emnist(task):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    try:
        train_ds = torchvision.datasets.EMNIST('./data', split='letters', train=True,  download=True, transform=transform)
        val_ds   = torchvision.datasets.EMNIST('./data', split='letters', train=False, download=True, transform=transform)
        # EMNIST letters are 1-indexed, shift to 0-indexed
        train_ds.targets = train_ds.targets - 1
        val_ds.targets   = val_ds.targets   - 1
    except Exception:
        print("  ⚠️  EMNIST unavailable, falling back to MNIST digits...")
        return _load_mnist({**task, 'num_classes': 10})

    train_loader = DataLoader(Subset(train_ds, range(task['train_size'])), batch_size=128, shuffle=True)
    val_loader   = DataLoader(Subset(val_ds,   range(task['val_size'])),   batch_size=128, shuffle=False)
    print(f"  ✓ EMNIST Letters loaded — {task['train_size']} train / {task['val_size']} val samples")
    return train_loader, val_loader, task


def _load_spam(task):
    """
    Generate a realistic synthetic spam dataset using simple word-count features.
    Words associated with spam vs. ham give a learnable signal.
    """
    print("  ℹ️  Generating synthetic spam dataset (keyword frequency features)...")

    spam_words  = ['free', 'win', 'prize', 'click', 'offer', 'cash', 'urgent', 'deal', 'buy', 'discount']
    ham_words   = ['meeting', 'report', 'schedule', 'project', 'team', 'update', 'review', 'document', 'work', 'plan']
    all_words   = spam_words + ham_words
    n_features  = len(all_words)

    np.random.seed(42)
    n_total = task['train_size'] + task['val_size']
    labels = np.random.randint(0, 2, n_total)  # 0=ham, 1=spam

    features = np.zeros((n_total, n_features))
    for i, label in enumerate(labels):
        if label == 1:  # spam
            features[i, :10]  = np.random.poisson(3, 10)   # more spam words
            features[i, 10:]  = np.random.poisson(0.5, 10)
        else:           # ham
            features[i, :10]  = np.random.poisson(0.3, 10)
            features[i, 10:]  = np.random.poisson(2.5, 10) # more ham words
        # Add noise
        features[i] += np.random.normal(0, 0.5, n_features)
        features[i] = np.clip(features[i], 0, None)

    # Normalize
    features = (features - features.mean(0)) / (features.std(0) + 1e-8)

    X_train = torch.tensor(features[:task['train_size']], dtype=torch.float32)
    y_train = torch.tensor(labels[:task['train_size']], dtype=torch.long)
    X_val   = torch.tensor(features[task['train_size']:], dtype=torch.float32)
    y_val   = torch.tensor(labels[task['train_size']:], dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=128, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val,   y_val),   batch_size=128, shuffle=False)

    task = {**task, 'input_dim': n_features}
    print(f"  ✓ Spam dataset ready — {task['train_size']} train / {task['val_size']} val samples, {n_features} features")
    return train_loader, val_loader, task


def _load_diabetes(task):
    """
    Use the Pima Indians Diabetes dataset.
    Downloads from sklearn or generates synthetic version if unavailable.
    """
    print("  ℹ️  Loading diabetes dataset...")

    try:
        from sklearn.datasets import fetch_openml
        data = fetch_openml('diabetes', version=1, as_frame=True, parser='auto')
        X = data.data.values.astype(np.float32)
        y = (data.target.values == 'tested_positive').astype(np.int64)
    except Exception:
        print("  ⚠️  Could not fetch dataset online, generating synthetic version...")
        np.random.seed(42)
        n = task['train_size'] + task['val_size']
        # 8 health features: pregnancies, glucose, BP, skin, insulin, BMI, pedigree, age
        X = np.random.randn(n, 8).astype(np.float32)
        # Create a learnable rule: high glucose + high BMI → diabetic
        score = 0.6 * X[:, 1] + 0.4 * X[:, 5] + 0.2 * X[:, 7]
        y = (score > 0.3).astype(np.int64)

    # Normalize
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)

    n_train = min(task['train_size'], int(len(X) * 0.8))
    n_val   = min(task['val_size'],   len(X) - n_train)

    X_train = torch.tensor(X[:n_train],          dtype=torch.float32)
    y_train = torch.tensor(y[:n_train],          dtype=torch.long)
    X_val   = torch.tensor(X[n_train:n_train+n_val], dtype=torch.float32)
    y_val   = torch.tensor(y[n_train:n_train+n_val], dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val,   y_val),   batch_size=64, shuffle=False)

    task = {**task, 'input_dim': X.shape[1]}
    print(f"  ✓ Diabetes dataset ready — {len(X_train)} train / {len(X_val)} val samples, {X.shape[1]} features")
    return train_loader, val_loader, task


def _load_cifar10(task):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    train_ds = torchvision.datasets.CIFAR10('./data', train=True,  download=True, transform=transform)
    val_ds   = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(Subset(train_ds, range(task['train_size'])), batch_size=128, shuffle=True)
    val_loader   = DataLoader(Subset(val_ds,   range(task['val_size'])),   batch_size=128, shuffle=False)
    print(f"  ✓ CIFAR-10 loaded — {task['train_size']} train / {task['val_size']} val samples")
    return train_loader, val_loader, task


# ==============================================================================
# SEARCH SPACE & ARCHITECTURE ENCODING
# ==============================================================================

class Architecture:
    """Represents a neural architecture as a list of operation indices."""

    def __init__(self, ops: Optional[List[int]] = None, n_ops: int = 5):
        self.n_ops = n_ops
        if ops is None:
            ops = [random.randint(0, n_ops - 1) for _ in range(NUM_NODES)]
        self.ops = ops
        self.accuracy: Optional[float] = None
        self.params:   Optional[int]   = None
        self.arch_id:  Optional[str]   = None

    def encoding(self) -> np.ndarray:
        enc = np.zeros(NUM_NODES * self.n_ops)
        for i, op_idx in enumerate(self.ops):
            enc[i * self.n_ops + op_idx] = 1.0
        return enc

    def __hash__(self):
        return hash(tuple(self.ops))

    def __eq__(self, other):
        return self.ops == other.ops

    def mutate(self, prob: float = 0.3) -> 'Architecture':
        new_ops = self.ops[:]
        for i in range(len(new_ops)):
            if random.random() < prob:
                new_ops[i] = random.randint(0, self.n_ops - 1)
        return Architecture(new_ops, self.n_ops)

    @staticmethod
    def crossover(p1: 'Architecture', p2: 'Architecture') -> 'Architecture':
        point = random.randint(1, NUM_NODES - 1)
        return Architecture(p1.ops[:point] + p2.ops[point:], p1.n_ops)

    @staticmethod
    def random(n_ops: int = 5) -> 'Architecture':
        return Architecture(n_ops=n_ops)


# ==============================================================================
# NETWORK BUILDERS
# ==============================================================================

def get_image_op(op_name: str, in_ch: int, out_ch: int) -> nn.Module:
    ops_map = {
        'conv3x3':  nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)),
        'conv5x5':  nn.Sequential(nn.Conv2d(in_ch, out_ch, 5, padding=2, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)),
        'maxpool':  nn.Sequential(nn.MaxPool2d(3, stride=1, padding=1),
                                   nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity(),
                                   nn.BatchNorm2d(out_ch)),
        'avgpool':  nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1),
                                   nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity(),
                                   nn.BatchNorm2d(out_ch)),
        'identity': nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity(),
    }
    return ops_map.get(op_name, nn.Identity())


def get_tabular_op(op_name: str, dim: int) -> nn.Module:
    ops_map = {
        'linear_big':   nn.Sequential(nn.Linear(dim, dim), nn.ReLU()),
        'linear_small': nn.Sequential(nn.Linear(dim, max(dim//2, 8)), nn.ReLU(), nn.Linear(max(dim//2, 8), dim)),
        'dropout':      nn.Dropout(0.3),
        'batchnorm':    nn.BatchNorm1d(dim),
        'identity':     nn.Identity(),
    }
    return ops_map.get(op_name, nn.Identity())


class ImageNet(nn.Module):
    """CNN built from an Architecture spec."""
    def __init__(self, arch: Architecture, op_names: List[str], in_channels: int, num_classes: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, CHANNELS, 3, padding=1, bias=False),
            nn.BatchNorm2d(CHANNELS), nn.ReLU(inplace=True)
        )
        self.ops = nn.ModuleList([get_image_op(op_names[i], CHANNELS, CHANNELS) for i in arch.ops])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(CHANNELS, num_classes)

    def forward(self, x):
        x = self.stem(x)
        for op in self.ops:
            x = op(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.head(x)


class TabularNet(nn.Module):
    """MLP built from an Architecture spec for tabular data."""
    def __init__(self, arch: Architecture, op_names: List[str], input_dim: int, num_classes: int):
        super().__init__()
        hidden = 64
        self.input_proj = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU())
        self.ops = nn.ModuleList([get_tabular_op(op_names[i], hidden) for i in arch.ops])
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = self.input_proj(x)
        for op in self.ops:
            x = op(x)
        return self.head(x)


def build_model(arch: Architecture, task_info: dict) -> nn.Module:
    """Build the right model type for the task."""
    if task_info['type'] == 'image':
        op_names = OPS_IMAGE
        return ImageNet(arch, op_names, task_info['input_channels'], task_info['num_classes'])
    else:
        op_names = OPS_TABULAR
        return TabularNet(arch, op_names, task_info['input_dim'], task_info['num_classes'])


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def op_name_for_arch(arch: Architecture, task_info: dict) -> List[str]:
    ops = OPS_IMAGE if task_info['type'] == 'image' else OPS_TABULAR
    return [ops[i] for i in arch.ops]


# ==============================================================================
# TRAINING PIPELINE
# ==============================================================================

def train_evaluate(arch: Architecture, task_info: dict,
                   train_loader: DataLoader, val_loader: DataLoader,
                   epochs: int = 5) -> Tuple[float, int]:
    try:
        model   = build_model(arch, task_info).to(DEVICE)
        n_params = count_params(model)
        opt     = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched   = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        crit    = nn.CrossEntropyLoss()

        for _ in range(epochs):
            model.train()
            for X, y in train_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                opt.zero_grad()
                crit(model(X), y).backward()
                opt.step()
            sched.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                preds = model(X).argmax(1)
                correct += preds.eq(y).sum().item()
                total   += y.size(0)

        return 100.0 * correct / total, n_params
    except Exception as e:
        return 0.0, 0


# ==============================================================================
# SEARCH STRATEGIES
# ==============================================================================

class RandomSearch:
    def __init__(self, evaluated, n_ops):
        self.evaluated = evaluated
        self.n_ops = n_ops

    def next_candidates(self, n):
        out, attempts = [], 0
        while len(out) < n and attempts < n * 15:
            a = Architecture.random(self.n_ops)
            if hash(a) not in self.evaluated:
                out.append(a)
            attempts += 1
        return out


class EvolutionarySearch:
    def __init__(self, evaluated, n_ops, pop_size=15):
        self.evaluated = evaluated
        self.n_ops     = n_ops
        self.pop_size  = pop_size
        self.population = [Architecture.random(n_ops) for _ in range(pop_size)]

    def _select(self, k=3):
        pool = random.sample(self.population, min(k, len(self.population)))
        return max(pool, key=lambda a: self.evaluated.get(hash(a), (0,))[0])

    def next_candidates(self, n):
        evaled = [a for a in self.population if hash(a) in self.evaluated]
        if len(evaled) < 2:
            kids = [Architecture.random(self.n_ops) for _ in range(n)]
            self.population += kids
            return kids

        kids, attempts = [], 0
        while len(kids) < n and attempts < n * 20:
            if random.random() < 0.5:
                child = self._select().mutate()
            else:
                child = Architecture.crossover(self._select(), self._select())
            if hash(child) not in self.evaluated:
                kids.append(child)
            attempts += 1

        self.population += kids
        if len(self.population) > self.pop_size * 4:
            evaled_sorted = sorted(
                [a for a in self.population if hash(a) in self.evaluated],
                key=lambda a: self.evaluated[hash(a)][0], reverse=True
            )
            unevaled = [a for a in self.population if hash(a) not in self.evaluated]
            self.population = evaled_sorted[:self.pop_size] + unevaled[-self.pop_size:]

        return kids


class BayesianSearch:
    def __init__(self, evaluated, n_ops, n_init=5):
        self.evaluated  = evaluated
        self.n_ops      = n_ops
        self.n_init     = n_init
        self.encodings  = {}
        self.gp         = GaussianProcessRegressor(
            kernel=ConstantKernel(1.0) * Matern(nu=2.5),
            n_restarts_optimizer=2, normalize_y=True, alpha=1e-3
        )
        self.fitted = False

    def _register(self, arch):
        self.encodings[hash(arch)] = arch.encoding()

    def _fit(self):
        X, y = [], []
        for h, (acc, _) in self.evaluated.items():
            if h in self.encodings:
                X.append(self.encodings[h]); y.append(acc)
        if len(X) >= 2:
            self.gp.fit(np.array(X), np.array(y))
            self.fitted = True

    def _ei(self, X_cand, y_best, xi=0.01):
        from scipy.stats import norm
        mu, sigma = self.gp.predict(X_cand, return_std=True)
        imp = mu - y_best - xi
        Z   = imp / (sigma + 1e-9)
        ei  = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma < 1e-10] = 0
        return ei

    def next_candidates(self, n):
        if len(self.evaluated) < self.n_init or not self.fitted:
            self._fit()
            cands = [Architecture.random(self.n_ops) for _ in range(n)]
            for c in cands: self._register(c)
            return cands

        self._fit()
        y_best = max(v[0] for v in self.evaluated.values())
        pool   = [Architecture.random(self.n_ops) for _ in range(300)]
        X_pool = np.array([a.encoding() for a in pool])
        scores = self._ei(X_pool, y_best)

        selected = []
        for idx in np.argsort(scores)[::-1]:
            a = pool[idx]
            if hash(a) not in self.evaluated and len(selected) < n:
                self._register(a)
                selected.append(a)
        while len(selected) < n:
            a = Architecture.random(self.n_ops)
            self._register(a)
            selected.append(a)
        return selected


# ==============================================================================
# SURROGATE MODEL
# ==============================================================================

class SurrogateModel:
    def __init__(self, input_dim):
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32),        nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.opt     = optim.Adam(self.net.parameters(), lr=1e-3)
        self.X_data  = []
        self.y_data  = []
        self.trained = False
        self.error   = float('inf')

    def add(self, arch: Architecture, acc: float):
        self.X_data.append(arch.encoding())
        self.y_data.append(acc)

    def train(self, epochs=80):
        if len(self.X_data) < 3:
            return
        X = torch.tensor(np.array(self.X_data), dtype=torch.float32)
        y = torch.tensor(np.array(self.y_data), dtype=torch.float32).unsqueeze(1)
        y_mean, y_std = y.mean(), y.std() + 1e-8
        y_n = (y - y_mean) / y_std

        self.net.train()
        for _ in range(epochs):
            self.opt.zero_grad()
            nn.MSELoss()(self.net(X), y_n).backward()
            self.opt.step()

        self.net.eval()
        with torch.no_grad():
            preds = self.net(X) * y_std + y_mean
            self.error = (preds.squeeze() - y.squeeze()).abs().mean().item()
        self.trained = True


# ==============================================================================
# VISUALIZATION
# ==============================================================================

def print_header(title: str, width: int = 64):
    print("\n" + "═" * width)
    pad = (width - len(title)) // 2
    print(" " * pad + title)
    print("═" * width)


def print_arch_ascii(arch: Architecture, task_info: dict,
                     accuracy: float = None, params: int = None, rank: int = None):
    ops = op_name_for_arch(arch, task_info)
    input_label  = "Input (image)" if task_info['type'] == 'image' else f"Input ({task_info.get('input_dim','?')} features)"
    output_label = f"Output ({task_info['num_classes']} classes)"

    print()
    if rank:
        print(f"  🏆  Rank {rank} Architecture")
    if accuracy is not None:
        print(f"  Accuracy: {accuracy:.1f}%  |  Params: {params:,}")
    print()
    print(f"       {input_label}")
    print( "            │")
    print( "            ▼")
    for i, op in enumerate(ops):
        label = op.replace('_', ' ').title()
        border = "─" * (len(label) + 4)
        print(f"       ┌{border}┐")
        print(f"       │  {label}  │")
        print(f"       └{border}┘")
        print( "            │")
        print( "            ▼")
    print(f"       {output_label}")
    print()


def print_leaderboard(leaderboard: list, task_info: dict, top_k: int = 5):
    print_header("🏆  FINAL LEADERBOARD")
    task_type = task_info['type']
    for rank, (arch, acc, params) in enumerate(leaderboard[:top_k], 1):
        ops = " → ".join(op_name_for_arch(arch, task_info))
        medal = ["🥇", "🥈", "🥉", "  4.", "  5."][rank - 1]
        print(f"\n  {medal}  Rank {rank}  |  Acc: {acc:.1f}%  |  Params: {params:,}")
        print(f"       Input → {ops} → Output")


def plot_results(history: list, task_info: dict, strategy: str, save_path: str):
    try:
        accs     = [h['accuracy'] for h in history]
        best_acc = []
        best     = 0
        for a in accs:
            best = max(best, a)
            best_acc.append(best)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(
            f"NeuroSearch — {task_info['name']}  ({strategy.capitalize()} Strategy)",
            fontsize=13, fontweight='bold'
        )

        # 1: trajectory
        x = range(1, len(accs) + 1)
        axes[0].scatter(x, accs, alpha=0.5, s=20, color='steelblue', label='All architectures')
        axes[0].plot(x, best_acc, color='crimson', linewidth=2, label='Best so far')
        axes[0].set_xlabel('Architecture #'); axes[0].set_ylabel('Accuracy (%)')
        axes[0].set_title('Search Trajectory'); axes[0].legend(); axes[0].grid(alpha=0.3)

        # 2: distribution
        axes[1].hist(accs, bins=12, color='steelblue', edgecolor='white', alpha=0.8)
        axes[1].axvline(max(accs), color='crimson', linestyle='--', label=f'Best: {max(accs):.1f}%')
        axes[1].set_xlabel('Accuracy (%)'); axes[1].set_ylabel('Count')
        axes[1].set_title('Accuracy Distribution'); axes[1].legend(); axes[1].grid(alpha=0.3)

        # 3: top 5 bar chart
        top5   = sorted(history, key=lambda h: h['accuracy'], reverse=True)[:5]
        labels = [h['arch_id'] for h in top5]
        vals   = [h['accuracy'] for h in top5]
        colors = ['gold', 'silver', '#cd7f32', 'steelblue', 'steelblue']
        axes[2].barh(labels[::-1], vals[::-1], color=colors[::-1])
        axes[2].set_xlabel('Accuracy (%)')
        axes[2].set_title('Top 5 Architectures')
        axes[2].grid(alpha=0.3, axis='x')
        for i, v in enumerate(vals[::-1]):
            axes[2].text(v + 0.1, i, f'{v:.1f}%', va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"\n  ✓ Plot saved → {save_path}")
    except Exception as e:
        print(f"\n  ⚠️  Could not save plot: {e}")


# ==============================================================================
# PROGRESS BAR
# ==============================================================================

def pbar(current, total, prefix='', width=32):
    filled = int(width * current / total)
    bar    = '█' * filled + '░' * (width - filled)
    print(f'\r  {prefix} [{bar}] {current}/{total}', end='', flush=True)
    if current == total:
        print()


# ==============================================================================
# MAIN SEARCH ENGINE
# ==============================================================================

class NeuroSearch:
    def __init__(self, task_num: int, strategy_name: str,
                 train_loader, val_loader, task_info: dict,
                 generations: int = 8, pop_size: int = 10, epochs_per_arch: int = 5):

        self.task_num      = task_num
        self.strategy_name = strategy_name
        self.train_loader  = train_loader
        self.val_loader    = val_loader
        self.task_info     = task_info
        self.generations   = generations
        self.pop_size      = pop_size
        self.epochs        = epochs_per_arch

        self.evaluated:  Dict = {}   # hash -> (acc, params)
        self.arch_store: Dict = {}   # hash -> Architecture
        self.history:    List = []

        self.best_arch     = None
        self.best_accuracy = 0.0

        n_ops = len(OPS_IMAGE) if task_info['type'] == 'image' else len(OPS_TABULAR)
        self.n_ops = n_ops

        enc_dim = NUM_NODES * n_ops
        self.surrogate = SurrogateModel(enc_dim)

        if strategy_name == 'random':
            self.strategy = RandomSearch(self.evaluated, n_ops)
        elif strategy_name == 'evolutionary':
            self.strategy = EvolutionarySearch(self.evaluated, n_ops, pop_size)
        elif strategy_name == 'bayesian':
            self.strategy = BayesianSearch(self.evaluated, n_ops)
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

    def _eval(self, arch: Architecture, arch_id: str):
        acc, params = train_evaluate(arch, self.task_info, self.train_loader,
                                     self.val_loader, self.epochs)
        arch.accuracy = acc
        arch.params   = params
        arch.arch_id  = arch_id

        h = hash(arch)
        self.evaluated[h]  = (acc, params)
        self.arch_store[h] = arch
        self.history.append({'arch_id': arch_id, 'accuracy': acc, 'params': params, 'ops': arch.ops[:]})
        self.surrogate.add(arch, acc)

        if acc > self.best_accuracy:
            self.best_accuracy = acc
            self.best_arch     = arch

        return acc, params

    def run(self):
        task  = self.task_info
        start = time.time()
        counter = 0

        print_header(f"{task['emoji']}  {task['name'].upper()}")
        print(f"  Strategy:    {self.strategy_name.capitalize()}")
        print(f"  Generations: {self.generations}  |  Epochs/arch: {self.epochs}")
        print()

        for gen in range(1, self.generations + 1):
            gen_start = time.time()

            # Get candidates
            if self.strategy_name == 'evolutionary' and gen == 1:
                candidates = [a for a in self.strategy.population if hash(a) not in self.evaluated]
            else:
                n = self.pop_size if self.strategy_name == 'evolutionary' else 5
                candidates = self.strategy.next_candidates(n)

            candidates = [c for c in candidates if hash(c) not in self.evaluated]
            if not candidates:
                candidates = [Architecture.random(self.n_ops) for _ in range(3)]

            elapsed = int(time.time() - start)
            mm, ss  = elapsed // 60, elapsed % 60
            print(f"  [{mm:02d}:{ss:02d}]  Generation {gen}/{self.generations} — {len(candidates)} architectures")

            gen_best = 0
            gen_best_id = ''
            for i, arch in enumerate(candidates):
                counter += 1
                arch_id = f"arch_{counter:03d}"
                pbar(i, len(candidates), prefix='  Evaluating ')
                acc, _ = self._eval(arch, arch_id)
                if acc > gen_best:
                    gen_best    = acc
                    gen_best_id = arch_id
            pbar(len(candidates), len(candidates), prefix='  Evaluating ')

            print(f"  Gen best: {gen_best:.1f}% ({gen_best_id})  |  Overall best: {self.best_accuracy:.1f}%  "
                  f"|  {time.time()-gen_start:.1f}s")

            if len(self.evaluated) >= 3:
                self.surrogate.train(epochs=60)

        total_time = time.time() - start
        self._report(total_time)
        return self.best_arch, self.best_accuracy

    def _report(self, total_time: float):
        task = self.task_info

        # Leaderboard
        leaderboard = sorted(
            [(self.arch_store[h], acc, params) for h, (acc, params) in self.evaluated.items()],
            key=lambda x: x[1], reverse=True
        )
        print_leaderboard(leaderboard, task)

        # Best architecture visualization
        print_header("🔬  BEST ARCHITECTURE")
        if self.best_arch:
            acc, params = self.evaluated[hash(self.best_arch)]
            print_arch_ascii(self.best_arch, task, accuracy=acc, params=params, rank=1)

        # Stats
        print_header("📊  SEARCH STATS")
        h = int(total_time // 3600)
        m = int((total_time % 3600) // 60)
        s = int(total_time % 60)
        print(f"  Total architectures evaluated : {len(self.evaluated)}")
        print(f"  Total time                    : {h}h {m}m {s}s")
        print(f"  Best accuracy                 : {self.best_accuracy:.2f}%")
        if self.surrogate.trained:
            print(f"  Surrogate model error         : ±{self.surrogate.error:.1f}%")

        # Save plot
        slug = task['name'].lower().replace(' ', '_')
        plot_path = f"neurosearch_{slug}_{self.strategy_name}.png"
        plot_results(self.history, task, self.strategy_name, plot_path)

        # Save best arch JSON
        if self.best_arch:
            out = {
                'task':        task['name'],
                'strategy':    self.strategy_name,
                'ops':         op_name_for_arch(self.best_arch, task),
                'accuracy':    round(self.best_accuracy, 2),
                'params':      int(self.best_arch.params or 0),
                'total_evals': len(self.evaluated),
            }
            fname = f"best_arch_{slug}.json"
            with open(fname, 'w') as f:
                json.dump(out, f, indent=2)
            print(f"  Best architecture saved       → {fname}")

        print("═" * 64)


# ==============================================================================
# MAIN — INTERACTIVE ENTRY POINT
# ==============================================================================

def main():
    print_banner()

    # Step 1: Pick task
    task_num  = show_task_menu()
    task_info_base = TASKS[task_num]

    # Step 2: Pick strategy
    strategy = show_strategy_menu()

    # Step 3: Confirm
    if not confirm_and_run(task_num, strategy):
        print("\n  Cancelled. Run again whenever you're ready!\n")
        return

    # Step 4: Load data
    train_loader, val_loader, task_info = load_task_data(task_num)

    # Step 5: Run search
    # Balanced settings: 8 generations, 8 archs/gen → ~40-60 total evaluations
    engine = NeuroSearch(
        task_num       = task_num,
        strategy_name  = strategy,
        train_loader   = train_loader,
        val_loader     = val_loader,
        task_info      = task_info,
        generations    = 8,
        pop_size       = 8,
        epochs_per_arch= 5,
    )

    print()
    best_arch, best_acc = engine.run()

    print(f"\n  ✅  Done! Best accuracy achieved: {best_acc:.1f}%")
    print(f"  Want to try another task? Just run the script again!\n")


if __name__ == '__main__':
    main()
