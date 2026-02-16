# 🧠 Neural Network from Scratch

> A minimal neural network implementation in pure Python/NumPy — no PyTorch, no TensorFlow, just math.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-only_dependency-013243?logo=numpy&logoColor=white)](https://numpy.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What is this?

This project implements a **fully connected feedforward neural network** entirely from scratch to learn mathematical functions from input-output pairs. No autograd — gradients are computed via **central finite differences**, and weights are updated with **AdamW**.

Originally written in MATLAB, then ported to Python for clarity and speed.

### Two problems are solved:

| | Problem 1 (linear) | Problem 2 (nonlinear) |
|---|---|---|
| **Target** | $x_1 = q_1 + q_2 - 1$ | $x_1 = q_1 + q_2^2 - 1$ |
| | $x_2 = q_1 - q_2 + 1$ | $x_2 = q_1 - q_2 + 1$ |
| **Architecture** | `[2 → 2 → 2 → 2]` | `[2 → 2 → 4 → 2]` |
| **Training samples** | 100 | 70 |
| **Convergence target** | Loss < 10⁻¹¹ | Loss < 10⁻⁶ |

---

## Project Structure

```
nn-from-scratch/
├── activation.py      # Activation functions (ReLU, paraReLU, Sigmoid, Tanh, ELU, Softmax)
├── network.py         # Forward pass, MSE loss, numerical gradient, utilities
├── loesung1.py        # Training script — linear target problem
├── loesung2.py        # Training script — nonlinear target problem
├── requirements.txt   # Dependencies (numpy, matplotlib)
├── .vscode/           # VS Code debug launch configs
│   ├── launch.json
│   └── settings.json
└── README.md
```

---

## Quick Start

```bash
# Clone the repo
git clone https://github.com/<your-username>/nn-from-scratch.git
cd nn-from-scratch

# Install dependencies
pip install -r requirements.txt

# Run
python loesung1.py   # linear problem
python loesung2.py   # nonlinear problem
```

---

## How It Works

### Forward Pass

Each layer computes $z = xW + b$, then applies an activation:

| Layer | Activation |
|---|---|
| Hidden 1 | Parametric ReLU ($\alpha^+ = 1.0,\ \alpha^- = 0.1$) |
| Hidden 2+ | Tanh |
| Output | Linear (identity) |

### Gradient Computation

Since there's no autograd, gradients are approximated numerically using **central differences**:

$$\frac{\partial \mathcal{L}}{\partial w_i} \approx \frac{\mathcal{L}(w_i + \varepsilon) - \mathcal{L}(w_i - \varepsilon)}{2\varepsilon}$$

This is simple and reliable, but scales as $O(n)$ forward passes per step (where $n$ = number of parameters). Fine for small networks — intentionally kept this way for educational clarity.

### AdamW Optimizer

The weight update follows [AdamW (Loshchilov & Hutter, 2019)](https://arxiv.org/abs/1711.05101):

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

$$w_t = w_{t-1} - \frac{\alpha}{\sqrt{\hat{v}_t} + \varepsilon} \hat{m}_t - \alpha \lambda w_{t-1}$$

with bias correction $\hat{m}_t = m_t / (1 - \beta_1^t)$ and $\hat{v}_t = v_t / (1 - \beta_2^t)$.

| Parameter | Value |
|---|---|
| Learning rate $\alpha$ | 0.001 |
| $\beta_1$ | 0.9 |
| $\beta_2$ | 0.999 |
| $\varepsilon$ | 10⁻⁸ |
| Weight decay $\lambda$ | 0.0 |

---

## Available Activation Functions

All implemented in `activation.py` and usable via `Activation.<name>(x)`:

| Function | Formula |
|---|---|
| ReLU | $\max(0, x)$ |
| Parametric ReLU | $\alpha^+ x$ if $x > 0$, else $\alpha^- x$ |
| Leaky ReLU | $\max(\alpha x, x)$ |
| Sigmoid | $\frac{1}{1 + e^{-x}}$ |
| Tanh | $\tanh(x)$ |
| ELU | $x$ if $x \geq 0$, else $\alpha(e^x - 1)$ |
| Softmax | $\frac{e^{x_i}}{\sum_j e^{x_j}}$ |

---

## Customization

Want to experiment? The key knobs are at the top of each training script:

```python
ARCH = [2, 2, 4, 2]    # change network topology
ALPHA = 0.001           # learning rate
WEIGHT_DECAY = 0.0      # L2 regularization strength
NUMBER_OF_EPOCHS = 10**5
```

You can also swap activations in `network.py` → `forward()`.

---

## VS Code Integration

The workspace includes ready-to-use debug configurations. Open the folder in VS Code and press **F5** to launch either solution with the integrated debugger.

---

## License

MIT — do whatever you want with it.