"""
Lösung 1: Lineares Zielproblem
    x1 = q1 + q2 - 1
    x2 = q1 - q2 + 1

Architektur: [2, 2, 2, 2]  (alle hidden: paraReLU, output: linear)
Optimierer:  AdamW
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from network import forward, calc_gradient, calc_weight_size

# ── Hyperparameter ──────────────────────────────────────────────
NUMBER_OF_EPOCHS = 10**5
ARCH = [2, 2, 2, 2]

# AdamW
ALPHA = 0.001
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-8
WEIGHT_DECAY = 0.0

# ── Trainingsdaten ──────────────────────────────────────────────
N = 100
q1 = np.random.uniform(-1, 1, (N, 1))
q2 = np.random.uniform(-1, 1, (N, 1))
q = np.hstack([q1, q2])

x1_label = q1 + q2 - 1
x2_label = q1 - q2 + 1
x_labels = np.hstack([x1_label, x2_label])

# ── Gewichte initialisieren ─────────────────────────────────────
w_size = calc_weight_size(ARCH)
w = np.random.uniform(-1, 1, w_size)

# AdamW-Zustand
m = np.zeros_like(w)
v = np.zeros_like(w)

# ── Training ────────────────────────────────────────────────────
losses = np.zeros(NUMBER_OF_EPOCHS)

start = time.perf_counter()

for i in range(1, NUMBER_OF_EPOCHS + 1):
    grad, loss_val = calc_gradient(q, x_labels, w, ARCH)

    # AdamW update
    m = BETA1 * m + (1 - BETA1) * grad
    v = BETA2 * v + (1 - BETA2) * grad**2

    m_hat = m / (1 - BETA1**i)
    v_hat = v / (1 - BETA2**i)

    r_k = ALPHA / (np.sqrt(v_hat) + EPSILON)
    w = w - r_k * m_hat - ALPHA * WEIGHT_DECAY * w

    losses[i - 1] = loss_val

    # Log im gleichen Rhythmus wie MATLAB-Original
    log_step = 10 ** int(np.floor(np.log10(max(i, 1))))
    if i % log_step == 0:
        print(f"step: {i:>7d} ===== loss: {loss_val:.10f}")

    if loss_val < 1e-11:
        losses = losses[:i]
        break

elapsed = time.perf_counter() - start
minutes = int(elapsed // 60)
seconds = int(elapsed % 60)
millis = int((elapsed - int(elapsed)) * 1000)

# ── Plot ────────────────────────────────────────────────────────
plt.figure()
plt.plot(range(1, len(losses) + 1), losses)
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Loss value")
plt.grid(True)
plt.title("Lösung 1 – Training Loss")
plt.tight_layout()
plt.savefig("loss_loesung1.png", dpi=150)
plt.show()

# ── Schnelltest ─────────────────────────────────────────────────
q_test = np.array([[0.75, -0.25]])
# Erwartung: x1 = 0.75 + (-0.25) - 1 = -0.5,  x2 = 0.75 - (-0.25) + 1 = 2.0
x1_test, x2_test = forward(q_test, w, ARCH)
print(f"\nQuick Verification: x1 = {x1_test[0]:.10f}  ===== x2 = {x2_test[0]:.10f}")
print(f"Elapsed Time: {minutes} min, {seconds} s, {millis} ms")
