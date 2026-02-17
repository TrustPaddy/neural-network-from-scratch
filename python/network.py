import numpy as np
from activation import Activation


def forward(q: np.ndarray, w: np.ndarray, arch: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """
    Forward pass through the network.

    Parameters
    ----------
    q : (N, arch[0]) input data
    w : flat weight vector (all weights + biases concatenated)
    arch : list of layer sizes, e.g. [2, 2, 4, 2]

    Returns
    -------
    x1, x2 : output columns
    """
    num_layers = len(arch) - 1
    x = q
    idx = 0

    for l in range(num_layers):
        n_in = arch[l]
        n_out = arch[l + 1]

        w_size = n_in * n_out
        W = w[idx: idx + w_size].reshape(n_in, n_out)
        idx += w_size

        b = w[idx: idx + n_out]
        idx += n_out

        z = x @ W + b  # broadcasting handles (N, n_out) + (n_out,)

        if l == 0:
            x = Activation.para_relu(z)
        elif l < num_layers - 1:
            x = Activation.tanh(z)
        else:
            x = z  # linear output layer

    return x[:, 0], x[:, 1]


def loss_mse(x1_label: np.ndarray, x2_label: np.ndarray,
             x1: np.ndarray, x2: np.ndarray) -> float:
    """Mean squared error over both outputs."""
    return float(np.mean((x1_label - x1) ** 2 + (x2_label - x2) ** 2))


def loss_from_forward(q: np.ndarray, x_labels: np.ndarray,
                      w: np.ndarray, arch: list[int]) -> float:
    """Run forward pass and return loss."""
    x1, x2 = forward(q, w, arch)
    return loss_mse(x_labels[:, 0], x_labels[:, 1], x1, x2)


def calc_gradient(q: np.ndarray, x_labels: np.ndarray,
                  w: np.ndarray, arch: list[int],
                  epsilon: float = 1e-6) -> tuple[np.ndarray, float]:
    """
    Numerical gradient via central differences.

    Returns
    -------
    gradient : same shape as w
    loss_val : loss at current weights
    """
    grad = np.zeros_like(w)

    for i in range(len(w)):
        w_right = w.copy()
        w_right[i] += epsilon
        loss_right = loss_from_forward(q, x_labels, w_right, arch)

        w_left = w.copy()
        w_left[i] -= epsilon
        loss_left = loss_from_forward(q, x_labels, w_left, arch)

        grad[i] = (loss_right - loss_left) / (2.0 * epsilon)

    # loss at current weights (use left-perturbed of last param as in original)
    current_loss = loss_left
    return grad, current_loss


def calc_weight_size(arch: list[int]) -> int:
    """Total number of trainable parameters (weights + biases)."""
    size = 0
    for i in range(len(arch) - 1):
        size += arch[i] * arch[i + 1]  # weights
        size += arch[i + 1]            # biases
    return size
