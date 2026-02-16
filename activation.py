import numpy as np


class Activation:
    """Static activation functions for neural network layers."""

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    @staticmethod
    def para_relu(x: np.ndarray, alpha_pos: float = 1.0, alpha_neg: float = 0.1) -> np.ndarray:
        y = np.zeros_like(x)
        y[x > 0] = alpha_pos * x[x > 0]
        y[x <= 0] = alpha_neg * x[x <= 0]
        return y

    @staticmethod
    def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        return np.maximum(alpha * x, x)

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    @staticmethod
    def elu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        y = x.copy()
        mask = x < 0
        y[mask] = alpha * (np.exp(x[mask]) - 1)
        return y

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        x_shifted = x - np.max(x, axis=1, keepdims=True)
        ex = np.exp(x_shifted)
        return ex / np.sum(ex, axis=1, keepdims=True)
