import os

import matplotlib.pyplot as plt
import numpy as np


NUM_CLASSES = 5


def load_emg_data(path):
    """Carrega o EMG e retorna X, Y e os rótulos originais das classes."""
    try:
        raw = np.loadtxt(path, delimiter=",")
    except ValueError:
        raw = np.loadtxt(path, delimiter=None)

    # O arquivo pode estar em 3 x N ou em N x 3. Normaliza ambos os formatos.
    if raw.ndim != 2:
        raise ValueError("O CSV deve conter uma matriz bidimensional.")
    if raw.shape[0] == 3 and raw.shape[1] != 3:
        samples = raw.T
    elif raw.shape[1] == 3:
        samples = raw
    else:
        raise ValueError("Esperado um CSV com dois sensores e uma coluna de classes.")

    X = samples[:, :2].astype(float)
    labels = samples[:, 2].astype(int)
    classes = np.arange(1, NUM_CLASSES + 1)

    if not np.all(np.isin(labels, classes)):
        raise ValueError("Os rótulos devem estar entre 1 e 5.")

    # One-hot encoding manual: a coluna j representa a classe j + 1.
    Y = np.zeros((labels.size, NUM_CLASSES), dtype=float)
    for row, label in enumerate(labels):
        Y[row, label - 1] = 1.0

    return X, Y, labels, classes


class OLSClassifier:
    """Classificador MQO tradicional para respostas one-hot."""

    def __init__(self):
        self.W = None
        self.classes_ = None

    def fit(self, X, Y, classes=None):
        X_bias = np.column_stack((np.ones(X.shape[0]), X))
        self.W = np.linalg.pinv(X_bias) @ Y
        self.classes_ = np.arange(1, Y.shape[1] + 1) if classes is None else np.asarray(classes)
        return self

    def predict_scores(self, X):
        if self.W is None:
            raise RuntimeError("O classificador precisa ser treinado com fit antes de predict.")
        X_bias = np.column_stack((np.ones(X.shape[0]), X))
        return X_bias @ self.W

    def predict(self, X):
        scores = self.predict_scores(X)
        return self.classes_[np.argmax(scores, axis=1)]


class RegularizedOLSClassifier(OLSClassifier):
    """Classificador MQO regularizado por Tikhonov/Ridge."""

    def __init__(self, lambd=0.0):
        super().__init__()
        if lambd < 0:
            raise ValueError("lambda deve ser não negativo.")
        self.lambd = float(lambd)

    def fit(self, X, Y, classes=None):
        X_bias = np.column_stack((np.ones(X.shape[0]), X))
        penalty = np.eye(X_bias.shape[1])
        penalty[0, 0] = 0.0  # O intercepto não é regularizado.
        system = X_bias.T @ X_bias + self.lambd * penalty
        self.W = np.linalg.pinv(system) @ X_bias.T @ Y
        self.classes_ = np.arange(1, Y.shape[1] + 1) if classes is None else np.asarray(classes)
        return self


class PolynomialOLSClassifier(OLSClassifier):
    """Classificador MQO com termos polinomiais de grau até q."""

    def __init__(self, q=1):
        super().__init__()
        if int(q) != q or q < 1:
            raise ValueError("q deve ser um inteiro maior ou igual a 1.")
        self.q = int(q)

    def _transform(self, X):
        if X.shape[1] != 2:
            raise ValueError("O classificador polinomial espera exatamente dois sensores.")

        features = [np.ones(X.shape[0])]
        for degree in range(1, self.q + 1):
            for degree_sensor_1 in range(degree, -1, -1):
                degree_sensor_2 = degree - degree_sensor_1
                features.append(
                    X[:, 0] ** degree_sensor_1 * X[:, 1] ** degree_sensor_2
                )
        return np.column_stack(features)

    def fit(self, X, Y, classes=None):
        polynomial_X = self._transform(X)
        self.W = np.linalg.pinv(polynomial_X) @ Y
        self.classes_ = np.arange(1, Y.shape[1] + 1) if classes is None else np.asarray(classes)
        return self

    def predict_scores(self, X):
        if self.W is None:
            raise RuntimeError("O classificador precisa ser treinado com fit antes de predict.")
        return self._transform(X) @ self.W


def plot_data(X, labels, classes):
    colors = ("tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple")
    plt.figure(figsize=(9, 6))
    for class_label, color in zip(classes, colors):
        mask = labels == class_label
        plt.scatter(
            X[mask, 0],
            X[mask, 1],
            s=10,
            alpha=0.55,
            color=color,
            label=f"Classe {class_label}",
        )
    plt.xlabel("Sensor 1 (Corrugador do Supercílio)")
    plt.ylabel("Sensor 2 (Zigomático Maior)")
    plt.title("Sinais EMG por classe")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()


if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), "dados", "EMG1.csv")
    X, Y, labels, classes = load_emg_data(data_path)

    print(f"X: {X.shape}")
    print(f"Y: {Y.shape}")

    models = (
        OLSClassifier(),
        RegularizedOLSClassifier(lambd=1.0),
        PolynomialOLSClassifier(q=2),
    )
    for model in models:
        model.fit(X, Y, classes)
        predictions = model.predict(X)
        print(f"{type(model).__name__}: {predictions.shape}")

    plot_data(X, labels, classes)
    plt.show()