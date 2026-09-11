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


import time


def select_polynomial_degree(X, Y, labels, classes, q_values=range(1, 7), test_size=0.2, random_state=None, tol=0.002):
    """Avalia o classificador MQO polinomial para diferentes valores de q.

    Mede, num único particionamento holdout, a acurácia e o tempo de estimação
    dos parâmetros (fit) de cada q. Aplica uma estratégia de poda: escolhe o
    MENOR q cuja acurácia fica a até `tol` da melhor acurácia observada, em vez
    de pegar sempre o q com maior acurácia bruta — assim não se paga tempo
    extra de treino por um ganho de acurácia irrelevante.

    Retorna (q_escolhido, tabela), onde tabela é uma lista de dicts
    {"q", "acuracia", "tempo_segundos"}.
    """
    rng = np.random.default_rng(random_state)
    n_samples = X.shape[0]
    n_train = round((1 - test_size) * n_samples)
    idx = rng.permutation(n_samples)
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    X_train, X_test = X[train_idx], X[test_idx]
    labels_train, labels_test = labels[train_idx], labels[test_idx]
    Y_train = Y[train_idx]

    table = []
    for q in q_values:
        model = PolynomialOLSClassifier(q=q)

        start = time.perf_counter()
        model.fit(X_train, Y_train, classes)
        elapsed = time.perf_counter() - start

        predictions = model.predict(X_test)
        accuracy = np.mean(predictions == labels_test)
        table.append({"q": q, "acuracia": accuracy, "tempo_segundos": elapsed})

    best_accuracy = max(row["acuracia"] for row in table)
    chosen = next(row for row in table if row["acuracia"] >= best_accuracy - tol)

    return chosen["q"], table


def print_degree_table(table):
    print(f"{'q':>3} | {'acurácia':>9} | {'tempo (s)':>10}")
    for row in table:
        print(f"{row['q']:>3} | {row['acuracia']:>9.4f} | {row['tempo_segundos']:>10.4f}")


def monte_carlo_validation(X, labels, classes, model_builders, R=500, train_frac=0.8, random_state=None):
    """Valida modelos por amostragem aleatória (Monte Carlo).

    model_builders: dict {nome: função sem args que retorna um modelo novo (ex.: fit()-ável)}.
    Retorna dict {nome: lista de acurácias, uma por rodada}.
    """
    rng = np.random.default_rng(random_state)
    n_samples = X.shape[0]
    n_train = round(train_frac * n_samples)
    num_classes = classes.size

    results = {name: [] for name in model_builders}

    for _ in range(R):
        idx = rng.permutation(n_samples)
        train_idx, test_idx = idx[:n_train], idx[n_train:]

        X_train, X_test = X[train_idx], X[test_idx]
        labels_train, labels_test = labels[train_idx], labels[test_idx]

        Y_train = np.zeros((labels_train.size, num_classes), dtype=float)
        for row, label in enumerate(labels_train):
            Y_train[row, label - 1] = 1.0

        for name, build_model in model_builders.items():
            model = build_model()
            model.fit(X_train, Y_train, classes)
            predictions = model.predict(X_test)
            accuracy = np.mean(predictions == labels_test)
            results[name].append(accuracy)

    return results


def summarize_accuracies(results):
    """Imprime média, desvio padrão, maior e menor valor da acurácia de cada modelo."""
    for name, accuracies in results.items():
        accuracies = np.asarray(accuracies)
        print(
            f"{name}: acurácia_média={accuracies.mean():.4f}  "
            f"desvio_padrao={accuracies.std():.4f}  "
            f"maior={accuracies.max():.4f}  "
            f"menor={accuracies.min():.4f}"
        )


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

    #4: seleção do grau q do classificador polinomial (acurácia vs. tempo).
    best_q, degree_table = select_polynomial_degree(
        X, Y, labels, classes, q_values=range(1, 7), test_size=0.2, random_state=42
    )
    print("\nSeleção do hiperparâmetro q (item 4):")
    print_degree_table(degree_table)
    print(f"q escolhido: {best_q}\n")

    #5: validação por Monte Carlo (R=500, 80% treino / 20% teste).
    model_builders = {
        "MQO tradicional": lambda: OLSClassifier(),
        "MQO regularizado (lambda=1.0)": lambda: RegularizedOLSClassifier(lambd=1.0),
        f"MQO polinomial (q={best_q})": lambda: PolynomialOLSClassifier(q=best_q),
    }
    results = monte_carlo_validation(
        X, labels, classes, model_builders, R=500, train_frac=0.8, random_state=42
    )
    summarize_accuracies(results)

    plot_data(X, labels, classes)
    plt.show()