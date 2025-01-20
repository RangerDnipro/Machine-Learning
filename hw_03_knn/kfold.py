"""
Модуль реалізує алгоритм K найближчих сусідів (K-Nearest Neighbors, KNN), щоб класифікувати дані.
Алгоритм використовує відстань між зразками для визначення їх класу,
а також дозволяє оцінити точність моделі за допомогою перехресної валідації (k-fold cross-validation).
"""

from typing import List, Tuple

import numpy as np
import pandas as pd


def euclidean_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    """
    Обчислює Евклідову відстань між двома точками.

    :param point1: Перша точка у вигляді масиву NumPy.
    :param point2: Друга точка у вигляді масиву NumPy.
    :return: Евклідова відстань між двома точками.
    """
    return np.sqrt(np.sum(np.power(point1 - point2, 2)))


class KNN:
    """
    Клас для реалізації класифікатора K найближчих сусідів (KNN).
    """

    def __init__(self, k: int) -> None:
        """
        Ініціалізує KNN з кількістю сусідів для врахування.

        :param k: Кількість найближчих сусідів.
        """
        self._X_train = None
        self._y_train = None
        self.k = k

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Навчання моделі KNN на основі тренувальних даних.

        :param X_train: Ознаки тренувального набору даних.
        :param y_train: Цільові значення тренувального набору даних.
        """
        self._X_train = X_train
        self._y_train = y_train

    def predict(self, X_test: np.ndarray, verbose: bool = False) -> np.ndarray:
        """
        Прогнозування цільових значень для тестових даних.

        :param X_test: Ознаки тестового набору даних.
        :param verbose: Друкувати прогрес під час прогнозування. За замовчуванням False.
        :return: Прогнозовані цільові значення.
        """
        n = X_test.shape[0]
        y_pred = np.empty(n, dtype=self._y_train.dtype)

        for i in range(n):
            distances = np.array([euclidean_distance(x, X_test[i]) for x in self._X_train])
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self._y_train[k_indices]
            y_pred[i] = np.bincount(k_nearest_labels).argmax()

            if verbose:
                print(f"Прогнозовано {i + 1}/{n} зразків", end="\r")

        if verbose:
            print("")
        return y_pred


def kfold_cross_validation(X: np.ndarray, y: np.ndarray, k: int) -> List[
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Розбиває набір даних на k фолдів для перехресної валідації.

    :param X: Ознаки набору даних.
    :param y: Цільові значення набору даних.
    :param k: Кількість фолдів.
    :return: Список кортежів (X_train, y_train, X_test, y_test).
    """
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    fold_size = n_samples // k
    folds = []

    for i in range(k):
        test_mask = np.zeros(n_samples, dtype=bool)
        test_mask[i * fold_size:(i + 1) * fold_size] = True

        X_train, X_test = X[~test_mask], X[test_mask]
        y_train, y_test = y[~test_mask], y[test_mask]
        folds.append((X_train, y_train, X_test, y_test))

    return folds


def evaluate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Обчислює точність прогнозування.

    :param y_true: Справжні цільові значення.
    :param y_pred: Прогнозовані цільові значення.
    :return: Значення точності.
    """
    return np.sum(y_true == y_pred) / len(y_true)


def main() -> None:
    """
    Основна функція для демонстрації роботи класифікатора KNN та перехресної валідації.
    """
    # Читання тренувальних і тестових даних з CSV файлів
    training_data = pd.read_csv("data/train.csv")[:1000]
    testing_data = pd.read_csv("data/test.csv")

    # Витягання ознак і цільових значень з тренувальних даних
    X = training_data.iloc[:, 1:].values
    y = training_data.iloc[:, 0].values

    # Витягання ознак і цільових значень з тестових даних
    X_test = testing_data.iloc[:, 1:].values
    y_test = testing_data.iloc[:, 0].values

    k_values = [3, 4, 5, 6, 7, 9, 10, 15, 20, 21, 40, 41]
    results = []

    for k in k_values:
        print(f"Оцінка KNN з k = {k}")

        model = KNN(k=k)
        model.fit(X, y)

        # Оцінка на тестових даних
        y_pred_test = model.predict(X_test)
        test_accuracy = evaluate_accuracy(y_test, y_pred_test)
        results.append((k, test_accuracy))
        print(f"Точність на тестових даних: {test_accuracy:.2f}")

    # Запис результатів у README.md
    with open("README.md", "a", encoding="utf-8") as readme_file:
        readme_file.write("\n\n## Результати точності KNN на тестових даних\n")
        readme_file.write("| k | Точність |\n")
        readme_file.write("|---|----------|\n")
        for k, accuracy in results:
            readme_file.write(f"| {k} | {accuracy:.2f} |\n")

    # Виконання перехресної валідації з найкращим k
    best_k = max(results, key=lambda x: x[1])[0]
    print(f"Найкраще k: {best_k}")

    num_folds = 5
    cross_val_accuracies = []

    for X_train, y_train, X_val, y_val in kfold_cross_validation(X, y, k=num_folds):
        model = KNN(k=best_k)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        accuracy = evaluate_accuracy(y_val, y_pred)
        cross_val_accuracies.append(accuracy)
        print(f"Точність на фолді: {accuracy:.2f}")

    avg_cross_val_accuracy = np.mean(cross_val_accuracies)
    print(f"Середня точність перехресної валідації: {avg_cross_val_accuracy:.2f}")


if __name__ == "__main__":
    main()
