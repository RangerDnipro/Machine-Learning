"""
Цей файл містить розв'язання завдань з використанням бібліотек NumPy та Pandas.
"""

import numpy as np
import pandas as pd


def numpy_tasks():
    """
    Виконує завдання, пов'язані з бібліотекою NumPy.
    """

    # Завдання 1.1: Операції з 1D масивом
    array_1d = np.random.randint(1, 101, size=10)  # Масив із 10 випадкових цілих чисел
    mean_value = np.mean(array_1d)  # Середнє значення
    median_value = np.median(array_1d)  # Медіана
    std_deviation = np.std(array_1d)  # Стандартне відхилення
    array_1d_even_zeroed = np.where(array_1d % 2 == 0, 0, array_1d)  # Парні числа замінено на 0

    # Завдання 1.2: Індексація та зрізка в 2D масиві
    matrix_2d = np.random.randint(1, 101, size=(3, 3))  # 2D масив (3x3)
    first_row = matrix_2d[0, :]  # Перший рядок
    last_column = matrix_2d[:, -1]  # Останній стовпець
    diagonal_elements = np.diagonal(matrix_2d)  # Діагональні елементи

    # Завдання 1.3: Broadcasting
    matrix_2d_broadcast = np.random.randint(1, 101, size=(3, 3))  # 2D масив
    array_1d_broadcast = np.random.randint(1, 101, size=3)  # 1D масив
    broadcast_result = matrix_2d_broadcast + array_1d_broadcast.reshape(1, -1)  # Broadcasting

    # Завдання 1.4: Унікальні елементи та рядки за умовою
    matrix_5x5 = np.random.randint(1, 101, size=(5, 5))  # 5x5 масив
    unique_elements = np.unique(matrix_5x5)  # Унікальні елементи
    row_sums_threshold = 150  # Поріг для суми елементів
    rows_above_threshold = matrix_5x5[np.sum(matrix_5x5, axis=1) > row_sums_threshold]  # Рядки з сумою > порогу

    # Завдання 1.5: Перетворення 1D масиву в 2D
    array_reshaped = np.arange(1, 21).reshape(4, 5)  # Масив розміром (4, 5)

    # Виведення результатів
    print("\n1D Масив:", array_1d)
    print("\nСереднє:", mean_value, "Медіана:", median_value, "Стандартне відхилення:", std_deviation)
    print("\nМасив із парними числами, заміненими на 0:", array_1d_even_zeroed)
    print("\n2D Масив:\n", matrix_2d)
    print("\nПерший рядок:", first_row)
    print("\nОстанній стовпець:", last_column)
    print("\nДіагональні елементи:", diagonal_elements)
    print("\nРезультат Broadcasting:\n", broadcast_result)
    print("\nУнікальні елементи в 5x5 масиві:\n", unique_elements)
    print("\nРядки із сумою елементів > 150:\n", rows_above_threshold)
    print("\nПеретворений масив (4x5):\n", array_reshaped)


def pandas_tasks():
    """
    Виконує завдання, пов'язані з бібліотекою Pandas.
    """

    # Завдання 2.1: Створення та фільтрація DataFrame
    data = {
        "Name": ["Alice", "Bob", "Charlie", "David", "Eve"],
        "Age": [25, 30, 35, 40, 45],
        "City": ["NY", "LA", "SF", "CHI", "SEA"]
    }
    df = pd.DataFrame(data)  # Створення DataFrame
    df["Score"] = [88, 92, 85, 78, 95]  # Додавання нового стовпця
    filtered_df = df[df["Score"] > 85]  # Фільтрація за умовою

    # Завдання 2.2: Завантаження та аналіз набору даних
    wine_data = pd.read_csv('wine.csv')  # Завантаження набору даних
    wine_stats = wine_data.describe()  # Загальна статистика для числових стовпців
    if not wine_data.select_dtypes(include=["object"]).empty:
        categorical_column = wine_data.select_dtypes(include=["object"]).columns[0]
        unique_values = wine_data[categorical_column].unique()  # Унікальні значення категорійного стовпця
    else:
        unique_values = "No categorical columns in dataset"

    # Виведення результатів
    print("\nDataFrame:\n", df)
    print("\nВідфільтрований DataFrame:\n", filtered_df)
    print("\nПерші 5 рядків набору даних Wine:\n", wine_data.head())
    print("\nЗагальна статистика:\n", wine_stats)
    print("\nУнікальні значення у категорійному стовпці:", unique_values)


if __name__ == "__main__":
    # Виконання задач
    print("Результати NumPy завдань:")
    numpy_tasks()
    print("\nРезультати Pandas завдань:")
    pandas_tasks()
