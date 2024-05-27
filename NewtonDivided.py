import numpy as np
from prettytable import PrettyTable


def newton_divided_differences(x, y):
    n = len(x)
    coef = np.zeros([n, n])
    coef[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            coef[i][j] = (coef[i + 1][j - 1] - coef[i][j - 1]) / (x[i + j] - x[i])

    return coef


def print_diffs(x, coef):
    table = PrettyTable()
    table.field_names = ["x", "y"] + [f"d{i}y" for i in range(1, len(x))]
    for i in range(len(x)):
        row = [x[i]] + [round(value, 4) for value in coef[i]]
        table.add_row(row)
    print("Таблица разделенных разностей:\n", table)


def newton_interpolation(x_data, y_data, x):
    coef = newton_divided_differences(x_data, y_data)
    n = len(x_data) - 1
    p = coef[0][n]
    for k in range(1, n + 1):
        p = coef[0][n - k] + (x - x_data[n - k]) * p
    return p


