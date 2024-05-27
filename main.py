import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from Lagrange import lagrange_interpolation
from NewtonDivided import newton_interpolation, print_diffs, newton_divided_differences
from NewtonFinite import newton_finite_differences_first, newton_finite_differences_second


def finite_difference_table(x, y):
    n = len(y)
    table = np.zeros((n, n))
    table[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            table[i, j] = table[i + 1, j - 1] - table[i, j - 1]

    return table


def check_equal_spacing(x):
    spacing = np.diff(x)
    return np.allclose(spacing, spacing[0])


def input_data_from_keyboard():
    while True:
        try:
            n = int(input("Введите количество точек: "))
            if n <= 0:
                raise ValueError("Количество точек должно быть положительным.")
            elif n == 1:
                raise ValueError("Должно быть как минимум две точки.")
            break
        except ValueError as e:
            print(f"Ошибка: {e}. Попробуйте снова.")

    x = []
    y = []
    points = set()
    for i in range(n):
        while True:
            try:
                x_val = float(input(f"Введите x[{i}]: "))
                y_val = float(input(f"Введите y[{i}]: "))
                point = (x_val, y_val)
                if point in points:
                    raise ValueError("Эта точка уже была введена. Введите другую точку.")
                points.add(point)
                x.append(x_val)
                y.append(y_val)
                break
            except ValueError as e:
                print(f"Ошибка: {e}. Попробуйте снова.")

    return np.array(x), np.array(y)


def input_data_from_file():
    while True:
        try:
            filename = input("Введите имя файла: ")
            data = pd.read_csv(filename)
            if 'x' not in data.columns or 'y' not in data.columns:
                raise ValueError("Файл должен содержать столбцы 'x' и 'y'.")
            points = set()
            for i in range(len(data)):
                point = (data['x'][i], data['y'][i])
                if point in points:
                    raise ValueError("Файл содержит дублирующиеся точки.")
                points.add(point)
            return data['x'].values, data['y'].values
        except FileNotFoundError:
            print("Ошибка: Файл не найден. Попробуйте снова.")
        except pd.errors.EmptyDataError:
            print("Ошибка: Файл пуст. Попробуйте снова.")
        except ValueError as e:
            print(f"Ошибка: {e}. Попробуйте снова.")


def generate_data_from_function():
    func_dict = {
        1: np.sin,
        2: np.cos,
        3: np.exp
    }

    while True:
        try:
            print("Выберите функцию:")
            print("1: sin(x)")
            print("2: cos(x)")
            print("3: exp(x)")
            func_choice = int(input("Ваш выбор (1/2/3): "))
            if func_choice not in func_dict:
                raise ValueError("Неверный выбор функции.")
            break
        except ValueError as e:
            print(f"Ошибка: {e}. Попробуйте снова.")

    chosen_func = func_dict[func_choice]

    while True:
        try:
            start = float(input("Введите начальное значение интервала: "))
            end = float(input("Введите конечное значение интервала: "))
            if start >= end:
                raise ValueError("Начальное значение интервала должно быть меньше конечного.")
            break
        except ValueError as e:
            print(f"Ошибка: {e}. Попробуйте снова.")

    while True:
        try:
            num_points = int(input("Введите количество точек: "))
            if num_points <= 0:
                raise ValueError("Количество точек должно быть положительным.")
            break
        except ValueError as e:
            print(f"Ошибка: {e}. Попробуйте снова.")

    x = np.linspace(start, end, num_points)
    y = chosen_func(x)
    return x, y


def main():
    while True:
        try:
            print("Выберите способ ввода данных:")
            print("1: Ввод с клавиатуры")
            print("2: Загрузка из файла")
            print("3: Генерация на основе функции")
            choice = input("Ваш выбор (1/2/3): ")
            if choice not in ['1', '2', '3']:
                raise ValueError("Неверный выбор.")
            break
        except ValueError as e:
            print(f"Ошибка: {e}. Попробуйте снова.")

    if choice == '1':
        x, y = input_data_from_keyboard()
    elif choice == '2':
        x, y = input_data_from_file()
    elif choice == '3':
        x, y = generate_data_from_function()

    if x is not None and y is not None:
        sorted_indices = np.argsort(x)
        x = x[sorted_indices]
        y = y[sorted_indices]
        table_points = PrettyTable()
        table_points.field_names = ["x", "y"]
        for xi, yi in zip(x, y):
            table_points.add_row([round(xi, 6), round(yi, 6)])

        print("\nТаблица точек (x, y):")
        print(table_points)

        while True:
            try:
                x_val = float(input("Введите значение аргумента для интерполяции: "))
                if x_val < min(x) or x_val > max(x):
                    raise ValueError("Значение аргумента должно быть в пределах интервала значений x.")
                break
            except ValueError as e:
                print(f"Ошибка: {e} Попробуйте снова.")

        y_val_lag = lagrange_interpolation(x, y, x_val)
        print(f"\nПриближенное значение функции в точке (многочлен Лагранжа) {x_val}: {y_val_lag}")
        x_plot = np.linspace(min(x), max(x), 50)
        y_plot_lagrange = [lagrange_interpolation(x, y, xi) for xi in x_plot]

        plt.figure(figsize=(10, 8))
        plt.plot(x, y, 'o', label='Узлы интерполяции')
        plt.plot(x_plot, y_plot_lagrange, 'g-', label='Многочлен Лагранжа')
        plt.plot(x_val, y_val_lag, 'rx', label=f'Интерполированное значение (Лагранж) {y_val_lag}')
        if check_equal_spacing(x):
            print("\nЗначения x равномерно распределены.")
            table = finite_difference_table(x, y)
            # df = pd.DataFrame(table)
            # print("\nТаблица конечных разностей (через Pandas DataFrame):")
            # print(df)

            pt = PrettyTable()
            pt.field_names = [f"Δ^{i}y" for i in range(len(table))]
            for row in table:
                pt.add_row([round(value, 4) for value in row])

            print("\nТаблица конечных разностей:")
            print(pt)

            if x_val < x[round(len(x) / 2)]:
                print("Рассматриваемое значение лежит в первой половине")
                y_val_new_fin_first = newton_finite_differences_first(x, x_val, table)
                print(
                    f"\nПриближенное значение функции в точке (многочлен Ньютона вперед) {x_val}: {y_val_new_fin_first}")
                y_plot_newton_fin = [newton_finite_differences_first(x, xi, table) for xi in x_plot]
                plt.plot(x_plot, y_plot_newton_fin, 'b-', label='Многочлен Ньютона для конечных разностей')
                plt.plot(x_val, y_val_new_fin_first, 'rv',
                         label=f'Интерполированное значение (Ньютон равноотстоящий вперед) {y_val_new_fin_first}')
            else:
                print("Рассматриваемое значение лежит во второй  половине")
                y_val_new_fin_second = newton_finite_differences_second(x, x_val, table)
                print(
                    f"\nПриближенное значение функции в точке (многочлен Ньютона назад) {x_val}: {y_val_new_fin_second}")
                y_plot_newton_fin = [newton_finite_differences_second(x, xi, table) for xi in x_plot]
                plt.plot(x_plot, y_plot_newton_fin, 'r-', label='Многочлен Ньютона для конечных разностей')
                plt.plot(x_val, y_val_new_fin_second, 'rx',
                         label=f'Интерполированное значение (Ньютон равноотстоящий назад) {y_val_new_fin_second}')

        else:
            print("\nВнимание: значения x не равномерно распределены.")
            y_val_new = newton_interpolation(x, y, x_val)
            print_diffs(x, newton_divided_differences(x, y))
            print(f"\nПриближенное значение функции в точке (многочлен Ньютона) {x_val}: {y_val_new}")
            y_plot_newton = [newton_interpolation(x, y, xi) for xi in x_plot]
            plt.plot(x_plot, y_plot_newton, 'r-', label='Многочлен Ньютона для разделенных разностей')
            plt.plot(x_val, y_val_new, 'ro', label=f'Интерполированное значение (Ньютон) {y_val_new}')

        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.title('Интерполяция: Лагранж и Ньютон')
        plt.show()
    else:
        print("Ошибка при получении данных")


if __name__ == "__main__":
    main()
