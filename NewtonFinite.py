def newton_finite_differences_first(x, x_value, table):
    n = len(x)
    coef = table[0]
    t = (x_value - x[0]) / (x[1] - x[0])
    result = round(coef[0], 4)

    for j in range(1, n):
        delta_t = t
        term = round(coef[j], 4)
        for i in range(j):
            term *= (delta_t - i)
            term /= (i + 1)
        result += term
    return result


def newton_finite_differences_second(x, x_value, coef):
    n = len(x)
    t = (x_value - x[-1]) / (x[1] - x[0])
    result = round(coef[-1][0], 4)
    for j in range(1, n):
        delta_t = t
        term = round(coef[-(j + 1)][j], 4)
        for i in range(j):
            term *= (delta_t + i)
            term /= (i + 1)
        result += term
    return result
