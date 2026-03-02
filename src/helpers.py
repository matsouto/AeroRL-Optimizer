import numpy as np
from scipy.special import comb
from typing import Optional


def eng_string(
    x: float,
    unit: str = "",
    format="%.3g",
    si=True,
    add_space_after_number: Optional[bool] = None,
) -> str:
    """
    Taken from: https://stackoverflow.com/questions/17973278/python-decimal-engineering-notation-for-mili-10e-3-and-micro-10e-6/40691220

    Returns float/int value <x> formatted in a simplified engineering format -
    using an exponent that is a multiple of 3.

    Args:

        x: The value to be formatted. Float or int.

        unit: A unit of the quantity to be expressed, given as a string. Example: Newtons -> "N"

        format: A printf-style string used to format the value before the exponent.

        si: if true, use SI suffix for exponent. (k instead of e3, n instead of
            e-9, etc.)

    Examples:

    With format='%.2f':
        1.23e-08 -> 12.30e-9
             123 -> 123.00
          1230.0 -> 1.23e3
      -1230000.0 -> -1.23e6

    With si=True:
          1230.0 -> "1.23k"
      -1230000.0 -> "-1.23M"

    With unit="N" and si=True:
          1230.0 -> "1.23 kN"
      -1230000.0 -> "-1.23 MN"
    """

    sign = ""
    if x < 0:
        x = -x
        sign = "-"
    elif x == 0:
        return format % 0
    elif np.isnan(x):
        return "NaN"

    exp = int(np.floor(np.log10(x)))
    exp3 = exp - (exp % 3)
    x3 = x / (10**exp3)

    if si and exp3 >= -24 and exp3 <= 24:
        if exp3 == 0:
            suffix = ""
        else:
            suffix = "yzafpnμm kMGTPEZY"[(exp3 + 24) // 3]

        if add_space_after_number is None:
            add_space_after_number = unit != ""

        if add_space_after_number:
            suffix = " " + suffix + unit
        else:
            suffix = suffix + unit

    else:
        suffix = f"e{exp3}"

        if add_space_after_number:
            add_space_after_number = unit != ""

        if add_space_after_number:
            suffix = suffix + " " + unit
        else:
            suffix = suffix + unit

    return f"{sign}{format % x3}{suffix}"


def bernstein_poly(x, n, i):
    """Calcula o polinômio de Bernstein básico"""
    return comb(n, i) * (x**i) * ((1 - x) ** (n - i))


def cst_to_coords(weights, params, n_points=100):
    """
    Converte os pesos e parâmetros do CST em coordenadas X, Y.
    weights: Matriz (2, NPV) com pesos [Superior, Inferior]
    params: Vetor com [Espessura do Bordo de Fuga, Raio do Bordo de Ataque (se usado)]
    """
    w_upper = weights[0]
    w_lower = weights[1]
    dz_te = params[0]  # Espessura do bordo de fuga (TE thickness)

    # Vetor X distribuído de 0 a 1
    x = np.linspace(0, 1, n_points)

    # Class Function (N1=0.5, N2=1.0 para aerofólios padrão)
    C = (x**0.5) * ((1 - x) ** 1.0)

    # Shape Functions (S)
    n_order = len(w_upper) - 1
    S_upper = np.zeros_like(x)
    S_lower = np.zeros_like(x)

    for i in range(len(w_upper)):
        b = bernstein_poly(x, n_order, i)
        S_upper += w_upper[i] * b
        S_lower += w_lower[i] * b

    # Calcular Y superior e inferior
    y_upper = C * S_upper + x * (dz_te / 2)
    y_lower = C * S_lower - x * (dz_te / 2)

    # Montar coordenadas em ordem (Bordo de fuga superior -> Ataque -> Fuga inferior)
    x_coords = np.concatenate([x[::-1], x[1:]])
    y_coords = np.concatenate([y_upper[::-1], y_lower[1:]])

    return x_coords, y_coords
