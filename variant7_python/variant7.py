## @file variant7.py
#  @brief Лабораторная работа №2, вариант 7: \f$ f(x) = 1 + \sin x - 1.15 e^{-x} \f$.
#
#  Поиск наименьшего положительного корня с точностью \f$ \varepsilon = 0.5 \cdot 10^{-5} \f$
#  методом Ньютона и методом простых итераций.
#
#  Запуск: @code python3 variant7.py @endcode
#  @author Участник Б (вариант 7)
#  @date 2026

import math
from methods import newton, simple_iteration, a_priori_iterations

## Требуемая точность.
EPS = 0.5e-5
## Левая граница отрезка локализации.
A = 0.0
## Правая граница отрезка локализации.
B = 0.1
## Параметр λ в преобразовании \f$ \varphi(x) = x - \lambda f(x) \f$.
LAMBDA = 0.47


## @brief Исследуемая функция \f$ f(x) = 1 + \sin x - 1.15 e^{-x} \f$.
#  @param x аргумент
#  @return значение f(x)
def f(x: float) -> float:
    return 1 + math.sin(x) - 1.15 * math.exp(-x)


## @brief Первая производная \f$ f'(x) = \cos x + 1.15 e^{-x} \f$.
#  @param x аргумент
#  @return значение f'(x)
def df(x: float) -> float:
    return math.cos(x) + 1.15 * math.exp(-x)


## @brief Вторая производная \f$ f''(x) = -\sin x - 1.15 e^{-x} \f$.
#  @param x аргумент
#  @return значение f''(x)
def d2f(x: float) -> float:
    return -math.sin(x) - 1.15 * math.exp(-x)


## @brief Функция для МПИ: \f$ \varphi(x) = x - 0.47\,f(x) \f$.
#
#  \f$ \varphi'(x) = 1 - 0.47 f'(x) \in [-0.0105;\ 0.0433] \f$ на [0; 0.1], поэтому q ≈ 0.0433.
#  @param x аргумент
#  @return значение φ(x)
def phi(x: float) -> float:
    return x - LAMBDA * f(x)


## @brief Точка входа: вычисляет корень двумя методами и печатает таблицу результатов.
def main() -> None:
    # Константы для метода Ньютона: f' убывает, |f''| максимален в 0 на [0; 0.1]
    m1 = df(B)
    M2 = abs(d2f(A))
    # Коэффициент сжатия для МПИ
    q = max(abs(1 - LAMBDA * df(A)), abs(1 - LAMBDA * df(B)))
    x0_mpi = (A + B) / 2

    print("Вариант 7: f(x) = 1 + sin x - 1.15e^(-x)")
    print(f"Отрезок [{A}; {B}]: f(a) = {f(A):.4f}, f(b) = {f(B):.4f}")
    print(f"Ньютон: m1 = {m1:.4f}, M2 = {M2:.4f}, x0 = {A} (f(x0)*f''(x0) = {f(A) * d2f(A):.4f} > 0)")
    print(f"МПИ: q = {q:.4f}, априорная оценка n >= {a_priori_iterations(phi, x0_mpi, q, EPS)}")
    print()

    results = [
        newton(f, df, A, m1, M2, EPS),
        simple_iteration(phi, x0_mpi, q, EPS),
    ]
    print(f"{'Метод':<10}{'Нач. прибл.':>14}{'Итераций':>10}{'Корень':>14}")
    for r in results:
        print(f"{r.method:<10}{r.x0:>14.4f}{r.iterations:>10}{r.root:>14.7f}")


if __name__ == "__main__":
    main()
