## @file methods.py
#  @brief Численные методы уточнения корня: метод Ньютона и метод простых итераций (МПИ).
#  @author Участник Б (вариант 7)
#  @date 2026

import math
from dataclasses import dataclass
from typing import Callable


## @brief Результат работы численного метода.
@dataclass
class RootResult:
    ## Название метода.
    method: str
    ## Начальное приближение x0.
    x0: float
    ## Фактическое число выполненных итераций.
    iterations: int
    ## Полученное приближённое значение корня.
    root: float


## @brief Метод Ньютона (касательных).
#
#  Расчётная формула:
#  \f[ x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)} \f]
#
#  Условие выхода из цикла:
#  \f[ |x_n - x_{n-1}| \le \sqrt{\frac{2 m_1 \varepsilon}{M_2}} \f]
#  где \f$ m_1 = \min |f'(x)| \f$, \f$ M_2 = \max |f''(x)| \f$ на отрезке локализации.
#
#  @param f      исследуемая функция f(x)
#  @param df     производная f'(x)
#  @param x0     начальное приближение (выбирается по условию Фурье: f(x0)·f''(x0) > 0)
#  @param m1     минимум модуля первой производной на отрезке
#  @param M2     максимум модуля второй производной на отрезке
#  @param eps    требуемая точность
#  @param max_iter защита от зацикливания
#  @return объект RootResult с корнем и числом итераций
#  @exception ZeroDivisionError если f'(x_n) = 0
def newton(f: Callable[[float], float], df: Callable[[float], float],
           x0: float, m1: float, M2: float, eps: float,
           max_iter: int = 1000) -> RootResult:
    stop = math.sqrt(2 * m1 * eps / M2)
    x_prev = x0
    for n in range(1, max_iter + 1):
        d = df(x_prev)
        if d == 0:
            raise ZeroDivisionError("f'(x) = 0, метод Ньютона неприменим")
        x = x_prev - f(x_prev) / d
        if abs(x - x_prev) <= stop:
            return RootResult("Ньютона", x0, n, x)
        x_prev = x
    raise RuntimeError("Метод Ньютона не сошёлся за max_iter итераций")


## @brief Метод простых итераций (МПИ).
#
#  Уравнение предварительно приводится к виду \f$ x = \varphi(x) \f$.
#  Расчётная формула: \f$ x_{n+1} = \varphi(x_n) \f$.
#
#  Достаточное условие сходимости: \f$ |\varphi'(x)| \le q < 1 \f$ на отрезке
#  и \f$ \varphi([a,b]) \subset [a,b] \f$.
#
#  Условие выхода из цикла:
#  \f[ |x_n - x_{n-1}| \le \frac{1-q}{q}\,\varepsilon \f]
#
#  @param phi    функция \f$ \varphi(x) \f$
#  @param x0     начальное приближение
#  @param q      оценка \f$ \max|\varphi'(x)| \f$ на отрезке, 0 < q < 1
#  @param eps    требуемая точность
#  @param max_iter защита от зацикливания
#  @return объект RootResult с корнем и числом итераций
def simple_iteration(phi: Callable[[float], float], x0: float, q: float,
                     eps: float, max_iter: int = 1000) -> RootResult:
    stop = (1 - q) / q * eps
    x_prev = x0
    for n in range(1, max_iter + 1):
        x = phi(x_prev)
        if abs(x - x_prev) <= stop:
            return RootResult("МПИ", x0, n, x)
        x_prev = x
    raise RuntimeError("МПИ не сошёлся за max_iter итераций")


## @brief Априорная оценка числа итераций МПИ.
#
#  \f[ n \ge \frac{\ln\left(\dfrac{\varepsilon (1-q)}{|x_1 - x_0|}\right)}{\ln q} \f]
#
#  @param phi функция \f$ \varphi(x) \f$
#  @param x0  начальное приближение
#  @param q   коэффициент сжатия
#  @param eps требуемая точность
#  @return минимальное целое n, удовлетворяющее оценке
def a_priori_iterations(phi: Callable[[float], float], x0: float,
                        q: float, eps: float) -> int:
    x1 = phi(x0)
    return math.ceil(math.log(eps * (1 - q) / abs(x1 - x0)) / math.log(q))
