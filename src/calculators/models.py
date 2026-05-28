import math
from typing import Callable, List, Tuple, Dict, Optional

# ============================================================================
# БЛОК 1: Вероятности победы и параметры превосходства
# ============================================================================

def calc_combat_superiority(
    morale_x: float, morale_y: float,
    tech_s_x: float, tech_s_y: float,
    tech_v_x: float, tech_v_y: float,
    tech_p_x: float, tech_p_y: float,
    tech_m_x: float, tech_m_y: float
) -> float:
    """
    \beta = \var_phi * \rho
    \var_phi - моральное превосходство, \rho - технологическое
    (среднее геометрическое произвольного числа осей, пока что 4)
    """
    phi = morale_x / (morale_y + 1e-9)
    rho = (
        (tech_s_x / (tech_s_y + 1e-9)) *
        (tech_v_x / (tech_v_y + 1e-9)) *
        (tech_p_x / (tech_p_y + 1e-9)) *
        (tech_m_x / (tech_m_y + 1e-9))
    ) ** 0.25
    return phi * rho


def victory_probability(r_x: float, r_y: float, beta: float, m: float = 1.0) -> float:
    """
    P(x > y) = (\beta * r_x)^m / ((\beta * r_x)^m + (\beta * r_y)^m)
    Возвращает вероятность победы стороны X над Y.
    """
    num = (beta * r_x) ** m
    den = (beta * r_y) ** m
    return num / (num + den + 1e-9)

# ============================================================================
# БЛОК 2: Динамика потерь (Ланчестер/Осипов)
# ============================================================================

def lanchester_step(x: float, y: float, a_x: float, a_y: float, dt: float = 1.0) -> Tuple[float, float]:
    """
    Дискретный квадратичный закон: x_{n+1} = x_n - a_y * dt * y_n
    """
    dx = -a_y * dt * y
    dy = -a_x * dt * x
    return max(0.0, x + dx), max(0.0, y + dy)


def generalized_lanchester_step(
    x: float, y: float, a_x: float, a_y: float, 
    p: float, q: float, dt: float = 1.0
) -> Tuple[float, float]:
    """
    Обобщённая модель: dx/dt = -a_y * y^p * x^q
    p=1, q=1 -> квадратичная модель (развитый огнестрел).
    p=1, q=0 -> линейная модель (рудиментарный огнестрел и рукопашный бой).
    Если в victory_probability m = 3, график будет параболлическим
    (сверхмощное оружие - научная фантастика или магия)
    """
    dx = -a_y * (y ** p) * (x ** q) * dt
    dy = -a_x * (x ** p) * (y ** q) * dt
    return max(0.0, x + dx), max(0.0, y + dy)

# ============================================================================
# БЛОК 3: Вероятности поражения и полная вероятность (Колмогоров)
# Источник: "Базовые функции в моделях.md" -> Критерий эффективности стрельбы
# ============================================================================

def kolmogorov_hit_prob(alpha: float, hits: int) -> float:
    """
    P(A|x) = 1 - e^{-\alpha x}
    """
    return 1.0 - math.exp(-alpha * hits)


def bernoulli_prob(n: int, m: int, p: float) -> float:
    """Вероятность ровно m попаданий из n выстрелов."""
    if not (0 <= m <= n) or not (0.0 <= p <= 1.0):
        return 0.0
    return math.comb(n, m) * (p ** m) * ((1 - p) ** (n - m))


def total_probability(p_hit: float, n_shots: int, g_func: Callable[[int], float]) -> float:
    """
    W = \sum P_{m,n} * G(m)
    g_func: лямбда/функция, возвращающая вероятность уничтожения цели при m попаданиях.
    """
    W = 0.0
    for m in range(n_shots + 1):
        W += bernoulli_prob(n_shots, m, p_hit) * g_func(m)
    return W

# ============================================================================
# БЛОК 4: Специфические модели и пороговые функции
# ============================================================================

def markov_step(prev_probs: List[float], trans_matrix: List[List[float]]) -> List[float]:
    """
    P_i(k) = \sum P_j(k-1) * p_{ij}
    Чистая функциональная реализация без numpy.
    """
    n = len(prev_probs)
    next_probs = [0.0] * n
    for j in range(n):
        if prev_probs[j] == 0.0:
            continue
        for i in range(n):
            next_probs[i] += prev_probs[j] * trans_matrix[j][i]
    return next_probs


def pavlovsky_step(x: float, y: float, y0: float, a_y: float, phi_crit: float, dt: float = 1.0) -> float:
    """
    Индикаторная модель Павловского: dx/dt = -a_y * y * I(y/y0 > φ)
    """
    indicator = 1.0 if (y / (y0 + 1e-9)) > phi_crit else 0.0
    return max(0.0, x - a_y * y * indicator * dt)


def combat_continuation_prob(x0: float, x: float, lam: float) -> float:
    """
    P_x(t) = 1 - ((x0 - x(t))/x0)^λ
    Вероятность, что сторона продолжит бой несмотря на потери.
    """
    if x0 <= 0:
        return 0.0
    loss_ratio = max(0.0, min(1.0, (x0 - x) / x0))
    return 1.0 - (loss_ratio ** lam)


def gaussian_dispersion_prob(x: float, y: float, ax: float, ay: float, sx: float, sy: float) -> float:
    """Плотность вероятности рассеивания (нормальное распределение Гаусса)."""
    coeff = 1.0 / (2.0 * math.pi * sx * sy)
    exp_val = -0.5 * (((x - ax)**2 / sx**2) + ((y - ay)**2 / sy**2))
    return coeff * math.exp(exp_val)

# ============================================================================
# БЛОК 5: Стратегическая безопасность и суверенность
# ============================================================================

def base_sovereignty(z: float, z_max: float, s: float, s_max: float, omega_z: float, omega_s: float) -> float:
    """w_bi = (z/z_max)^ωz * (s/s_max)^ωs"""
    return ((z / (z_max + 1e-9)) ** omega_z) * ((s / (s_max + 1e-9)) ** omega_s)


def social_tech_factor(I_i: float, chi: float) -> float:
    """A_i = (1 + I_i)^χ"""
    return (1.0 + I_i) ** chi


def sovereignty_function(w_bi: float, A_i: float) -> float:
    """w_i = A_i * w_bi"""
    return A_i * w_bi


def security_utility(w_i: float, q_i: float) -> float:
    """u_i = w_i * q_i (Суверенность * Выживание)"""
    return w_i * q_i


COMBAT_MODELS_REGISTRY: Dict[str, Callable] = {
    'lanchester_step': lambda kwargs: lanchester_step(**kwargs),
    'generalized_lanchester': lambda kwargs: generalized_lanchester_step(**kwargs),
    'victory_prob': lambda kwargs: victory_probability(**kwargs),
    'combat_continuation': lambda kwargs: combat_continuation_prob(**kwargs),
    'pavlovsky': lambda kwargs: pavlovsky_step(**kwargs),
    'security_model': lambda kwargs: security_utility(**kwargs),
    # Добавлять новые шаблоны по мере надобности
}