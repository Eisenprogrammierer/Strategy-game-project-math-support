import math
from typing import Callable, List, Tuple, Dict, Optional

# ============================================================================
# Базовые модели боевых действий
# ============================================================================


def kill_per_shot_prob(e, a, eff):
    return 1 - e**(-a * eff)


def eff_shot_quantity():
    pass


def compute_damage_ability():
    pass


def compute_forces_change():
    pass


def captured_prob():
    pass


def wounded_prob():
    pass


def killed_prob():
    pass


def compute_env_influence():
    pass


# ============================================================================
# Специфические модели и пороговые функции
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
# Стратегическая безопасность и суверенность
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