import math
from typing import Callable, List, Tuple, Dict, Optional, Sequence
from dataclasses import dataclass
from functools import reduce


@dataclass(frozen=True)
class Weapon:
    fire_rate: float         # выстрелов в такт
    base_hit_prob: float     # базовая вероятность попадания
    alpha: float             # коэффициент уязвимости (параметр Колмогорова)


@dataclass(frozen=True)
class UnitStats:
    count: float             # боеспособная численность
    morale_voltage: float    # U_morale
    is_encircled: bool       # бит state.encircled
    ops_loss_coeff: float    # b_x — операционные потери


@dataclass(frozen=True)
class EnvFactor:
    intensity: float
    susceptibility: float


@dataclass(frozen=True)
class Losses:
    killed: float
    wounded: float
    captured: float


def kill_per_shot_prob(alpha: float, expected_hits: float) -> float:
    """Вероятность поражения цели при матожидании expected_hits попаданий."""
    return 1.0 - math.exp(-alpha * expected_hits)


def eff_shot_quantity(
    fire_rate: float,
    hit_prob: float,
    dt: float = 1.0,
) -> float:
    """Ожидаемое число попаданий в одну цель за такт."""
    return fire_rate * hit_prob * dt


def compute_damage_ability(
    weapon: Weapon,
    expected_hits: float,
    beta: float,
    k_env: float,
    k_front: float = 1.0,
) -> float:
    """Поражающая способность стороны за такт (коэффициент Ланчестера)."""
    p_kill = kill_per_shot_prob(weapon.alpha, expected_hits)
    return weapon.fire_rate * weapon.base_hit_prob * p_kill * beta * k_env * k_front


def compute_forces_change(
    x: UnitStats,
    y_damage: float,     # a_y — урон, наносимый Y по X
    y: UnitStats,
    x_damage: float,     # a_x — урон, наносимый X по Y
    dt: float = 1.0,
) -> tuple[float, float]:
    """Возвращает (Δx, Δy) — приращения численности за такт."""
    dx = -y_damage * y.count * dt - x.ops_loss_coeff * x.count * dt
    dy = -x_damage * x.count * dt - y.ops_loss_coeff * y.count * dt
    return dx, dy


def captured_prob(
    base_prob: float,
    morale_voltage: float,
    is_encircled: bool,
    k_m: float = 1.0,
) -> float:
    """Вероятность пленения/дезертирства. Экспоненциально растёт от стресса."""
    if not is_encircled and morale_voltage <= 0.0:
        return base_prob
    encircled_mul = 2.5 if is_encircled else 1.0
    return min(1.0, base_prob * math.exp(k_m * morale_voltage) * encircled_mul)


def wounded_prob(base_prob: float, p_capture: float) -> float:
    """Вероятность ранения (условно-зависимая от плена)."""
    return base_prob * (1.0 - p_capture)


def killed_prob(p_capture: float, p_wounded: float) -> float:
    """Вероятность безвозвратной гибели — дополнение до единицы."""
    return max(0.0, 1.0 - p_capture - p_wounded)


def compute_loss_distribution(
    base_capture: float,
    base_wounded: float,
    morale_voltage: float,
    is_encircled: bool,
) -> tuple[float, float, float]:
    """Возвращает (p_killed, p_wounded, p_captured) — нормированное распределение."""
    pc = captured_prob(base_capture, morale_voltage, is_encircled)
    pw = wounded_prob(base_wounded, pc)
    pk = killed_prob(pc, pw)
    return pk, pw, pc


def compute_env_influence(factors: Sequence[EnvFactor]) -> float:
    """Итоговый модификатор среды ∈ (0, 1]."""
    if not factors:
        return 1.0
    return reduce(
        lambda acc, f: acc * (1.0 - f.susceptibility * f.intensity),
        factors,
        1.0,
    )


def tick(
    x: UnitStats, y: UnitStats,
    wx: Weapon, wy: Weapon,
    env_x: Sequence[EnvFactor], env_y: Sequence[EnvFactor],
    beta_xy: float, beta_yx: float,
    k_front: float = 1.0,
    dt: float = 1.0,
    base_capture: float = 0.1,
    base_wounded: float = 0.4,
) -> tuple[float, float, Losses, Losses]:
    """
    Один такт боя. Возвращает (new_x_count, new_y_count, losses_x, losses_y).
    Все функции — чистые, состояние передаётся как значения, не мутируется.
    """
    # Среда
    k_env_x = compute_env_influence(env_x)
    k_env_y = compute_env_influence(env_y)

    # Эффективные попадания (упрощённо — без модификаторов среды на P_hit)
    h_x = eff_shot_quantity(wx.fire_rate, wx.base_hit_prob * k_env_x, dt)
    h_y = eff_shot_quantity(wy.fire_rate, wy.base_hit_prob * k_env_y, dt)

    # Способности
    a_x = compute_damage_ability(wx, h_y, beta_xy, k_env_y, k_front)
    a_y = compute_damage_ability(wy, h_x, beta_yx, k_env_x, k_front)

    # Дельты численности
    dx, dy = compute_forces_change(x, a_y, y, a_x, dt)

    # Распределение потерь по типам (для стороны X)
    pk_x, pw_x, pc_x = compute_loss_distribution(
        base_capture, base_wounded, x.morale_voltage, x.is_encircled,
    )
    # Для стороны Y
    pk_y, pw_y, pc_y = compute_loss_distribution(
        base_capture, base_wounded, y.morale_voltage, y.is_encircled,
    )

    # Абсолютные значения потерь
    loss_x_total = -dx
    loss_y_total = -dy

    losses_x = Losses(
        killed=loss_x_total * pk_x,
        wounded=loss_x_total * pw_x,
        captured=loss_x_total * pc_x,
    )
    losses_y = Losses(
        killed=loss_y_total * pk_y,
        wounded=loss_y_total * pw_y,
        captured=loss_y_total * pc_y,
    )

    new_x = max(0.0, x.count + dx)
    new_y = max(0.0, y.count + dy)
    return new_x, new_y, losses_x, losses_y


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