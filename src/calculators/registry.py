from typing import Callable, Dict, Any
import math
from .models import (
    calc_combat_superiority, victory_probability,
    lanchester_step, generalized_lanchester_step,
    kolmogorov_hit_prob, bernoulli_prob, total_probability,
    markov_step, pavlovsky_step, combat_continuation_prob,
    gaussian_dispersion_prob,
    base_sovereignty, social_tech_factor, sovereignty_function, security_utility
)


def make_kolmogorov_g(alpha: float = 1.0) -> Callable[[int], float]:
    """
    G(m) = 1 - \exp(-\alpha * m)
    Экспоненциальный закон поражения цели по Колмогорову.
    Возвращает замыкание, готовое к подстановке в total_probability.
    """
    return lambda m: 1.0 - math.exp(-alpha * m)


MODELS_REGISTRY: Dict[str, Callable[..., Any]] = {
    # Превосходство и вероятность победы
    'combat_superiority': calc_combat_superiority,
    'victory_probability': victory_probability,

    # Динамика потерь (дискретный шаг)
    'lanchester_step': lanchester_step,
    'generalized_lanchester_step': generalized_lanchester_step,
    'pavlovsky_step': pavlovsky_step,
    'combat_continuation_prob': combat_continuation_prob,

    # Стрельба, рассеивание, полная вероятность
    'kolmogorov_hit_prob': kolmogorov_hit_prob,
    'bernoulli_prob': bernoulli_prob,
    'total_probability': total_probability,
    'gaussian_dispersion': gaussian_dispersion_prob,

    # Марковские цепи
    'markov_step': markov_step,

    # Суверенность, безопасность, социальные технологии
    'base_sovereignty': base_sovereignty,
    'social_tech_factor': social_tech_factor,
    'sovereignty_function': sovereignty_function,
    'security_utility': security_utility,
}


def get_model(name: str) -> Callable:
    """Безопасное получение функции из реестра. Выбрасывает KeyError, если модель не найдена."""
    if name not in MODELS_REGISTRY:
        raise KeyError(f"Модель '{name}' не найдена в реестре. Доступные: {list(MODELS_REGISTRY.keys())}")
    return MODELS_REGISTRY[name]