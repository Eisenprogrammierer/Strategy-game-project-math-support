from typing import Dict, Optional, List, Tuple


Signal = Dict[str, int]
Bitmask = int
Voltage = float


class Entity:
    """
    Игровая сущность: Data-Driven State Machine + Bitmask + Voltages.
    Хранит состояние, обрабатывает сигналы и среду, переключает биты по порогам Шмитта.
    """
    def __init__(self, entity_id: str, entity_class: str, config: dict):
        self.id = entity_id
        self.entity_class = entity_class

        self.mask_layout: Dict[str, Tuple[int, int]] = config.get("mask_layout", {})
        self.bitmask: Bitmask = 0
        self.voltages: Dict[str, Voltage] = {seg: 0.0 for seg in self.mask_layout}

        self.thresholds: Dict[str, dict] = config.get("thresholds", {})
        self.rules: List[dict] = config.get("rules", [])
        self.env_reactions: Dict[str, dict] = config.get("env_reactions", {})

        # Начальное состояние (загружается из конфига класса)
        for seg, val in config.get("initial_state", {}).items():
            self.set_segment(seg, val)

        self.parent: Optional["Entity"] = None
        self.children: List["Entity"] = []
        self.peers: List["Entity"] = []

    def get_segment(self, name: str) -> int:
        if name not in self.mask_layout:
            raise KeyError(f"Segment '{name}' not in mask layout")
        offset, length = self.mask_layout[name]
        mask = ((1 << length) - 1)
        return (self.bitmask >> offset) & mask

    def set_segment(self, name: str, value: int) -> None:
        if name not in self.mask_layout:
            return
        offset, length = self.mask_layout[name]
        max_val = (1 << length) - 1
        value = max(0, min(value, max_val))
        # Маскируем 64 битами, чтобы не уходить в произвольную точность Python
        clear_mask = ~(((1 << length) - 1) << offset) & ((1 << 64) - 1)
        self.bitmask = (self.bitmask & clear_mask) | (value << offset)

    def get_voltage(self, name: str) -> Voltage:
        return self.voltages.get(name, 0.0)

    def add_voltage(self, name: str, delta: Voltage) -> None:
        if name not in self.voltages:
            return
        self.voltages[name] += delta
        self._apply_schmitt_trigger(name)

    def _apply_schmitt_trigger(self, segment: str) -> None:
        if segment not in self.thresholds:
            return
        cfg = self.thresholds[segment]
        vol = self.voltages[segment]
        if vol >= cfg.get("high", 1.0):
            self.set_segment(segment, cfg.get("next_state_up", 1))
            self.voltages[segment] = 0.0
        elif vol <= cfg.get("low", -1.0):
            self.set_segment(segment, cfg.get("next_state_down", 0))
            self.voltages[segment] = 0.0

    def apply_signal(self, signal: Signal) -> None:
        """Рассогласование сигнал/состояние -> накопление напряжения."""
        for seg, val in signal.items():
            current = self.get_segment(seg)
            self.add_voltage(seg, val - current)

    def generate_signal(self) -> Signal:
        return {seg: self.get_segment(seg) for seg in self.mask_layout}

    def apply_environment(self, env: Dict[str, float]) -> None:
        """Свойство-Фильтр-Реакция: среда -> коэффициенты -> напряжение."""
        for modality, strength in env.items():
            reaction = self.env_reactions.get(modality)
            if reaction:
                target_seg = reaction.get("segment")
                if target_seg:
                    coeff = reaction.get("coefficient", 1.0)
                    self.add_voltage(target_seg, strength * coeff)

    def evaluate_step(self) -> None:
        """Обработка правил WHEN/THEN из конфига."""
        for rule in self.rules:
            if self._check_condition(rule.get("when", {})):
                self._execute_action(rule.get("then", {}))

    def _check_condition(self, condition: dict) -> bool:
        for seg, expected in condition.items():
            if isinstance(expected, dict):
                min_v = expected.get("min", 0)
                max_v = expected.get("max", 15)
                if not (min_v <= self.get_segment(seg) <= max_v):
                    return False
            else:
                if self.get_segment(seg) != expected:
                    return False
        return True

    def _execute_action(self, action: dict) -> None:
        for seg, val in action.get("set_segment", {}).items():
            self.set_segment(seg, val)
        for seg, delta in action.get("add_voltage", {}).items():
            self.add_voltage(seg, delta)