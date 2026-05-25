from typing import Dict, List, Optional
from entity import Entity, Signal

class Protocol:
    """
    Адаптер-ребро графа. Stateless, двунаправленный.
    Преобразует сигналы между сущностями по правилам маппинга.
    """
    def __init__(self, protocol_id: str, config: dict):
        self.id = protocol_id
        self.rules_down: List[dict] = config.get("down", [])
        self.rules_up: List[dict] = config.get("up", [])

    def _transform(self, value: int, rule: dict) -> Optional[int]:
        t_type = rule.get("type", "direct")
        if t_type == "direct":
            return value
        if t_type == "linear":
            src_max = rule.get("source_max", 15)
            tgt_max = rule.get("target_max", 15)
            if src_max <= 0: return 0
            return int(round(value * (tgt_max / src_max)))
        if t_type == "threshold":
            thr = rule.get("threshold", 0.5)
            return 1 if value >= thr else 0
        return None

    def compute_signal(self, sender: Entity, direction: str = "down") -> Signal:
        """Вычисляет преобразованный сигнал без применения к получателю."""
        rules = self.rules_down if direction == "down" else self.rules_up
        signal: Signal = {}
        for rule in rules:
            src = rule.get("source_segment")
            tgt = rule.get("target_segment")
            if not src or not tgt:
                continue
            try:
                val = sender.get_segment(src)
                transformed = self._transform(val, rule)
                if transformed is not None:
                    signal[tgt] = transformed
            except KeyError:
                continue
        return signal

    # --- Интерфейс (для прямой передачи вне тактового цикла) ---
    def transfer(self, sender: Entity, receiver: Entity, direction: str = "down") -> None:
        signal = self.compute_signal(sender, direction)
        if signal:
            receiver.apply_signal(signal)

    def transfer_down(self, parent: Entity, child: Entity) -> None:
        self.transfer(parent, child, "down")

    def transfer_up(self, child: Entity, parent: Entity) -> None:
        self.transfer(child, parent, "up")