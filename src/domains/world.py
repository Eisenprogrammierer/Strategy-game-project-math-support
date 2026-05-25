from typing import Dict, List, Tuple
from entity import Entity, Signal
from protocol import Protocol

class EntityClass:
    """Приспособленец конфигураций. Один экземпляр на тип сущности."""
    def __init__(self, class_id: str, config: dict):
        self.id = class_id
        self.config = config

class World:
    """
    Контроллер симуляции. Реализует фазовую буферизацию тактов (t / t+1),
    детерминированный обход графа и агрегацию входящих сигналов.
    """
    def __init__(self, environment: Dict[str, float] = None):
        self.entities: Dict[str, Entity] = {}
        self.connections: List[Tuple[Protocol, str, str, str]] = []
        self.classes: Dict[str, EntityClass] = {}
        self.env = environment or {}
        
        # Двойной буфер: только для входящих сигналов. Состояние живёт в Entity.
        self.buffer_t: Dict[str, Signal] = {}
        self.buffer_t1: Dict[str, Signal] = {}

    def register_class(self, class_id: str, config: dict) -> EntityClass:
        cls = EntityClass(class_id, config)
        self.classes[class_id] = cls
        return cls

    def create_entity(self, entity_id: str, class_id: str) -> Entity:
        entity = Entity(entity_id, class_id, self.classes[class_id].config)
        self.entities[entity_id] = entity
        self.buffer_t[entity_id] = {}
        self.buffer_t1[entity_id] = {}
        return entity

    def add_connection(self, protocol: Protocol, sender_id: str, receiver_id: str, direction: str = "down") -> None:
        self.connections.append((protocol, sender_id, receiver_id, direction))

    def _accumulate(self, target: Signal, source: Signal) -> None:
        for seg, val in source.items():
            target[seg] = target.get(seg, 0) + val

    def step(self) -> None:
        # 1. Очистка буфера следующего такта
        for eid in self.entities:
            self.buffer_t1[eid] = {}

        # 2. Фаза протоколов: читаем состояния t, вычисляем сигналы, пишем в t+1
        for proto, s_id, r_id, direction in self.connections:
            if s_id in self.entities and r_id in self.entities:
                signal = proto.compute_signal(self.entities[s_id], direction)
                self._accumulate(self.buffer_t1[r_id], signal)

        # 3. Фаза сущностей: детерминированный обход, применение сигналов и среды
        for eid in sorted(self.entities.keys()):
            entity = self.entities[eid]
            entity.apply_signal(self.buffer_t1[eid])
            entity.apply_environment(self.env)
            entity.evaluate_step()

        # 4. Сдвиг буферов
        self.buffer_t = self.buffer_t1
        self.buffer_t1 = {eid: {} for eid in self.entities}