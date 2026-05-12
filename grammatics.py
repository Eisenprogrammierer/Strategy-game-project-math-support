from dataclasses import dataclass, field
from typing import Dict, List, Any, Set, Optional
import random
import pandas as pd


@dataclass
class Domain:
    name: str
    characteristics: Dict[str, Any] = field(default_factory=dict)
    value_pools: Dict[str, List[str]] = field(default_factory=dict)


class BaseIPOAlphabet:
    """Базовая схема: Input → Process → Output → State"""
    IPO_CATEGORIES = ("input", "process", "output", "state")
    
    def __init__(self):
        self.structure: Dict[str, Dict[str, List[str]]] = {
            cat: {} for cat in self.IPO_CATEGORIES
        }


class SpecializedIPOAlphabet(BaseIPOAlphabet):
    def __init__(self, entity_class: str, domain: Domain, trait_spec: Dict[str, Dict[str, List[str]]]):
        super().__init__()
        self.entity_class = entity_class
        self.domain = domain
        
        for cat in self.IPO_CATEGORIES:
            self.structure[cat] = trait_spec.get(cat, {})
            
    def get_all_traits(self) -> Set[str]:
        """Возвращает полный набор ожидаемых черт"""
        return {trait for traits_dict in self.structure.values() for trait in traits_dict}


class EntityGenerator:
    def __init__(self, spec_alphabet: SpecializedIPOAlphabet):
        self.spec = spec_alphabet

    def generate(self, entity_id: str) -> Dict[str, Any]:
        """Собирает валидную сущность из специализированного алфавита и домена"""
        entity = {
            "id": entity_id,
            "entity_class": self.spec.entity_class,
            "domain": self.spec.domain.name
        }

        for category, traits in self.spec.structure.items():
            for trait_name, allowed_values in traits.items():
                if allowed_values:
                    entity[trait_name] = random.choice(allowed_values)
                else:
                    fallback = self.spec.domain.value_pools.get(trait_name, [None])
                    entity[trait_name] = random.choice(fallback)

        entity["domain_attrs"] = self.spec.domain.characteristics.copy()
        return entity


class EntityRegistry:
    def __init__(self, spec_alphabet: SpecializedIPOAlphabet):
        self.spec = spec_alphabet
        self.expected_traits = spec_alphabet.get_all_traits()
        self.entities: Dict[str, Dict[str, Any]] = {}

    def add_entity(self, entity: Dict[str, Any]) -> None:
        self._validate(entity)
        self.entities[entity["id"]] = entity

    def _validate(self, entity: Dict[str, Any]) -> None:
        missing = self.expected_traits - set(entity.keys())
        if missing:
            raise ValueError(f"Сущность '{entity.get('id', 'unknown')}' отсутствует черты: {missing}")

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame.from_dict(self.entities, orient="index")
        if "domain_attrs" in df.columns:
            attrs = pd.json_normalize(df["domain_attrs"])
            df = pd.concat([df.drop(columns="domain_attrs"), attrs], axis=1)
        return df

    def __getitem__(self, entity_id: str) -> Dict[str, Any]:
        return self.entities[entity_id]