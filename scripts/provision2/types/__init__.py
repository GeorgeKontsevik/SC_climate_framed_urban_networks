from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class ProvisionResult:
    provision: float
    demand: float
    capacity_left: float = 0.0

@dataclass
class FlowAssignment:
    source: int
    target: int
    amount: float
    path: List[int]