import random
from typing import List, Tuple
from selection_strategy import SelectionStrategy

class TournamentSelection(SelectionStrategy):
    """Tournament selection strategy with fixed tournament size of 4."""
    
    def __init__(self, tournament_size: int = 4):
        self._tournament_size = tournament_size
    
    def select_parents(self, population: List[List[int]], fitness_scores: List[float]) -> Tuple[List[int], List[int]]:
        """Select two parents using tournament selection."""
        # First parent tournament
        candidates1 = random.sample(range(len(population)), self._tournament_size)
        parent1_idx = max(candidates1, key=lambda i: fitness_scores[i])
        
        # Second parent tournament
        candidates2 = random.sample(range(len(population)), self._tournament_size)
        parent2_idx = max(candidates2, key=lambda i: fitness_scores[i])
        
        return population[parent1_idx], population[parent2_idx]
    
    @property
    def name(self) -> str:
        return f"Tournament (k={self._tournament_size})"
