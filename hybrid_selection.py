import random
from typing import List, Tuple
from selection_strategy import SelectionStrategy

class HybridSelection(SelectionStrategy):
    """Hybrid selection strategy: best of two random for first parent, random for second."""
    
    def select_parents(self, population: List[List[int]], fitness_scores: List[float]) -> Tuple[List[int], List[int]]:
        """Select two parents using hybrid selection method."""
        # Select two random candidates for first parent
        candidates = random.sample(range(len(population)), 2)
        # Choose the better one as first parent
        parent1_idx = max(candidates, key=lambda i: fitness_scores[i])
        
        # Select second parent completely randomly
        parent2_idx = random.randint(0, len(population) - 1)
        
        return population[parent1_idx], population[parent2_idx]
    
    @property
    def name(self) -> str:
        return "Hybrid"
