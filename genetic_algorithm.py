import random
import numpy as np
from typing import List, Tuple
from roulette_selection import RouletteSelection
from hill_climber_algorithm import HillClimberAlgorithm
from scipy import stats as qmc

class GeneticAlgorithm:
    """Main genetic algorithm implementation with hill climbing optimization"""
    def __init__(self, config, fitness_evaluator, hill_climber=None, elite_ratio=0.05):
        self.config = config
        self.fitness_evaluator = fitness_evaluator
        self.selection_strategy = RouletteSelection()
        self.hill_climber = hill_climber if hill_climber else HillClimberAlgorithm(config, fitness_evaluator)
        self.local_search_rate = 0.7
        self.elite_ratio = elite_ratio

    def initialize_population(self) -> List[List[int]]:
        population = []
        # Get valid indices (where preference > 0)
        valid_indices = [i for i in range(self.config.TOTAL_HOURS) 
                        if self.fitness_evaluator.preferences[i] > 0]
        
        num_valid_slots = len(valid_indices)
        
        # Use Latin Hypercube Sampling for high dimensions
        sampler = qmc.qmc.LatinHypercube(d=num_valid_slots)
        lhs_samples = sampler.random(n=self.config.POPULATION_SIZE)
        
        for i in range(self.config.POPULATION_SIZE):
            # Create a solution with all zeros
            solution = [0] * self.config.TOTAL_HOURS
            
            # Convert LHS samples to probabilities for this individual
            probabilities = lhs_samples[i]
            # Sort indices by their probabilities
            sorted_indices = [valid_indices[j] for j in np.argsort(probabilities)]
            
            # Select the first REQUIRED_STUDY_HOURS indices
            selected_indices = sorted_indices[:self.config.REQUIRED_STUDY_HOURS]
            
            # Set selected indices to 1
            for idx in selected_indices:
                solution[idx] = 1
            
            population.append(solution)
            
        return population

    def crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """Simple single-point crossover without any modifications."""
        if random.random() < self.config.CROSSOVER_RATE:
            point = random.randint(1, self.config.TOTAL_HOURS - 2)
            child = parent1[:point] + parent2[point:]
            return child
        return parent1.copy()

    def mutate(self, solution: List[int]) -> List[int]:
        """Simple bit-flip mutation without any modifications."""
        if random.random() < self.config.MUTATION_RATE:
            mutated = solution.copy()
            # Calculate number of positions to mutate
            num_positions = max(1, int(len(solution) * self.config.MUTATION_RATE))
            # Select random positions to mutate
            positions_to_mutate = random.sample(range(len(solution)), num_positions)
            for pos in positions_to_mutate:
                mutated[pos] = 1 - mutated[pos]  # Flip the bit
            return mutated
        return solution.copy()

    def run(self) -> Tuple[List[int], float]:
        population = self.initialize_population()
        best_solution = None
        best_fitness = float('-inf')
        
        elite_size = max(1, int(self.config.POPULATION_SIZE * self.elite_ratio))
        
        for generation in range(self.config.GENERATIONS):
            # Evaluate fitness for all individuals
            fitness_scores = [self.fitness_evaluator.calculate_fitness(individual) 
                            for individual in population]
            
            # Sort population by fitness
            population_fitness = list(zip(population, fitness_scores))
            population_fitness.sort(key=lambda x: x[1], reverse=True)
            
            # Update best solution if found
            if population_fitness[0][1] > best_fitness:
                best_fitness = population_fitness[0][1]
                best_solution = population_fitness[0][0]
            
            # Keep elite solutions
            elite_solutions = [ind for ind, _ in population_fitness[:elite_size]]
            
            # Select parents for next generation
            new_population = elite_solutions.copy()
            
            while len(new_population) < self.config.POPULATION_SIZE:
                # Select parents using the current selection strategy
                parent1, parent2 = self.selection_strategy.select_parents(population, fitness_scores)
                
                # Crossover
                if random.random() < self.config.CROSSOVER_RATE:
                    child = self.crossover(parent1, parent2)
                else:
                # If crossover is not applied, just copy the parent
                    child = parent1.copy()
                
                # Apply mutation
                if random.random() < self.config.MUTATION_RATE:
                    child = self.mutate(child)
                
                # Apply hill climbing optimization based on local search rate
                if random.random() < self.local_search_rate:
                    child = self.hill_climber.optimize(child)
                
                new_population.append(child)
            
            population = new_population
        
        
        return best_solution, best_fitness
