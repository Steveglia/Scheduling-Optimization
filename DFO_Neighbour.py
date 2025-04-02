from typing import List
import numpy as np
from scipy.stats import qmc
from config import StudyScheduleConfig
from fitness_evaluator import FitnessEvaluator
from hill_climber_algorithm import HillClimberAlgorithm

class DFO_Neighbour:
    def __init__(self, config: StudyScheduleConfig, fitness_evaluator: FitnessEvaluator):
        """Initialize DFO with neighbor strategy"""
        self.config = config
        self.fitness_evaluator = fitness_evaluator
        
        # Algorithm parameters
        self.population_size = config.POPULATION_SIZE
        self.solution_size = config.TOTAL_HOURS
        self.delta = config.DELTA  # Use delta from config
        self.lower_bound = 0  # Binary solution space
        self.upper_bound = 1
        self.hill_climbing_rate = 0.3  # Apply hill climbing to 30% of population
        
    def initialize_population(self) -> List[List[int]]:
        """Initialize population using Latin Hypercube Sampling"""
        population = []
        
        # Get valid indices (where preference > 0)
        valid_indices = [i for i in range(self.solution_size) 
                        if self.fitness_evaluator.preferences[i] > 0]
        
        num_valid_slots = len(valid_indices)
        
        # Use Latin Hypercube Sampling for high dimensions
        sampler = qmc.LatinHypercube(d=num_valid_slots)
        lhs_samples = sampler.random(n=self.population_size)
        
        for i in range(self.population_size):
            # Create a solution with all zeros
            solution = [0] * self.solution_size
            
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
    
    def find_best_neighbor(self, population: List[List[int]], index: int) -> List[int]:
        """Find the best neighbor in a ring topology."""
        population_size = len(population)
        if population_size <= 1:
            return population[0]
        
        # Get indices of neighbors in ring topology
        left_idx = (index - 1) % population_size
        right_idx = (index + 1) % population_size
        
        # Get fitness of neighbors
        left_fitness = self.fitness_evaluator.calculate_fitness(population[left_idx])
        right_fitness = self.fitness_evaluator.calculate_fitness(population[right_idx])
        
        # Return the better neighbor
        if left_fitness > right_fitness:
            return population[left_idx]
        return population[right_idx]

    def optimize_population(self, population: List[List[int]]) -> List[List[int]]:
        """Optimize population using DFO with neighbor strategy."""
        new_population = []
        population_size = len(population)
        
        # Apply hill climbing to a portion of the population
        if self.hill_climbing_rate > 0:
            hill_climber = HillClimberAlgorithm(self.config, self.fitness_evaluator)
            num_hill_climb = int(population_size * self.hill_climbing_rate)
            
            # Sort population by fitness for hill climbing
            population_with_fitness = [
                (solution, self.fitness_evaluator.calculate_fitness(solution))
                for solution in population
            ]
            population_with_fitness.sort(key=lambda x: x[1], reverse=True)
            
            # Apply hill climbing to top solutions
            for i in range(min(num_hill_climb, len(population_with_fitness))):
                solution = population_with_fitness[i][0]
                improved_solution = hill_climber.optimize(solution.copy())
                new_population.append(improved_solution)
        
        # Process remaining solutions
        remaining_solutions = population[len(new_population):]
        for i, solution in enumerate(remaining_solutions):
            if np.random.random() < self.delta:
                # Random restart
                new_solution = [
                    1 if np.random.random() >= 0.5 else 0
                    for _ in range(self.solution_size)
                ]
            else:
                # Update position based on best neighbor
                best_neighbor = self.find_best_neighbor(population, i + len(new_population))
                
                # Generate new solution
                new_solution = []
                for j in range(self.solution_size):
                    if np.random.random() < 0.5:
                        new_solution.append(solution[j])
                    else:
                        new_solution.append(best_neighbor[j])
            
            new_population.append(new_solution)
        
        return new_population

    def run(self):
        """Run the DFO optimization process and return solution with fitness"""
        population = self.initialize_population()
        for _ in range(self.config.GENERATIONS):
            population = self.optimize_population(population)
        
        # Find best solution in final population
        best_solution = max(population, key=self.fitness_evaluator.calculate_fitness)
        fitness = self.fitness_evaluator.calculate_fitness(best_solution)
        return best_solution, fitness
