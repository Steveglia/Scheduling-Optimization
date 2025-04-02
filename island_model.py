# Implementation of Island Model optimization with multiple populations
import random
import numpy as np
from typing import List, Dict, Any
from genetic_algorithm import GeneticAlgorithm
from DFO_Best import DFO_Best
from DFO_Neighbour import DFO_Neighbour
from hill_climber_algorithm import HillClimberAlgorithm

class IslandModel:
    """Implements Island Model optimization with multiple sub-populations"""
    
    def __init__(self, config, fitness_evaluator, hill_climber):
        """Initialize Island Model with configuration and helper components"""
        self.config = config
        self.fitness_evaluator = fitness_evaluator
        self.hill_climber = hill_climber
        
        # Initialize algorithms for each island
        self.algorithms = {
            'GA': GeneticAlgorithm(config, fitness_evaluator, hill_climber),
            'DFO_Best': DFO_Best(config, fitness_evaluator),
            'DFO_Neighbour': DFO_Neighbour(config, fitness_evaluator)
        }
        
        # Initialize populations for each island
        self.populations = {name: [] for name in self.algorithms.keys()}
        
        # Migration parameters
        self.migration_interval = 20 # Migrate every 5 generations
        self.migration_size = 2      # Number of solutions to migrate
        
    def _initialize_populations(self):
        """Initialize populations for each island using their respective algorithms"""
        for name, algorithm in self.algorithms.items():
            self.populations[name] = algorithm.initialize_population()
    
    def _perform_migration(self):
        """Perform migration between islands"""
        # Get all island names
        islands = list(self.populations.keys())
        
        # For each island
        for i, source_island in enumerate(islands):
            # Get target island (next island in the list, wrap around to first)
            target_island = islands[(i + 1) % len(islands)]
            
            # Get fitness scores for source population
            source_pop = self.populations[source_island]
            fitness_scores = [self.fitness_evaluator.calculate_fitness(ind) for ind in source_pop]
            
            # Sort population by fitness
            pop_fitness = list(zip(source_pop, fitness_scores))
            pop_fitness.sort(key=lambda x: x[1], reverse=True)
            
            # Select best solutions to migrate
            migrants = [ind.copy() for ind, _ in pop_fitness[:self.migration_size]]
            
            # Replace worst solutions in target population
            target_pop = self.populations[target_island]
            target_fitness = [self.fitness_evaluator.calculate_fitness(ind) for ind in target_pop]
            
            # Sort target population by fitness
            target_pop_fitness = list(zip(target_pop, target_fitness))
            target_pop_fitness.sort(key=lambda x: x[1])
            
            # Replace worst solutions with migrants
            for i in range(self.migration_size):
                if i < len(target_pop):
                    target_pop[i] = migrants[i]
    
    def optimize(self, generations=50) -> Dict[str, Any]:
        """Run the island model optimization process"""
        # Initialize populations
        self._initialize_populations()
        
        best_solution = None
        best_fitness = float('-inf')
        fitness_history = []
        
        # Main optimization loop
        for generation in range(generations):
            # Optimize each island
            for name, algorithm in self.algorithms.items():
                population = self.populations[name]
                
                # Get fitness scores
                fitness_scores = [self.fitness_evaluator.calculate_fitness(ind) for ind in population]
                max_fitness = max(fitness_scores)
                
                # Update best solution if better found
                if max_fitness > best_fitness:
                    best_fitness = max_fitness
                    best_solution = population[fitness_scores.index(max_fitness)].copy()
                
                # Optimize population using island's algorithm
                if isinstance(algorithm, GeneticAlgorithm):
                    # For GA, we need to run one generation
                    new_population = []
                    elite_size = max(1, int(len(population) * algorithm.elite_ratio))
                    
                    # Sort population by fitness
                    pop_fitness = list(zip(population, fitness_scores))
                    pop_fitness.sort(key=lambda x: x[1], reverse=True)
                    
                    # Keep elite solutions
                    new_population = [ind for ind, _ in pop_fitness[:elite_size]]
                    
                    # Generate rest of population
                    while len(new_population) < len(population):
                        parent1, parent2 = algorithm.selection_strategy.select_parents(population, fitness_scores)
                        child = algorithm.crossover(parent1, parent2)
                        child = algorithm.mutate(child)
                        if random.random() < algorithm.local_search_rate:
                            child = algorithm.hill_climber.optimize(child)
                        new_population.append(child)
                    
                    self.populations[name] = new_population
                else:
                    # For DFO algorithms, use their optimize_population method
                    self.populations[name] = algorithm.optimize_population(population)
            
            # Perform migration if needed
            if generation > 0 and generation % self.migration_interval == 0:
                self._perform_migration()
            
            # Record best fitness
            fitness_history.append({
                'best': best_fitness,
                'avg': np.mean([self.fitness_evaluator.calculate_fitness(ind) 
                              for pop in self.populations.values() 
                              for ind in pop])
            })
        
        return {
            'best_solution': best_solution,
            'best_fitness': best_fitness,
            'fitness_history': fitness_history
        }

    def run(self):
        """Run the island model optimization"""
        # Run optimization and get results
        results = self.optimize(self.config.GENERATIONS)
        
        # Return best solution and its fitness
        return results['best_solution'], results['best_fitness']

def main():
    """Run the island model optimization"""
    from config import StudyScheduleConfig
    from preferences_generator import PreferencesGenerator
    from fitness_evaluator import FitnessEvaluator
    
    # Setup
    config = StudyScheduleConfig()
    preferences = PreferencesGenerator(config).create_realistic_preferences()
    fitness_evaluator = FitnessEvaluator(config, preferences)
    hill_climber = HillClimberAlgorithm(config, fitness_evaluator)
    
    # Run optimization
    print("Starting Island Model Optimization")
    print("=" * 40)
    print("Algorithms:")
    print("1. DFO with Best Strategy")
    print("2. DFO with Neighbour Strategy")
    print("3. GA with Low Mutation (3%)")
    print("=" * 40)
    
    island_model = IslandModel(config, fitness_evaluator, hill_climber)
    results = island_model.run()
    
    # Print results
    print("\nOptimization Complete")
    print("=" * 40)
    print(f"Best Fitness: {results[1]:.2f}")
    print("=" * 40)

if __name__ == "__main__":
    main()
