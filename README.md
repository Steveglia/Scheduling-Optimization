# Study Schedule Optimization System

## Overview
This project implements an intelligent system for optimizing study schedules based on personal preferences. It uses multiple optimization algorithms to generate efficient study schedules that maximize preference satisfaction while respecting constraints like required study hours and maximum daily study time.

## Features
- Multiple optimization algorithms:
  - Genetic Algorithm
  - Dispersive Flies Optimization (DFO) with Best strategy
  - Dispersive Flies Optimization (DFO) with Neighbor strategy 
  - Island Model with multiple sub-populations
  - Hill Climber for local optimization
- Preference-based scheduling considering:
  - Time-slot preferences (rating from -1 to 9)
  - Required total study hours
  - Maximum daily study hours
- Comprehensive visualization:
  - Preference heatmaps
  - Schedule visualization
  - Algorithm comparison
  - Difference highlighting between algorithm solutions

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/study-schedule-optimization.git
cd study-schedule-optimization

# Install required dependencies
pip install numpy matplotlib seaborn tqdm
```

## Usage

### Basic Usage
To run the optimization system with default settings:

```bash
python main.py
```

This will:
1. Initialize the system with default preferences and constraints
2. Run all optimization algorithms
3. Display visualized results for each algorithm
4. Generate a comparison of all algorithms
5. Report the best performing algorithm

### Customizing Preferences

You can modify your schedule preferences in `study_schedule.py` or customize `preferences_generator.py` to create realistic preferences based on your availability and study preferences.

The preference matrix has dimensions of 5×12 (5 days × 12 hours) with values from -1 to 9:
- `-1`: Unavailable time slots
- `0-9`: Preference rating (higher values indicate greater preference)

## Configuration

Edit `config.py` to customize:

- `NUM_DAYS`: Number of days in the schedule (default: 5)
- `HOURS_PER_DAY`: Hours per day (default: 12, representing 8:00-20:00)
- `REQUIRED_STUDY_HOURS`: Total required study hours (default: 20)
- `MAX_DAILY_HOURS`: Maximum allowed study hours per day (default: 8)
- `POPULATION_SIZE`: Size of population for evolutionary algorithms (default: 100)
- `GENERATIONS`: Number of iterations for optimization (default: 90)
- `MUTATION_RATE`: Mutation probability for genetic algorithm (default: 0.08)
- `CROSSOVER_RATE`: Crossover probability for genetic algorithm (default: 0.1)
- `DELTA`: Disturbance threshold for DFO algorithm (default: 0.009)

## Algorithm Details

### Genetic Algorithm
Implements selection, crossover, and mutation operations to evolve a population of schedule solutions towards optimality.

### Dispersive Flies Optimization (DFO)
Nature-inspired algorithm modeling the behavior of flies:
- `DFO_Best`: Flies are attracted to the best global solution
- `DFO_Neighbour`: Flies are attracted to their neighboring flies

### Island Model
A parallel genetic algorithm implementation with multiple sub-populations (islands) that occasionally exchange individuals.

### Hill Climber Algorithm
A local search algorithm that iteratively improves a solution by exploring neighboring solutions.

## Visualization

The system provides rich visualization tools through `schedule_visualizer.py`:

- Preference heatmaps showing your time preferences
- Schedule heatmaps showing optimized study sessions
- Algorithm comparison visualizations
- Difference maps highlighting variations between algorithm solutions

## Example Output

When you run the program, you'll see:
1. Progress bars showing algorithm execution
2. Fitness scores for each algorithm
3. Visualizations of each algorithm's solution
4. Comparison between algorithms
5. The best overall algorithm and its fitness score

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests.

## License

[MIT License](LICENSE)
