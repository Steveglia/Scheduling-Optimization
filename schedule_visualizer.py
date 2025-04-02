# Visualization module for creating schedule heatmaps and algorithm comparisons
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Optional

class ScheduleVisualizer:
    """Handles visualization of schedules and algorithm comparisons"""
    
    def __init__(self, config, preferences):
        """Initialize visualizer with schedule configuration and preferences"""
        self.config = config
        self.preferences = preferences

    def plot_schedule(self, solution: List[int], comparison_solutions: Optional[Dict[str, List[int]]] = None, algorithm_name: str = "Study Schedule"):
        """Create heatmap visualization of a schedule solution
        
        Args:
            solution: Binary list of selected time slots
            comparison_solutions: Optional solutions to compare against
            algorithm_name: Name of algorithm that generated the solution
        """
        # Reshape the solution and preferences into 5x12 grids
        schedule_grid = np.array(solution).reshape(self.config.NUM_DAYS, self.config.HOURS_PER_DAY)
        preferences_grid = np.array(self.preferences)
        
        if comparison_solutions:
            # Create a figure with three subplots side by side
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(25, 6))
            
            # Plot preferences heatmap on the left
            sns.heatmap(preferences_grid,
                       cmap='YlOrRd',
                       linewidths=1,
                       linecolor='black',
                       xticklabels=[f'{i}:00' for i in range(8, 20)],
                       yticklabels=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
                       ax=ax1)
            
            # Add preference values as text in each cell
            for i in range(self.config.NUM_DAYS):
                for j in range(self.config.HOURS_PER_DAY):
                    ax1.text(j + 0.5, i + 0.5, f'{preferences_grid[i, j]:.1f}',
                            ha='center', va='center',
                            color='black')
            
            ax1.set_title('Preference Values\n(Darker = Higher Preference)')
            ax1.set_xlabel('Hour of Day')
            ax1.set_ylabel('Day of Week')
            
            # Plot schedule heatmap in the middle
            sns.heatmap(schedule_grid, 
                       cmap=['white', 'lightgreen'],
                       cbar=False,
                       linewidths=1,
                       linecolor='black',
                       xticklabels=[f'{i}:00' for i in range(8, 20)],
                       yticklabels=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
                       ax=ax2)
            
            # Add preference values as text in each cell
            for i in range(self.config.NUM_DAYS):
                for j in range(self.config.HOURS_PER_DAY):
                    text_color = 'black' if schedule_grid[i, j] == 1 else 'darkgray'
                    ax2.text(j + 0.5, i + 0.5, f'{preferences_grid[i, j]:.1f}',
                            ha='center', va='center',
                            color=text_color)
            
            ax2.set_title(f'{algorithm_name}\n(Green blocks are study sessions)')
            ax2.set_xlabel('Hour of Day')
            ax2.set_ylabel('Day of Week')
            
            # Create a difference heatmap on the right
            solutions_list = list(comparison_solutions.items())
            if solutions_list:
                solution_name, comparison_solution = solutions_list[0]  # Take the first comparison solution
                comparison_grid = np.array(comparison_solution).reshape(self.config.NUM_DAYS, self.config.HOURS_PER_DAY)
                
                # Create difference grid: -1 = only in primary, 1 = only in comparison, 0 = same
                difference_grid = np.zeros_like(schedule_grid)
                difference_grid[schedule_grid == 1] = -1
                difference_grid[comparison_grid == 1] = 1
                difference_grid[(schedule_grid == comparison_grid) & (schedule_grid == 1)] = 0
                
                # Custom colormap for differences
                colors = ['red', 'white', 'blue']
                sns.heatmap(difference_grid,
                           cmap=colors,
                           center=0,
                           vmin=-1,
                           vmax=1,
                           cbar=False,
                           linewidths=1,
                           linecolor='black',
                           xticklabels=[f'{i}:00' for i in range(8, 20)],
                           yticklabels=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
                           ax=ax3)
                
                # Add preference values as text in each cell
                for i in range(self.config.NUM_DAYS):
                    for j in range(self.config.HOURS_PER_DAY):
                        text_color = 'white' if difference_grid[i, j] != 0 else 'black'
                        ax3.text(j + 0.5, i + 0.5, f'{preferences_grid[i, j]:.1f}',
                                ha='center', va='center',
                                color=text_color)
                
                ax3.set_title(f'Difference Map\nBlue = Only in {solution_name}\nRed = Only in {algorithm_name}\nWhite = Same/No Session')
                ax3.set_xlabel('Hour of Day')
                ax3.set_ylabel('Day of Week')
        
        else:
            # Original two-plot visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
            
            # Plot preferences heatmap on the left
            sns.heatmap(preferences_grid,
                       cmap='YlOrRd',
                       linewidths=1,
                       linecolor='black',
                       xticklabels=[f'{i}:00' for i in range(8, 20)],
                       yticklabels=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
                       ax=ax1)
            
            # Add preference values as text in each cell
            for i in range(self.config.NUM_DAYS):
                for j in range(self.config.HOURS_PER_DAY):
                    ax1.text(j + 0.5, i + 0.5, f'{preferences_grid[i, j]:.1f}',
                            ha='center', va='center',
                            color='black')
            
            ax1.set_title('Preference Values\n(Darker = Higher Preference)')
            ax1.set_xlabel('Hour of Day')
            ax1.set_ylabel('Day of Week')
            
            # Plot schedule heatmap on the right
            sns.heatmap(schedule_grid, 
                       cmap=['white', 'lightgreen'],
                       cbar=False,
                       linewidths=1,
                       linecolor='black',
                       xticklabels=[f'{i}:00' for i in range(8, 20)],
                       yticklabels=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
                       ax=ax2)
            
            # Add preference values as text in each cell
            for i in range(self.config.NUM_DAYS):
                for j in range(self.config.HOURS_PER_DAY):
                    text_color = 'black' if schedule_grid[i, j] == 1 else 'darkgray'
                    ax2.text(j + 0.5, i + 0.5, f'{preferences_grid[i, j]:.1f}',
                            ha='center', va='center',
                            color=text_color)
            
            ax2.set_title(f'{algorithm_name}\n(Green blocks are study sessions)')
            ax2.set_xlabel('Hour of Day')
            ax2.set_ylabel('Day of Week')
        
        plt.suptitle(f'{algorithm_name} Optimization Results', fontsize=16, y=1.05)
        plt.tight_layout()
        plt.show()

    def plot_algorithm_differences(self, solutions: Dict[str, List[int]], fitness_scores: Dict[str, float]):
        """Create heatmap showing agreement between different algorithms
        
        Args:
            solutions: Dictionary of algorithm solutions
            fitness_scores: Dictionary of algorithm fitness scores
        """
        # Count how many algorithms chose each time slot
        frequency_grid = np.zeros((self.config.NUM_DAYS, self.config.HOURS_PER_DAY))
        for solution in solutions.values():
            solution_grid = np.array(solution).reshape(self.config.NUM_DAYS, self.config.HOURS_PER_DAY)
            frequency_grid += solution_grid
            
        # Create the figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot the heatmap
        sns.heatmap(frequency_grid,
                   cmap='YlOrRd',  # Yellow to Orange to Red colormap
                   vmin=0,
                   vmax=len(solutions),
                   linewidths=1,
                   linecolor='black',
                   xticklabels=[f'{i}:00' for i in range(8, 20)],
                   yticklabels=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
                   ax=ax)
        
        # Add text showing number of algorithms and preference value
        preferences_grid = np.array(self.preferences)
        for day in range(self.config.NUM_DAYS):
            for hour in range(self.config.HOURS_PER_DAY):
                if frequency_grid[day, hour] > 0:
                    text = f'{int(frequency_grid[day, hour])}\n({preferences_grid[day, hour]:.1f})'
                    ax.text(hour + 0.5, day + 0.5, text,
                           ha='center', va='center',
                           color='black' if frequency_grid[day, hour] < len(solutions)*0.7 else 'white',
                           fontweight='bold')
        
        # Add title with algorithm fitness scores
        title = 'Algorithm Agreement Heatmap\n'
        title += 'Number of algorithms choosing each slot (Preference value)\n\n'
        title += 'Algorithm Fitness Scores:\n'
        for algo, fitness in fitness_scores.items():
            title += f'{algo}: {fitness:.2f}  '
            
        ax.set_title(title)
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Day of Week')
        
        plt.tight_layout()
        plt.show()