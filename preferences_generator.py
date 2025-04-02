# Generator for realistic study time preferences based on common patterns
import numpy as np
from typing import Tuple, List, Dict

class PreferencesGenerator:
    """Generates realistic preference values for study time slots"""
    
    def __init__(self, config):
        """Initialize generator with schedule configuration"""
        self.num_days = config.NUM_DAYS
        self.hours_per_day = config.HOURS_PER_DAY
        
        # Filter fixed commitments based on available days
        base_commitments = [
            # Lectures (Mon, Wed, Fri)
            (0, 3, 5, "lecture"),    # Monday 9AM-11AM
            (2, 3, 5, "lecture"),    # Wednesday 9AM-11AM
            (4, 3, 5, "lecture"),    # Friday 9AM-11AM
            
            # Work shifts (Tue, Thu)
            (1, 6, 10, "work"),      # Tuesday 12PM-4PM
            (3, 6, 10, "work"),      # Thursday 12PM-4PM
        ]
        
        # Only keep commitments that fall within the available days
        self.fixed_commitments = [
            commitment for commitment in base_commitments 
            if commitment[0] < self.num_days
        ]
        
        # Peak study hours
        self.peak_hours = [(3, 6), (8, 11)]  # 9AM-12PM and 2PM-5PM

    def create_realistic_preferences(self) -> np.ndarray:
        """Generate preference matrix with realistic patterns
        
        Returns:
            2D numpy array of preference values for each time slot
        """
        # Initialize grid with low preferences (0-3)
        preferences = np.random.randint(0, 4, size=(self.num_days, self.hours_per_day))
        
        # Mark unavailable times with -1 (impossible to study)
        for day, start, end, _ in self.fixed_commitments:
            preferences[day, start:end] = -1
        
        # Higher preferences for peak study hours (if not in fixed commitments)
        for day in range(self.num_days):
            for start, end in self.peak_hours:
                # Only set high preference if not a fixed commitment
                mask = preferences[day, start:end] != -1
                preferences[day, start:end][mask] = np.random.randint(7, 10, size=mask.sum())
        
        # Lower preferences for early morning and evening
        early_morning_mask = preferences[:, 0:2] != -1
        evening_mask = preferences[:, -2:] != -1
        preferences[:, 0:2][early_morning_mask] = np.random.randint(0, 3, size=early_morning_mask.sum())
        preferences[:, -2:][evening_mask] = np.random.randint(0, 3, size=evening_mask.sum())
        
        # Weekend adjustments (assuming days 5-6 are weekends)
        weekend_mask = preferences[5:, :] != -1
        preferences[5:, :][weekend_mask] = (preferences[5:, :][weekend_mask] * 1.2).astype(int)
        
        return preferences


