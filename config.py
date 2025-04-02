class StudyScheduleConfig:
    """Configuration class for study schedule parameters"""
    def __init__(self):
        
        self.NUM_DAYS = 5
        self.HOURS_PER_DAY = 12
        self.TOTAL_HOURS = self.NUM_DAYS * self.HOURS_PER_DAY
        self.REQUIRED_STUDY_HOURS = 20
        self.MAX_DAILY_HOURS = 8
     
        self.POPULATION_SIZE = 100
        self.GENERATIONS = 90
        self.MUTATION_RATE = 0.08
        self.CROSSOVER_RATE = 0.1
        self.DELTA = 0.009
        
        
