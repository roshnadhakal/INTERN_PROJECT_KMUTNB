import logging
from ..data_processing.explore_data import DataExplorer
from ..data_processing.clean_data import DataCleaner
from ..config.constants import RESULTS_DIR
from ..utils.helpers import create_dir_if_not_exists

logger = logging.getLogger(__name__)

class DataPipeline:
    """Orchestrates the complete data processing pipeline."""
    
    def __init__(self):
        self.explorer = DataExplorer()
        self.cleaner = DataCleaner()
    
    def run_phase1(self):
        """Execute all Phase 1 tasks (exploration and cleaning)."""
        logger.info("Starting Phase 1: Data Preparation")
        
        # Step 1.1: Data Exploration
        logger.info("Running 1.1: Dataset Understanding and Exploration")
        self.explorer.explore_all()
        
        # Step 1.2: Data Cleaning
        logger.info("Running 1.2: Data Cleaning")
        self.cleaner.load_data(self.explorer.data)
        cleaned_data = self.cleaner.clean_all_datasets()
        
        logger.info("Phase 1 completed successfully")
        return cleaned_data