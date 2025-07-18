import logging
import os
from ..features.text_vectorizer import TextVectorizer
from ..config.constants import CONFIG_DIR, RESULTS_DIR
from ..utils.helpers import load_json, save_json
from ..features.encoding import DataEncoder
from ..features.splitting import DataSplitter
from typing import Dict

logger = logging.getLogger(__name__)

class FeaturePipeline:
    """Orchestrates feature engineering pipeline."""
    
    def __init__(self):
        # Load vectorization config
        self.vectorization_config = load_json(os.path.join(CONFIG_DIR, 'vectorization_config.json'))
        self.splitter_config = load_json(os.path.join(CONFIG_DIR, 'splitting_config.json'))
        self.encoder = DataEncoder()
        self.splitter = DataSplitter(self.splitter_config)

    def run_phase2(self, cleaned_data: Dict) -> Dict:
        """Execute complete feature engineering and data tranformation phase."""
        logger.info("Starting Phase 2: Feature Engineering and data transformation.")
        
        vectorizer = TextVectorizer(self.vectorization_config)
        vectorized_data = vectorizer.vectorize(cleaned_data)

        # 2.2 Encoding and Tagging
        encoded_data = self.encoder.encode_all(cleaned_data)  # Use original cleaned data
        
        # 2.3 Train-Test Split
        split_data = self.splitter.split_all(encoded_data)

        # Save artifacts for each dataset
        for name, result in vectorized_data.items():
            vectorizer.save_vectorization_artifacts(dataset_name=name, results=result)
        
        encoded_data = self.run_encoding(cleaned_data, vectorized_data)
        
        logger.info("Feature engineering completed successfully")
        return {
            'vectorized': vectorized_data,
            'encoded': encoded_data,
            'split': split_data
        }
    

