import os
import pickle
import logging
from datetime import datetime
from typing import Dict, Tuple, Union
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from ..config.constants import RESULTS_DIR
from ..utils.helpers import create_dir_if_not_exists, save_json

logger = logging.getLogger(__name__)

class DataEncoder:
    """ data encoding and tagging for threat detection."""
    
    def __init__(self):
        self.encoders = {}
        self.metadata = {}
        self.encoding_stats = {}
        
        # Create output directory
        self.output_dir = os.path.join(RESULTS_DIR, "phase2", "encoding")
        create_dir_if_not_exists(self.output_dir)
    
    def encode_all(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Execute full encoding pipeline on all datasets."""
        encoded_data = {}
        
        for name, df in data.items():
            try:
                logger.info(f"Starting encoding for {name}")
                encoded_df, metadata = self._encode_dataset(name, df)
                encoded_data[name] = encoded_df
                self.metadata[name] = metadata
                logger.info(f"Completed encoding for {name}")
            except Exception as e:
                logger.error(f"Error encoding {name}: {str(e)}")
                raise
                
        self._save_artifacts()
        return encoded_data
    
    def _encode_dataset(self, name: str, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Execute encoding steps for a single dataset."""
        # Initialize metadata storage
        metadata = {
            'source_file': name,
            'original_columns': list(df.columns),
            'encoding_date': datetime.now().isoformat(),
            'sample_count': len(df)
        }
        
        # Make a copy to avoid modifying original data
        df_encoded = df.copy()
        
        # Add attack type as categorical feature
        df_encoded = self._add_attack_type(name, df_encoded, metadata)
        
        # Convert labels to numerical format
        df_encoded = self._encode_labels(name, df_encoded, metadata)
        
        # Store additional metadata
        self._store_metadata(df_encoded, metadata)
        
        return df_encoded, metadata
    
    def _add_attack_type(self, name: str, df: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        """Add attack type as categorical feature."""
        attack_type = self._map_dataset_to_attack_type(name)
        df['attack_type'] = attack_type
        metadata['attack_type'] = attack_type
        
        logger.info(f"Added attack type '{attack_type}' to {name}")
        return df
    
    def _map_dataset_to_attack_type(self, name: str) -> str:
        """Map dataset name to standardized attack type."""
        name_lower = name.lower()
        if 'xss' in name_lower:
            return 'XSS'
        elif 'sql' in name_lower:
            return 'SQLI'
        elif 'path' in name_lower:
            return 'PathTraversal'
        return 'Unknown'
    
    def _encode_labels(self, name: str, df: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        """Convert labels to numerical format (0: benign, 1: malicious)."""
        # Determine label column based on dataset
        label_col = self._identify_label_column(name, df)
        metadata['original_label_col'] = label_col
        
        if label_col not in df.columns:
            logger.warning(f"No label column found in {name}")
            df['label'] = -1  # Default for unlabeled data
            return df
            
        # Store original labels
        metadata['original_labels'] = df[label_col].value_counts().to_dict()
        
        # Create binary labels (1: malicious, 0: benign)
        df['label'] = self._convert_to_binary_labels(df[label_col])
        metadata['label_distribution'] = df['label'].value_counts().to_dict()
        
        logger.info(f"Encoded labels for {name}. Distribution:\n{df['label'].value_counts()}")
        return df
    
    def _identify_label_column(self, name: str, df: pd.DataFrame) -> str:
        """Identify the most likely label column based on dataset."""
        possible_labels = ['alert', 'event_type', 'type', 'prediction']
        for col in possible_labels:
            if col in df.columns:
                return col
        return ''
    
    def _convert_to_binary_labels(self, labels: pd.Series) -> pd.Series:
        """Convert various label formats to binary (0/1)."""
        # Handle case where labels are already numeric
        if pd.api.types.is_numeric_dtype(labels):
            return (labels > 0).astype(int)
        
        # Handle string labels
        labels_lower = labels.astype(str).str.lower()
        malicious_indicators = [
            'malicious', 'attack', 'xss', 'sql', 'injection', 
            'path', 'traversal', 'true', '1', 'positive'
        ]
        
        return labels_lower.apply(
            lambda x: 1 if any(indicator in x for indicator in malicious_indicators) else 0
        )
    
    def _store_metadata(self, df: pd.DataFrame, metadata: Dict) -> None:
        """Store additional metadata about the dataset."""
        metadata['final_columns'] = list(df.columns)
        metadata['label_col'] = 'label'
        metadata['attack_type_col'] = 'attack_type'
        
        # Store sample of data
        metadata['data_sample'] = df.head(2).to_dict(orient='records')
    
    def _save_artifacts(self) -> None:
        """Save all encoding artifacts."""
        # Save metadata
        metadata_path = os.path.join(self.output_dir, 'encoding_metadata.json')
        save_json(self.metadata, metadata_path)
        logger.info(f"Saved encoding metadata to {metadata_path}")
        
        # Save encoders if any were used
        if self.encoders:
            encoders_path = os.path.join(self.output_dir, 'encoders.pkl')
            with open(encoders_path, 'wb') as f:
                pickle.dump(self.encoders, f)
            logger.info(f"Saved encoders to {encoders_path}")

    def get_metadata(self) -> Dict:
        """Return the collected metadata."""
        return self.metadata