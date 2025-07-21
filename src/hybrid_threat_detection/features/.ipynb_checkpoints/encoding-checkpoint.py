import os
import pickle
import logging
import re
import numpy as np
from datetime import datetime
from typing import Dict, Tuple, Union
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from ..config.constants import RESULTS_DIR
from ..utils.helpers import create_dir_if_not_exists, save_json

logger = logging.getLogger(__name__)

class DataEncoder:
    """Enhanced data encoding and tagging for threat detection with cross-dataset support."""
    
    def __init__(self):
        self.encoders = {}
        self.metadata = {}
        self.encoding_stats = {}
        
        self.output_dir = os.path.join(RESULTS_DIR, "phase2", "encoding")
        create_dir_if_not_exists(self.output_dir)
        
        self.patterns = {
            'XSS': [
                r'<script[^>]*>', r'javascript:', r'onerror=', r'onload=', 
                r'eval\(', r'alert\(', r'svg\'onload', r'document\.cookie',
                r'window\.location', r'fromCharCode\(',
                        # New patterns to catch more variants
                r'<img[^>]*src=[\'"]?javascript:',  # IMG tag XSS
                r'<iframe[^>]*src=',                # IFrame injection
                r'onmouseover=',                     # Mouse event handlers  
                r'style=[\'"]?expression\(',         # CSS expression()
                r'<link[^>]*href=[\'"]?javascript:', # LINK tag XSS
                r'<meta[^>]*http-equiv=[\'"]?refresh', # Meta refresh
                r'<object[^>]*data=',               # Object tag
                r'<form[^>]*action=[\'"]?javascript:' # Form action
            ],
            'SQLI': [
                r'union\s+select', r'select\s+.+\s+from', r'insert\s+into',
                r'update\s+.+\s+set', r'delete\s+from', r'drop\s+table',
                r'1=1', r'--\s*$', r'\/\*.*\*\/', r'waitfor\s+delay'
            ],
            'PathTraversal': [
                r'\.\./', r'\.\.\\', r'\/etc\/passwd', r'\/proc\/self',
                r'C:\\', r'%2e%2e%2f', r'\.\.%2f', r'\.\.%5c'
            ]
        }

    def encode_all(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        encoded_data = {}
        
        for name, df in data.items():
            try:
                logger.info(f"Starting encoding for {name}")
                encoded_df, metadata = self._encode_dataset(name, df)
                encoded_data[name] = encoded_df
                self.metadata[name] = self._convert_metadata_for_json(metadata)
                logger.info(f"Completed encoding for {name}")
            except Exception as e:
                logger.error(f"Error encoding {name}: {str(e)}")
                raise
                
        self._save_artifacts()
        return encoded_data

    def _encode_dataset(self, name: str, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        metadata = {
            'source_file': name,
            'original_columns': list(df.columns),
            'encoding_date': datetime.now().isoformat(),
            'sample_count': int(len(df))
        }

        df_encoded = df.copy()
        df_encoded = self._add_attack_type(name, df_encoded, metadata)
        df_encoded = self._detect_malicious_content(name, df_encoded, metadata)
        df_encoded = self._encode_labels(name, df_encoded, metadata)

        # ⚠️ Add class count warning
        if len(df_encoded['label'].unique()) == 1:
            logger.warning(f"WARNING: Only one class found in {name} labels!")

        self._store_metadata(df_encoded, metadata)
        return df_encoded, metadata

    def _add_attack_type(self, name: str, df: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        attack_type = self._map_dataset_to_attack_type(name)
        df['attack_type'] = attack_type
        metadata['attack_type'] = attack_type
        logger.info(f"Added attack type '{attack_type}' to {name}")
        return df

    def _map_dataset_to_attack_type(self, name: str) -> str:
        name_lower = name.lower()
        if any(x in name_lower for x in ['xss', 'cross-site']):
            return 'XSS'
        elif any(x in name_lower for x in ['sql', 'injection']):
            return 'SQLI'
        elif any(x in name_lower for x in ['path', 'traversal', 'directory']):
            return 'PathTraversal'
        return 'Unknown'

    def _detect_malicious_content(self, name: str, df: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        attack_type = self._map_dataset_to_attack_type(name)
        text_columns = self._identify_text_columns(df)

        df['malicious_detected'] = 0
        metadata['detection_stats'] = {'scanned_columns': text_columns}

        if attack_type == 'Unknown':
            return df
        
        patterns = [re.compile(p, re.IGNORECASE) for p in self.patterns[attack_type]]

        for col in text_columns:
            col_malicious = df[col].astype(str).apply(
                lambda x: any(p.search(x) for p in patterns)
            )
            df['malicious_detected'] |= col_malicious.astype(int)

        metadata['detection_stats']['malicious_count'] = int(df['malicious_detected'].sum())
        logger.info(f"Detected {df['malicious_detected'].sum()} malicious samples in {name}")
        return df

    def _identify_text_columns(self, df: pd.DataFrame) -> list:
        text_cols = []
        potential_cols = ['headers', 'http', 'post_data', 'path', 
                          'cookies', 'uri', 'url', 'request', 'response']
        for col in df.columns:
            col_lower = col.lower()
            if (any(x in col_lower for x in potential_cols)) or \
               (df[col].dtype == 'object' and df[col].astype(str).str.len().mean() > 10):
                text_cols.append(col)
        return text_cols

    def _encode_labels(self, name: str, df: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        # Existing label logic
        if 'malicious_detected' in df.columns:
            df['label'] = df['malicious_detected']
            metadata['label_source'] = 'pattern_detection'
        elif 'attack_type' in df.columns:
            df['label'] = (df['attack_type'] != 'Unknown').astype(int)
            metadata['label_source'] = 'attack_type'
        else:
            df['label'] = 0
            metadata['label_source'] = 'default_benign'
    
        min_malicious = 10  # Increased from 5 to ensure CV works
        if name.lower() == 'xss' and sum(df['label']) < min_malicious:
            needed = min_malicious - sum(df['label'])
            logger.warning(f"Only {sum(df['label'])} XSS samples found - augmenting with top {needed} candidates")
            
            # Score samples by pattern matches
            xss_scores = pd.Series(0, index=df.index)
            patterns = [re.compile(p, re.IGNORECASE) for p in self.patterns['XSS']]
            
            for col in self._identify_text_columns(df):
                xss_scores += df[col].astype(str).apply(
                    lambda x: sum(1 for p in patterns if p.search(x))
                )
            
            # Augment dataset
            candidates = xss_scores.nlargest(needed + 5).index  # Buffer
            df = pd.concat([df, df.loc[candidates].assign(label=1)], ignore_index=True)
            
            metadata['augmented_samples'] = int(needed)
            logger.info(f"After augmentation: {sum(df['label'])} malicious samples")
        logger.info(f"Final labels for {name}:\n{df['label'].value_counts()}")
        return df
        
    def _store_metadata(self, df: pd.DataFrame, metadata: Dict) -> None:
        metadata.update({
            'final_columns': list(df.columns),
            'label_col': 'label',
            'attack_type_col': 'attack_type',
            'malicious_detected_col': 'malicious_detected' if 'malicious_detected' in df.columns else None,
            'data_sample': df.head(2).astype(str).to_dict(orient='records')
        })

    def _convert_metadata_for_json(self, metadata: Dict) -> Dict:
        converted = {}
        for key, value in metadata.items():
            if isinstance(value, (np.integer, np.int64)):
                converted[key] = int(value)
            elif isinstance(value, (np.floating, np.float64)):
                converted[key] = float(value)
            elif isinstance(value, np.ndarray):
                converted[key] = value.tolist()
            elif isinstance(value, dict):
                converted[key] = self._convert_metadata_for_json(value)
            elif isinstance(value, list):
                converted[key] = [self._convert_metadata_for_json(item) if isinstance(item, dict) else item for item in value]
            else:
                converted[key] = value
        return converted

    def _save_artifacts(self) -> None:
        metadata_path = os.path.join(self.output_dir, 'encoding_metadata.json')
        save_json({
            'metadata': self.metadata,
            'detection_patterns': self.patterns
        }, metadata_path)

        if self.encoders:
            encoders_path = os.path.join(self.output_dir, 'encoders.pkl')
            with open(encoders_path, 'wb') as f:
                pickle.dump(self.encoders, f)

    def get_metadata(self) -> Dict:
        return {
            'metadata': self.metadata,
            'detection_patterns': self.patterns
        }
