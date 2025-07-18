import os
import logging
from typing import Dict, Tuple, Union
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from ..config.constants import RESULTS_DIR
from ..utils.helpers import create_dir_if_not_exists, save_json

logger = logging.getLogger(__name__)

class DataSplitter:
    """ data splitting for hybrid threat detection."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {
            'test_size': 0.2,
            'random_state': 42,
            'shuffle': True,
            'stratify': True,
            'cross_validation_folds': 5
        }
        self.split_metadata = {}
        
        # Create output directory
        self.output_dir = os.path.join(RESULTS_DIR, "phase2", "splits")
        create_dir_if_not_exists(self.output_dir)
    
    def split_all(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """Execute splitting pipeline on all datasets."""
        split_data = {}
        
        for name, df in data.items():
            try:
                logger.info(f"Splitting dataset: {name}")
                split_results = self._split_dataset(name, df)
                split_data[name] = split_results
                logger.info(f"Completed splitting for {name}")
            except Exception as e:
                logger.error(f"Error splitting {name}: {str(e)}")
                raise
                
        self._save_metadata()
        return split_data
    
    def _split_dataset(self, name: str, df: pd.DataFrame) -> Dict:
        """Split a single dataset into train/test or folds."""
        # Initialize metadata storage
        metadata = {
            'dataset': name,
            'samples_total': len(df),
            'split_config': self.config
        }
        
        # Verify we have labels
        if 'label' not in df.columns:
            logger.warning(f"No label column found in {name}, using simple split")
            return self._simple_split(df, metadata)
            
        # Get stratification column (attack_type + label for better distribution)
        stratify_col = self._create_stratification_column(df)
        
        # Perform the split
        if self.config.get('cross_validation_folds', 0) > 1:
            return self._cross_validation_split(df, stratify_col, metadata)
        else:
            return self._train_test_split(df, stratify_col, metadata)
    
    def _create_stratification_column(self, df: pd.DataFrame) -> np.ndarray:
        """Create combined stratification column."""
        if not self.config['stratify']:
            return None
            
        # Combine attack type and label for better stratification
        if 'attack_type' in df.columns:
            return (df['attack_type'] + '_' + df['label'].astype(str)).values
        return df['label'].values
    
    def _train_test_split(self, df: pd.DataFrame, 
                         stratify_col: np.ndarray,
                         metadata: Dict) -> Dict:
        """Perform stratified train-test split."""
        test_size = self.config['test_size']
        random_state = self.config['random_state']
        
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            shuffle=self.config['shuffle'],
            stratify=stratify_col
        )
        
        # Store metadata
        metadata.update({
            'split_type': 'train_test',
            'train_samples': len(train_df),
            'test_samples': len(test_df),
            'train_label_dist': train_df['label'].value_counts().to_dict(),
            'test_label_dist': test_df['label'].value_counts().to_dict()
        })
        
        if 'attack_type' in df.columns:
            metadata.update({
                'train_attack_dist': train_df['attack_type'].value_counts().to_dict(),
                'test_attack_dist': test_df['attack_type'].value_counts().to_dict()
            })
        
        logger.info(f"Train-test split completed. Train: {len(train_df)}, Test: {len(test_df)}")
        self.split_metadata[metadata['dataset']] = metadata
        
        return {
            'train': train_df,
            'test': test_df,
            'metadata': metadata
        }
    
    def _cross_validation_split(self, df: pd.DataFrame,
                              stratify_col: np.ndarray,
                              metadata: Dict) -> Dict:
        """Generate stratified K-Fold cross-validation splits."""
        n_splits = self.config['cross_validation_folds']
        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=self.config['shuffle'],
            random_state=self.config['random_state']
        )
        
        folds = []
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(df, stratify_col)):
            fold_data = {
                'train': df.iloc[train_idx],
                'test': df.iloc[test_idx],
                'fold': fold_idx
            }
            folds.append(fold_data)
        
        # Store metadata
        metadata.update({
            'split_type': 'cross_validation',
            'folds': n_splits,
            'fold_samples': len(df) // n_splits,
            'label_dist_per_fold': [
                fold['test']['label'].value_counts().to_dict()
                for fold in folds
            ]
        })
        
        if 'attack_type' in df.columns:
            metadata.update({
                'attack_dist_per_fold': [
                    fold['test']['attack_type'].value_counts().to_dict()
                    for fold in folds
                ]
            })
        
        logger.info(f"Generated {n_splits}-fold CV splits")
        self.split_metadata[metadata['dataset']] = metadata
        
        return {
            'folds': folds,
            'metadata': metadata
        }
    
    def _simple_split(self, df: pd.DataFrame, metadata: Dict) -> Dict:
        """Fallback split when no labels are available."""
        train_df, test_df = train_test_split(
            df,
            test_size=self.config['test_size'],
            random_state=self.config['random_state'],
            shuffle=self.config['shuffle']
        )
        
        metadata.update({
            'split_type': 'simple_train_test',
            'train_samples': len(train_df),
            'test_samples': len(test_df)
        })
        
        logger.warning("Performed simple split (no stratification)")
        self.split_metadata[metadata['dataset']] = metadata
        
        return {
            'train': train_df,
            'test': test_df,
            'metadata': metadata
        }
    
    def _save_metadata(self) -> None:
        """Save splitting metadata to file."""
        metadata_path = os.path.join(self.output_dir, 'split_metadata.json')
        save_json(self.split_metadata, metadata_path)
        logger.info(f"Saved splitting metadata to {metadata_path}")
        
        # Save sample splits for inspection
        sample_path = os.path.join(self.output_dir, 'split_samples.json')
        sample_data = {
            name: {
                'config': meta['split_config'],
                'stats': {
                    k: v for k, v in meta.items()
                    if k not in ['split_config', 'dataset']
                }
            }
            for name, meta in self.split_metadata.items()
        }
        save_json(sample_data, sample_path)

    def get_metadata(self) -> Dict:
        """Return splitting metadata."""
        return self.split_metadata