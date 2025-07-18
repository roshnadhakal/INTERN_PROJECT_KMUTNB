import os
import re
import logging
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
from ..config.constants import DATA_FILES, COLUMN_MAPPINGS, RESULTS_DIR
from ..utils.helpers import create_dir_if_not_exists, setup_logging

# Set up logging
logger = logging.getLogger(__name__)
setup_logging()

class DataCleaner:
    """Production-grade data cleaning pipeline for web attack datasets."""
    
    def __init__(self):
        self.data = {}
        self.cleaning_stats = {}
        self.smote = SMOTE(random_state=42)
        
    def load_data(self, data_dict: Dict[str, pd.DataFrame] = None):
        """Load data from either provided dict or configured files."""
        if data_dict:
            self.data = data_dict
        else:
            logger.info("Loading datasets from configured files...")
            for name, path in DATA_FILES.items():
                try:
                    self.data[name] = pd.read_csv(path)
                    logger.info(f"Loaded {name} with shape {self.data[name].shape}")
                except Exception as e:
                    logger.error(f"Error loading {name}: {str(e)}")
                    raise
    
    def clean_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """Execute full cleaning pipeline on all datasets."""
        cleaned_data = {}
        for name, df in self.data.items():
            try:
                logger.info(f"Starting cleaning for {name}")
                cleaned_df = self._clean_dataset(name, df)
                cleaned_data[name] = cleaned_df
                logger.info(f"Completed cleaning for {name}. Final shape: {cleaned_df.shape}")
            except Exception as e:
                logger.error(f"Error cleaning {name}: {str(e)}")
                raise
                
        return cleaned_data
    
    def _clean_dataset(self, name: str, df: pd.DataFrame) -> pd.DataFrame:
        """Execute cleaning steps for a single dataset."""
        # Initialize stats tracking
        self.cleaning_stats[name] = {
            'initial_rows': len(df),
            'null_rows_removed': 0,
            'duplicates_removed': 0,
            'final_rows': 0
        }
        
        # Step 1: Handle null values
        df_clean = self._handle_nulls(name, df)
        
        # Step 2: Remove duplicates
        df_clean = self._remove_duplicates(name, df_clean)
        
        # Step 3: Normalize payloads
        df_clean = self._normalize_payloads(name, df_clean)
        
        # Step 4: Handle class imbalance
        df_clean = self._balance_classes(name, df_clean)
        
        # Update stats
        self.cleaning_stats[name]['final_rows'] = len(df_clean)
        self._save_cleaning_stats(name)
        
        return df_clean
    
    def _handle_nulls(self, name: str, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with null values in critical columns."""
        mapping = COLUMN_MAPPINGS.get(name, {})
        critical_cols = ['payload_col', 'label_col']
        
        # Identify which columns to check based on mapping
        cols_to_check = []
        for col in critical_cols:
            if mapping.get(col) and mapping[col] in df.columns:
                cols_to_check.append(mapping[col])
        
        if not cols_to_check:
            logger.warning(f"No critical columns found for null checking in {name}")
            return df
            
        # Count nulls before removal
        initial_count = len(df)
        null_counts = df[cols_to_check].isnull().sum()
        logger.info(f"Null counts before cleaning in {name}:\n{null_counts}")
        
        # Remove rows with nulls in any critical column
        df_clean = df.dropna(subset=cols_to_check)
        
        # Update stats
        rows_removed = initial_count - len(df_clean)
        self.cleaning_stats[name]['null_rows_removed'] = rows_removed
        logger.info(f"Removed {rows_removed} rows with null values in {name}")
        
        return df_clean
    
    def _remove_duplicates(self, name: str, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate records within the dataset."""
        # Identify columns to use for duplicate detection
        mapping = COLUMN_MAPPINGS.get(name, {})
        dup_cols = []
        
        for col in ['payload_col', 'source_ip', 'timestamp']:
            if mapping.get(col) and mapping[col] in df.columns:
                dup_cols.append(mapping[col])
        
        if not dup_cols:
            logger.warning(f"No columns configured for duplicate detection in {name}")
            return df
            
        # Count duplicates before removal
        initial_count = len(df)
        duplicates = df.duplicated(subset=dup_cols, keep='first')
        duplicate_count = duplicates.sum()
        logger.info(f"Found {duplicate_count} duplicates in {name}")
        
        # Remove duplicates
        df_clean = df[~duplicates]
        
        # Update stats
        self.cleaning_stats[name]['duplicates_removed'] = duplicate_count
        logger.info(f"Removed {duplicate_count} duplicates from {name}")
        
        return df_clean
    
    def _normalize_payloads(self, name: str, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize payload text using multiple cleaning steps."""
        mapping = COLUMN_MAPPINGS.get(name, {})
        payload_col = mapping.get('payload_col')
        
        if not payload_col or payload_col not in df.columns:
            logger.warning(f"No payload column found in {name}")
            return df
            
        logger.info(f"Normalizing payloads in {name}...")
        
        # Make a copy to avoid SettingWithCopyWarning
        df_clean = df.copy()
        
        # Step 1: Convert to string and handle NaN
        df_clean[payload_col] = df_clean[payload_col].astype(str)
        
        # Step 2: Convert to lowercase
        df_clean[payload_col] = df_clean[payload_col].str.lower()
        
        # Step 3: Remove excessive whitespace
        df_clean[payload_col] = df_clean[payload_col].str.strip()
        df_clean[payload_col] = df_clean[payload_col].str.replace(r'\s+', ' ', regex=True)
        
        # Step 4: Remove special characters (keep alphanumeric and basic punctuation)
        df_clean[payload_col] = df_clean[payload_col].apply(
            lambda x: re.sub(r'[^\w\s\-\.\,\!\?]', '', x)
        )
        
        # Step 5: Optional HTML stripping for XSS
        if 'xss' in name.lower():
            df_clean[payload_col] = df_clean[payload_col].apply(
                lambda x: BeautifulSoup(x, 'html.parser').get_text()
            )
        
        logger.info(f"Payload normalization completed for {name}")
        return df_clean
    
    def _balance_classes(self, name: str, df: pd.DataFrame) -> pd.DataFrame:
        """Handle class imbalance using SMOTE or other methods."""
        mapping = COLUMN_MAPPINGS.get(name, {})
        label_col = mapping.get('label_col')
        
        if not label_col or label_col not in df.columns:
            logger.warning(f"No label column found for balancing in {name}")
            return df
            
        # Get class distribution
        class_counts = df[label_col].value_counts()
        logger.info(f"Class distribution before balancing in {name}:\n{class_counts}")
        
        # Skip balancing if only one class exists
        if len(class_counts) < 2:
            logger.warning(f"Only one class found in {name}, skipping balancing")
            return df
            
        # Determine minority and majority classes
        majority_class = class_counts.idxmax()
        minority_class = class_counts.idxmin()
        imbalance_ratio = class_counts[majority_class] / class_counts[minority_class]
        
        logger.info(f"Imbalance ratio in {name}: {imbalance_ratio:.1f}:1")
        
        # Apply SMOTE if imbalance is significant (> 2:1)
        if imbalance_ratio > 2:
            try:
                logger.info(f"Applying SMOTE to {name}...")
                
                # Prepare data for SMOTE
                X = df[payload_col].values.reshape(-1, 1)  # Using payload as feature
                y = df[label_col]
                
                # Apply SMOTE
                X_res, y_res = self.smote.fit_resample(X, y)
                
                # Create new balanced DataFrame
                balanced_df = pd.DataFrame({
                    payload_col: X_res.flatten(),
                    label_col: y_res
                })
                
                # Add back other columns if needed
                for col in df.columns:
                    if col not in [payload_col, label_col]:
                        balanced_df[col] = np.nan  # Or use appropriate default
                
                logger.info(f"SMOTE applied to {name}. New shape: {balanced_df.shape}")
                return balanced_df
                
            except Exception as e:
                logger.error(f"SMOTE failed for {name}: {str(e)}")
                logger.info("Falling back to random oversampling...")
                return self._random_oversample(df, label_col)
        else:
            logger.info(f"Imbalance ratio acceptable in {name}, skipping balancing")
            return df
    
    def _random_oversample(self, df: pd.DataFrame, label_col: str) -> pd.DataFrame:
        """Fallback method for class balancing."""
        # Separate majority and minority classes
        majority_class = df[label_col].value_counts().idxmax()
        minority_class = df[label_col].value_counts().idxmin()
        
        df_majority = df[df[label_col] == majority_class]
        df_minority = df[df[label_col] == minority_class]
        
        # Oversample minority class
        df_minority_oversampled = resample(
            df_minority,
            replace=True,
            n_samples=len(df_majority),
            random_state=42
        )
        
        # Combine back
        balanced_df = pd.concat([df_majority, df_minority_oversampled])
        logger.info(f"Random oversampling completed. New shape: {balanced_df.shape}")
        
        return balanced_df
    
    def _save_cleaning_stats(self, name: str):
        """Save cleaning statistics to file."""
        output_dir = os.path.join(RESULTS_DIR, "phase1", "cleaning")
        create_dir_if_not_exists(output_dir)
        
        # Save stats to JSON
        stats_path = os.path.join(output_dir, f"{name}_cleaning_stats.json")
        pd.DataFrame.from_dict(self.cleaning_stats[name], orient='index').to_json(stats_path)
        logger.info(f"Saved cleaning stats for {name} to {stats_path}")
        
        # Save sample of cleaned data
        sample_path = os.path.join(output_dir, f"{name}_cleaned_sample.csv")
        self.data[name].head(100).to_csv(sample_path, index=False)
        logger.info(f"Saved cleaned sample for {name} to {sample_path}")

    def get_cleaning_stats(self) -> Dict[str, Dict]:
        """Return cleaning statistics for all datasets."""
        return self.cleaning_stats