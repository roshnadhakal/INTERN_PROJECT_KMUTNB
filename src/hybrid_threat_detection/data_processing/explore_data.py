import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ..config.constants import DATA_FILES, COLUMN_MAPPINGS, RESULTS_DIR
from ..utils.helpers import create_dir_if_not_exists
class DataExplorer:
    def __init__(self):
        self.data = {}
        self.summaries = {}
        
    def load_data(self):
        """Load all datasets into memory"""
        print("Loading datasets...")
        for name, path in DATA_FILES.items():
            try:
                self.data[name] = pd.read_csv(path)
                print(f"Successfully loaded {name} with shape {self.data[name].shape}")
            except Exception as e:
                print(f"Error loading {name}: {str(e)}")
    
    def analyze_dataset(self, dataset_name):
        """Perform basic analysis on a dataset"""
        if dataset_name not in self.data:
            print(f"Dataset {dataset_name} not loaded")
            return None
            
        df = self.data[dataset_name]
        mapping = COLUMN_MAPPINGS.get(dataset_name, {})
        
        # Basic info
        print(f"\nAnalysis for {dataset_name}:")
        print("="*50)
        print(f"Shape: {df.shape}")
        print("\nFirst 5 rows:")
        print(df.head())
        print("\nColumns:")
        print(df.columns)
        print("\nData types:")
        print(df.dtypes)
        print("\nMissing values per column:")
        print(df.isnull().sum())
        
        # Check critical columns
        critical_cols = ['payload_col', 'label_col', 'source_ip', 'timestamp']
        for col in critical_cols:
            mapped_col = mapping.get(col)
            if mapped_col and mapped_col in df.columns:
                print(f"\n{col} mapped to {mapped_col}:")
                if col == 'label_col':
                    self._analyze_labels(df, mapped_col, dataset_name)
                else:
                    print(df[mapped_col].describe())
            else:
                print(f"\nWarning: {col} not found in dataset {dataset_name}")
        
        return df
    
    def _analyze_labels(self, df, label_col, dataset_name):
        """Analyze label distribution"""
        if label_col not in df.columns:
            print(f"Label column {label_col} not found")
            return
            
        print("\nLabel distribution:")
        label_counts = df[label_col].value_counts()
        print(label_counts)
        
        # Visualize
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x=label_col)
        plt.title(f"Label Distribution for {dataset_name}")
        plt.xticks(rotation=45)
        
        # Save plot
        output_dir = os.path.join(RESULTS_DIR, "phase1", "exploration")
        create_dir_if_not_exists(output_dir)
        plot_path = os.path.join(output_dir, f"{dataset_name}_label_distribution.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved label distribution plot to {plot_path}")
        
        # Save summary
        self.summaries[dataset_name] = {
            'label_counts': label_counts.to_dict(),
            'plot_path': plot_path
        }
    
    def explore_all(self):
        """Explore all datasets"""
        self.load_data()
        for name in self.data.keys():
            self.analyze_dataset(name)
        
        # Save combined summary
        self.save_summary()
    
    def save_summary(self):
        """Save exploration summary"""
        output_dir = os.path.join(RESULTS_DIR, "phase1", "exploration")
        create_dir_if_not_exists(output_dir)
        
        summary_path = os.path.join(output_dir, "data_exploration_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("Data Exploration Summary\n")
            f.write("="*50 + "\n\n")
            
            for name, summary in self.summaries.items():
                f.write(f"Dataset: {name}\n")
                f.write("-"*50 + "\n")
                f.write("Label Counts:\n")
                for label, count in summary['label_counts'].items():
                    f.write(f"{label}: {count}\n")
                f.write(f"\nVisualization saved to: {summary['plot_path']}\n\n")
        
        print(f"Saved exploration summary to {summary_path}")

if __name__ == "__main__":
    explorer = DataExplorer()
    explorer.explore_all()