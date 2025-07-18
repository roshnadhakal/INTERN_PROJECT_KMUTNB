import logging
import os
import pandas as pd
from src.hybrid_threat_detection.features.splitting import DataSplitter
from src.hybrid_threat_detection.utils.helpers import setup_logging, load_json

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # 1. Load encoded data
        data = {
            "xss": pd.read_csv("src/hybrid_threat_detection/results/phase2/encoding/xss_encoded.csv"),
            "sql_injection": pd.read_csv("src/hybrid_threat_detection/results/phase2/encoding/sql_injection_encoded.csv"),
            "path_traversal": pd.read_csv("src/hybrid_threat_detection/results/phase2/encoding/path_traversal_encoded.csv")
        }
        
        # 2. Load config
        config = load_json("src/hybrid_threat_detection/config/splitting_config.json")
        
        # 3. Initialize and run splitter
        splitter = DataSplitter(config)
        split_data = splitter.split_all(data)
        
        # 4. Save results
        for name, splits in split_data.items():
            if 'folds' in splits:
                for fold in splits['folds']:
                    fold_dir = f"data/splits/{name}/fold_{fold['fold']}"
                    os.makedirs(fold_dir, exist_ok=True)
                    fold['train'].to_csv(f"{fold_dir}/train.csv", index=False)
                    fold['test'].to_csv(f"{fold_dir}/test.csv", index=False)
            else:
                os.makedirs(f"data/splits/{name}", exist_ok=True)
                splits['train'].to_csv(f"data/splits/{name}/train.csv", index=False)
                splits['test'].to_csv(f"data/splits/{name}/test.csv", index=False)
        
        # 5. Print summary
        metadata = splitter.get_metadata()
        logger.info("Splitting completed successfully!")
        logger.info("\nSplitting Summary:")
        for name, meta in metadata.items():
            logger.info(f"\nDataset: {name}")
            logger.info(f"Split type: {meta['split_type']}")
            if meta['split_type'] == 'cross_validation':
                logger.info(f"Folds: {meta['folds']}")
                logger.info(f"Samples per fold: ~{meta['fold_samples']}")
            else:
                logger.info(f"Train samples: {meta['train_samples']}")
                logger.info(f"Test samples: {meta['test_samples']}")
            logger.info(f"Train label distribution: {meta.get('train_label_dist')}")
            
    except Exception as e:
        logger.error(f"Splitting failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()