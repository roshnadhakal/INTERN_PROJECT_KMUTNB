
import logging
import pandas as pd
from src.hybrid_threat_detection.features.encoding import DataEncoder
from src.hybrid_threat_detection.utils.helpers import setup_logging

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # 1. Load cleaned data (replace with your actual paths)
        cleaned_data = {
            "xss": pd.read_csv("src/hybrid_threat_detection/results/phase1/cleaning/xss_cleaned_sample.csv"),
            "sql_injection": pd.read_csv("src/hybrid_threat_detection/results/phase1/cleaning/sql_injection_cleaned_sample.csv"),
            "path_traversal": pd.read_csv("src/hybrid_threat_detection/results/phase1/cleaning/path_traversal_cleaned_sample.csv")}
        # 2. Initialize and run encoder
        encoder = DataEncoder()
        encoded_data = encoder.encode_all(cleaned_data)
        
        # 3. Save results
        for name, df in encoded_data.items():
            output_path = f"src/hybrid_threat_detection/results/phase2/encoding/{name}_encoded.csv"
            df.to_csv(output_path, index=False)
            logger.info(f"Saved encoded {name} data to {output_path}")
        
        # 4. Print summary
        metadata = encoder.get_metadata()
        logger.info("Encoding completed successfully!")
        logger.info("\nEncoding Summary:")
        for name, meta in metadata.items():
            logger.info(f"\nDataset: {name}")
            logger.info(f"Attack type: {meta.get('attack_type')}")
            logger.info(f"Label distribution: {meta.get('label_distribution')}")
            logger.info(f"Original label column: {meta.get('original_label_col')}")
            
    except Exception as e:
        logger.error(f"Encoding failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()