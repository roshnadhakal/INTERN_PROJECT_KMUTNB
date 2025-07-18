import logging
import pandas as pd
from hybrid_threat_detection.features.text_vectorizer import TextVectorizer
from hybrid_threat_detection.utils.helpers import setup_logging, load_json

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # 1. Load sample data 
        data = {
            "xss": pd.read_csv("hybrid-threat-detection/src/hybrid_threat_detection/results/phase1/cleaning/xss_cleaned_sample.csv"),
            "sql_injection": pd.read_csv("hybrid-threat-detection/src/hybrid_threat_detection/results/phase1/cleaning/sql_injection_cleaned_sample.csv"),
            "path_traversal": pd.read_csv("hybrid-threat-detection/src/hybrid_threat_detection/results/phase1/cleaning/path_traversal_cleaned_sample.csv")
        }
        
        # 2. Load config
        config = load_json("hybrid-threat-detection/src/hybrid_threat_detection/config/vectorization_config.json")
        
        # 3. Initialize and run vectorizer
        vectorizer = TextVectorizer(config)
        vectorized_data = vectorizer.vectorize(data)
        
        # 4. Save results
        logger.info("Vectorization completed successfully!")
        
        # access to results:
        xss_tfidf = vectorized_data["xss"]["tfidf"]["matrix"]
        xss_sequences = vectorized_data["xss"]["padded_sequences"]
        
        logger.info(f"XSS TF-IDF shape: {xss_tfidf.shape}")
        logger.info(f"XSS Padded Sequences shape: {xss_sequences.shape}")
        
    except Exception as e:
        logger.error(f"Vectorization failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()