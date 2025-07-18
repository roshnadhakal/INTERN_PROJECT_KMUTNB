import logging
from hybrid_threat_detection.pipelines.data_pipeline import DataPipeline
from hybrid_threat_detection.utils.helpers import setup_logging

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting Hybrid Threat Detection Pipeline")
        pipeline = DataPipeline()
        cleaned_data = pipeline.run_phase1()
        logger.info("Pipeline completed successfully")
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()