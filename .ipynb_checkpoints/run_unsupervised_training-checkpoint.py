import logging
from src.hybrid_threat_detection.models.unsupervised_trainer import UnsupervisedTrainer
from src.hybrid_threat_detection.utils.helpers import setup_logging

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting unsupervised model training pipeline")
        
        # Initialize trainer with config
        trainer = UnsupervisedTrainer(
            config_path="src/hybrid_threat_detection/config/unsupervised_config.json"
        )
        
        # Train and evaluate models
        results = trainer.train_models()
        
        logger.info("Unsupervised model training completed successfully!")
        
        # Print summary
        for dataset, models in results.items():
            logger.info(f"\nResults for {dataset}:")
            for model_name, model_info in models.items():
                logger.info(f"\n{model_name}:")
                logger.info(f"Metrics: {model_info['metrics']}")
                
    except Exception as e:
        logger.error(f"Unsupervised training failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()