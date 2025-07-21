import logging
from src.hybrid_threat_detection.models.deep_learning_trainer import DeepLearningTrainer
from src.hybrid_threat_detection.utils.helpers import setup_logging

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting deep learning model training pipeline")
        
        # Initialize trainer with config
        trainer = DeepLearningTrainer(
            config_path="src/hybrid_threat_detection/config/dl_config.json"
        )
        
        # Train and evaluate models
        results = trainer.train_models()
        
        logger.info("Deep learning model training completed successfully!")
        
        # Print summary
        for dataset, models in results.items():
            logger.info(f"\nResults for {dataset}:")
            for model_name, model_info in models.items():
                logger.info(f"\n{model_name}:")
                logger.info(f"Metrics: {model_info['metrics']}")
                
    except Exception as e:
        logger.error(f"DL model training failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()