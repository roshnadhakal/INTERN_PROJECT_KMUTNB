import logging
from src.hybrid_threat_detection.models.supervised_trainer import SupervisedModelTrainer
from src.hybrid_threat_detection.utils.helpers import setup_logging

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting supervised model training pipeline")
        
        # Initialize trainer with config
        trainer = SupervisedModelTrainer(
            config_path="src/hybrid_threat_detection/config/model_config.json"
        )
        
        # Train and evaluate models
        results = trainer.train_models()
        
        logger.info("Model training completed successfully!")
        
        # Print summary
        for dataset, models in results.items():
            logger.info(f"\nResults for {dataset}:")
            for model_name, model_info in models.items():
                logger.info(f"\n{model_name}:")
                logger.info(f"Best params: {model_info['best_params']}")
                logger.info(f"Metrics: {model_info['metrics']}")
                
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()