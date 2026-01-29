import os
import mlflow
import mlflow.pyfunc
import mlflow.sklearn
from typing import Optional, Tuple, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelLoader:
    
    def __init__(self):
        self.model: Optional[Any] = None
        self.sklearn_model: Optional[Any] = None 
        self.model_version: Optional[str] = None
        self.model_name: str = os.getenv("MLFLOW_MODEL_NAME", "diabetes-prediction-model")
        self.model_stage: str = os.getenv("MLFLOW_MODEL_STAGE", "Production")
        self.mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        
    def load_model(self) -> Tuple[bool, str]:

        try:
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            logger.info(f"MLflow tracking URI set to: {self.mlflow_tracking_uri}")
            
            model_uri = f"models:/{self.model_name}/{self.model_stage}"
            logger.info(f"Attempting to load model from: {model_uri}")
            
            try:
                self.sklearn_model = mlflow.sklearn.load_model(model_uri)
                self.model = self.sklearn_model
                self.model_version = self.model_stage
                logger.info(f"Successfully loaded model '{self.model_name}' from stage '{self.model_stage}'")
                return True, f"Model loaded successfully from {model_uri}"
            except Exception as e:
                logger.warning(f"Could not load model from stage '{self.model_stage}': {e}")
                
                try:
                    model_uri = f"models:/{self.model_name}/latest"
                    self.sklearn_model = mlflow.sklearn.load_model(model_uri)
                    self.model = self.sklearn_model
                    self.model_version = "latest"
                    logger.info(f"Successfully loaded latest version of model '{self.model_name}'")
                    return True, f"Model loaded successfully from {model_uri}"
                except Exception as e2:
                    logger.warning(f"Could not load latest model: {e2}")
                    
                    local_model_path = os.getenv("LOCAL_MODEL_PATH")
                    if local_model_path and os.path.exists(local_model_path):
                        self.sklearn_model = mlflow.sklearn.load_model(local_model_path)
                        self.model = self.sklearn_model
                        self.model_version = "local"
                        logger.info(f"Successfully loaded model from local path: {local_model_path}")
                        return True, f"Model loaded from local path: {local_model_path}"
                    
                    return False, f"Failed to load model: {e2}"
                    
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False, f"Error loading model: {str(e)}"
    
    def predict(self, data) -> Any:
        if self.model is None:
            raise ValueError("Model not loaded. Please load the model first.")
        
        return self.model.predict(data)
    
    def predict_proba(self, data) -> Optional[Any]:
        if self.model is None:
            raise ValueError("Model not loaded. Please load the model first.")
        
        try:
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(data)
            
            if hasattr(self.model, 'named_steps'):
                for step_name, step in self.model.named_steps.items():
                    if hasattr(step, 'predict_proba'):
                        return self.model.predict_proba(data)
                
        except Exception as e:
            logger.warning(f"Could not get prediction probabilities: {e}")
        
        return None
    
    def is_loaded(self) -> bool:
        return self.model is not None
    
    def get_model_info(self) -> dict:
        return {
            "model_name": self.model_name,
            "model_stage": self.model_stage,
            "model_version": self.model_version,
            "is_loaded": self.is_loaded(),
            "tracking_uri": self.mlflow_tracking_uri
        }


model_loader = ModelLoader()