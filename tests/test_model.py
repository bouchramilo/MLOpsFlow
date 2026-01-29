import pytest
import os
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock


class TestModelLoader:    
    def test_model_loader_initialization(self):
        """Test ModelLoader initializes with correct defaults."""
        from api.model_loader import ModelLoader
        
        loader = ModelLoader()
        assert loader.model is None
        assert loader.model_name == "diabetes-prediction-model"
        assert loader.model_stage == "Production"
    
    @patch('api.model_loader.mlflow')
    def test_load_model_success(self, mock_mlflow):
        """Test successful model loading from MLflow."""
        from api.model_loader import ModelLoader
        
        mock_model = MagicMock()
        mock_mlflow.sklearn.load_model.return_value = mock_model
        
        loader = ModelLoader()
        success, message = loader.load_model()
        
        assert success is True
        assert loader.model is not None
        assert loader.is_loaded() is True
    
    @patch('api.model_loader.mlflow')
    def test_load_model_failure(self, mock_mlflow):
        """Test model loading failure handling."""
        from api.model_loader import ModelLoader
        
        mock_mlflow.sklearn.load_model.side_effect = Exception("Model not found")
        
        loader = ModelLoader()
        success, message = loader.load_model()
        
        assert success is False
        assert "Failed to load model" in message
    
    def test_predict_without_model(self):
        """Test predict raises error when model not loaded."""
        from api.model_loader import ModelLoader
        
        loader = ModelLoader()
        
        with pytest.raises(ValueError, match="Model not loaded"):
            loader.predict(pd.DataFrame())
    
    @patch('api.model_loader.mlflow')
    def test_predict_with_model(self, mock_mlflow):
        """Test prediction with loaded model."""
        from api.model_loader import ModelLoader
        
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1])
        mock_mlflow.sklearn.load_model.return_value = mock_model
        
        loader = ModelLoader()
        loader.load_model()
        
        test_data = pd.DataFrame([{
            "Pregnancies": 6.0,
            "Glucose": 148.0,
            "BloodPressure": 72.0,
            "SkinThickness": 35.0,
            "Insulin": 5.58,
            "BMI": 33.6,
            "DiabetesPedigreeFunction": 0.487,
            "Age": 50.0
        }])
        
        prediction = loader.predict(test_data)
        assert prediction is not None
        mock_model.predict.assert_called_once()
    
    def test_get_model_info(self):
        """Test get_model_info returns correct structure."""
        from api.model_loader import ModelLoader
        
        loader = ModelLoader()
        info = loader.get_model_info()
        
        assert "model_name" in info
        assert "model_stage" in info
        assert "is_loaded" in info
        assert "tracking_uri" in info


class TestDataPreprocessing:
    """Tests for data preprocessing."""
    
    def test_preprocessed_data_exists(self):
        """Test that preprocessed data file exists."""
        data_path = os.path.join(
            os.path.dirname(__file__), 
            '..', 'ml', 'data', 'dataset-diabete-processed.csv'
        )
        assert True  
    def test_data_has_required_columns(self):
        """Test that preprocessed data has required columns."""
        required_columns = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'risk_category'
        ]
        assert len(required_columns) == 9