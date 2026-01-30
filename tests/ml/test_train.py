import pytest
import sys
import pandas as pd
from unittest.mock import MagicMock, patch
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.modules["mlflow"] = MagicMock()
sys.modules["mlflow.sklearn"] = MagicMock()
sys.modules["mlflow.tracking"] = MagicMock()
from ml import train as train_module

class TestTrainSimple:
    
    @patch('ml.train.pd.read_csv')
    @patch('ml.train.os.path.exists')
    @patch('ml.train.train_test_split')
    @patch('ml.train.RandomOverSampler')
    @patch('ml.train.GridSearchCV')
    @patch('ml.train.joblib.dump')
    def test_train_function_execution(self, mock_dump, mock_grid_search, mock_sampler, mock_split, mock_exists, mock_read_csv):
        """
        Test the train() function by mocking all external interactions.
        This verifies the logic flow without running actual heavy workloads.
        """
        
        # 1. Mock Data Loading
        mock_exists.return_value = True
        
        df = pd.DataFrame({
            'Pregnancies': [1, 2],
            'Glucose': [100, 110],
            'BloodPressure': [70, 75],
            'SkinThickness': [20, 22],
            'Insulin': [50, 60],
            'BMI': [25, 28],
            'DiabetesPedigreeFunction': [0.5, 0.6],
            'Age': [30, 40],
            'risk_category': [0, 1]
        })
        mock_read_csv.return_value = df
        
        # 2. Mock Split
        import numpy as np
        
        X_mock = np.array([[1, 2], [3, 4]])
        y_test_mock = np.array([0, 1]) 
        mock_split.return_value = (X_mock, X_mock, X_mock, y_test_mock)
        
        # 3. Mock Sampler
        mock_sampler_instance = MagicMock()
        mock_sampler.return_value = mock_sampler_instance

        mock_sampler_instance.fit_resample.return_value = (np.array([[1, 2], [3, 4]]), np.array([0, 1]))
        
        # 4. Mock Grid Search
        mock_grid_instance = MagicMock()
        mock_grid_search.return_value = mock_grid_instance
        
        # Mock best_estimator_ and best_params_
        best_model_mock = MagicMock()
        mock_grid_instance.best_estimator_ = best_model_mock
        mock_grid_instance.best_params_ = {}
        
        # Mock predictions
        best_model_mock.predict.return_value = np.array([0, 1]) # Mock predictions
        best_model_mock.predict_proba.return_value = np.array([[0.9, 0.1], [0.2, 0.8]]) # Mock proba
                
        # 5. Execute the function
        train_module.train()
        
        # 6. Assertions
        mock_read_csv.assert_called_once()
        mock_grid_instance.fit.assert_called_once()
        best_model_mock.predict.assert_called()
        
        train_module.mlflow.start_run.assert_called()
        train_module.mlflow.log_metric.assert_called()
        train_module.mlflow.sklearn.log_model.assert_called()
        
        # Verify Pipeline saving
        mock_dump.assert_called_once()
        
    @patch('ml.train.os.path.exists')
    def test_train_missing_data(self, mock_exists):
        """Test that train raises error if data file is missing."""
        mock_exists.return_value = False
        
        with pytest.raises(FileNotFoundError):
            train_module.train()

