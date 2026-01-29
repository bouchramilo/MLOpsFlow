# test_api.py
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np

from api.main import app
from api.schemas import DiabetesInput, DiabetesBatchInput


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def sample_input():
    """Sample input data for testing."""
    return {
        "Pregnancies": 6,
        "Glucose": 148.0,
        "BloodPressure": 72.0,
        "SkinThickness": 35.0,
        "Insulin": 0.0,
        "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627,
        "Age": 50
    }


@pytest.fixture
def sample_batch_input():
    """Sample batch input data for testing."""
    return {
        "instances": [
            {
                "Pregnancies": 6,
                "Glucose": 148.0,
                "BloodPressure": 72.0,
                "SkinThickness": 35.0,
                "Insulin": 0.0,
                "BMI": 33.6,
                "DiabetesPedigreeFunction": 0.627,
                "Age": 50
            },
            {
                "Pregnancies": 1,
                "Glucose": 85.0,
                "BloodPressure": 66.0,
                "SkinThickness": 29.0,
                "Insulin": 0.0,
                "BMI": 26.6,
                "DiabetesPedigreeFunction": 0.351,
                "Age": 31
            }
        ]
    }


class TestRootEndpoint:
    """Tests for the root endpoint."""
    
    def test_root_returns_welcome_message(self, client):
        """Test that root endpoint returns welcome message."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "docs" in data
        assert "health" in data


class TestHealthEndpoint:
    """Tests for the health check endpoint."""
    
    def test_health_endpoint_returns_status(self, client):
        """Test that health endpoint returns status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data


class TestModelInfoEndpoint:
    """Tests for the model info endpoint."""
    
    def test_model_info_returns_info(self, client):
        """Test that model info endpoint returns model information."""
        response = client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert "model_name" in data
        assert "is_loaded" in data


class TestPredictionEndpoint:
    """Tests for the prediction endpoint."""
    
    @patch('api.main.model_loader')
    def test_predict_success(self, mock_loader, client, sample_input):
        """Test successful prediction."""
        mock_loader.is_loaded.return_value = True
        mock_loader.predict.return_value = np.array([1])
        mock_loader.predict_proba.return_value = np.array([[0.3, 0.7]])
        
        response = client.post("/predict", json=sample_input)
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "status" in data
        assert data["status"] == "success"
    
    @patch('api.main.model_loader')
    def test_predict_model_not_loaded(self, mock_loader, client, sample_input):
        """Test prediction when model is not loaded."""
        mock_loader.is_loaded.return_value = False
        
        response = client.post("/predict", json=sample_input)
        assert response.status_code == 503
    
    def test_predict_invalid_input(self, client):
        """Test prediction with invalid input."""
        invalid_input = {
            "Pregnancies": -1,  
            "Glucose": 148.0,
            "BloodPressure": 72.0,
            "SkinThickness": 35.0,
            "Insulin": 0.0,
            "BMI": 33.6,
            "DiabetesPedigreeFunction": 0.627,
            "Age": 50
        }
        response = client.post("/predict", json=invalid_input)
        assert response.status_code == 422  
    
    def test_predict_missing_field(self, client):
        """Test prediction with missing field."""
        incomplete_input = {
            "Pregnancies": 6,
            "Glucose": 148.0,
            
        }
        response = client.post("/predict", json=incomplete_input)
        assert response.status_code == 422  


class TestBatchPredictionEndpoint:
    """Tests for the batch prediction endpoint."""
    
    @patch('api.main.model_loader')
    def test_batch_predict_success(self, mock_loader, client, sample_batch_input):
        """Test successful batch prediction."""
        
        mock_loader.is_loaded.return_value = True
        mock_loader.predict.return_value = np.array([1, 0])
        mock_loader.predict_proba.return_value = np.array([[0.3, 0.7], [0.8, 0.2]])
        
        response = client.post("/predict/batch", json=sample_batch_input)
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert "count" in data
        assert data["count"] == 2
        assert len(data["predictions"]) == 2
    
    @patch('api.main.model_loader')
    def test_batch_predict_model_not_loaded(self, mock_loader, client, sample_batch_input):
        """Test batch prediction when model is not loaded."""
        mock_loader.is_loaded.return_value = False
        
        response = client.post("/predict/batch", json=sample_batch_input)
        assert response.status_code == 503


class TestSchemaValidation:
    """Tests for Pydantic schema validation."""
    
    def test_diabetes_input_valid(self):
        """Test valid DiabetesInput creation."""
        input_data = DiabetesInput(
            Pregnancies=6,
            Glucose=148.0,
            BloodPressure=72.0,
            SkinThickness=35.0,
            Insulin=0.0,
            BMI=33.6,
            DiabetesPedigreeFunction=0.627,
            Age=50
        )
        assert input_data.Pregnancies == 6
        assert input_data.Glucose == 148.0
    
    def test_diabetes_input_invalid_pregnancies(self):
        """Test DiabetesInput with invalid pregnancies value."""
        with pytest.raises(ValueError):
            DiabetesInput(
                Pregnancies=-1,  
                Glucose=148.0,
                BloodPressure=72.0,
                SkinThickness=35.0,
                Insulin=0.0,
                BMI=33.6,
                DiabetesPedigreeFunction=0.627,
                Age=50
            )
    
    def test_diabetes_input_invalid_age(self):
        """Test DiabetesInput with invalid age value."""
        with pytest.raises(ValueError):
            DiabetesInput(
                Pregnancies=6,
                Glucose=148.0,
                BloodPressure=72.0,
                SkinThickness=35.0,
                Insulin=0.0,
                BMI=33.6,
                DiabetesPedigreeFunction=0.627,
                Age=0  
            )
    
    def test_batch_input_valid(self):
        """Test valid DiabetesBatchInput creation."""
        batch_input = DiabetesBatchInput(
            instances=[
                DiabetesInput(
                    Pregnancies=6,
                    Glucose=148.0,
                    BloodPressure=72.0,
                    SkinThickness=35.0,
                    Insulin=0.0,
                    BMI=33.6,
                    DiabetesPedigreeFunction=0.627,
                    Age=50
                )
            ]
        )
        assert len(batch_input.instances) == 1