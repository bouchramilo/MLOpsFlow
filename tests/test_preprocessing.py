import pandas as pd
import numpy as np

from ml.preprossessing import (
    knn_imputation,
    preparation_final_model,
    process_data
)

# =============================
# Tests for knn_imputation
# =============================

def test_knn_imputation_no_nan():
    df = pd.DataFrame({
        "Glucose": [0, 120, 130],
        "BloodPressure": [70, 0, 80],
        "SkinThickness": [0, 25, 30],
        "BMI": [0, 28, 35],
        "Insulin": [0, 100, 130],
        "Age": [25, 45, 60]
    })

    result = knn_imputation(df)

    assert not result.isnull().any().any()


def test_knn_imputation_shape_preserved():
    df = pd.DataFrame(
        np.random.rand(10, 6),
        columns=["Glucose", "BloodPressure", "SkinThickness", "BMI", "Insulin", "Age"]
    )

    result = knn_imputation(df)

    assert result.shape == df.shape


def test_knn_imputation_columns_preserved():
    df = pd.DataFrame(
        np.random.rand(5, 6),
        columns=["Glucose", "BloodPressure", "SkinThickness", "BMI", "Insulin", "Age"]
    )

    result = knn_imputation(df)

    assert list(result.columns) == list(df.columns)



# =============================
# Tests for preparation_final_model
# =============================

def test_preparation_final_model_no_nan_or_inf():
    df = pd.DataFrame({
        "Insulin": [10, 50, 200],
        "DiabetesPedigreeFunction": [0.1, 0.5, 2.0],
        "Pregnancies": [0, 3, 10],
        "Glucose": [110, 130, 150],
        "BMI": [25, 35, 40]
    })

    result = preparation_final_model(df)

    assert not np.isinf(result.values).any()
    assert not result.isnull().any().any()


def test_preparation_final_model_columns_preserved():
    df = pd.DataFrame({
        "Insulin": [10, 20],
        "DiabetesPedigreeFunction": [0.2, 0.4],
        "Pregnancies": [1, 2]
    })

    result = preparation_final_model(df)

    assert set(result.columns) == set(df.columns)



# =============================
# Tests for process_data
# =============================

def test_process_data_pipeline(monkeypatch, tmp_path):

    fake_df = pd.DataFrame({
        "Glucose": [0, 140],
        "BloodPressure": [80, 0],
        "SkinThickness": [0, 30],
        "BMI": [0, 35],
        "Insulin": [0, 120],
        "DiabetesPedigreeFunction": [0.3, 0.8],
        "Pregnancies": [1, 2]
    })

    def mock_read_csv(*args, **kwargs):
        return fake_df

    monkeypatch.setattr(pd, "read_csv", mock_read_csv)

    process_data()
