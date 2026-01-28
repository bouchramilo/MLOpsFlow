# train.py
import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
import joblib


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'dataset-diabete-processed.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model.pkl')

def train():
    
    # LOAD data
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Processing data not found at {DATA_PATH}, run preprocessing.py first.")
    
    data = pd.read_csv(DATA_PATH, index_col=0)
    
    # target and features
    X = data.drop(columns=['risk_category'])
    y = data[['risk_category']]
    
    print(f"Spliting data : ")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # gérer l'équiliprage 
    
    sampler = RandomOverSampler(random_state=42)
    X_train_resampled, y_train_resampled = sampler.fit_resample(X_train, y_train)
    
    mlflow.set_experiment("Diabetes_Prediction")
    
    with mlflow.start_run():
        # Hyperparams de modele
        n_estimators = 100
        min_samples_split = 10
        min_samples_leaf = 2
        max_features = "sqrt"
        max_depth = 5
        
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("min_samples_split", min_samples_split)
        mlflow.log_param("min_samples_leaf", min_samples_leaf)
        mlflow.log_param("max_features", max_features)
        mlflow.log_param("max_depth", max_depth)
        
        # pipline de entrainement
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestClassifier(
                n_estimators=n_estimators,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                max_depth=max_depth                
            ))
        ])
        
        
        # entrainenmt pipeline
        pipeline.fit(X_train_resampled, y_train_resampled)
        
        
        # Evaluation de model
        y_pred = pipeline.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Recall : {recall:.4f}")
        print(f"f1_score : {f1:.4f}")
        
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("Recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        mlflow.sklearn.log_model(pipeline, "model")
        
        print(f"Saving pipeline : ")
        joblib.dump(pipeline, MODEL_PATH)
        
        print("Training complete")





if __name__ == "__main__":
    train()
