import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'dataset-diabete.csv')
OUTPUT_PATH = os.path.join(BASE_DIR, 'data', 'dataset-diabete-processed.csv')



def knn_imputation(data):
    data_knn = data.copy()
    null_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin'] 
    
    for col in null_columns:
        data_knn[col] = data_knn[col].replace(0, np.nan)
        
    # Normalisation 
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_knn)
    data_scaled =pd.DataFrame(scaled_data, columns=data_knn.columns)
    
    # Imputation KNN
    knn_imputer = KNNImputer(n_neighbors=5)
    imputed_data = knn_imputer.fit_transform(data_scaled)
    data_imputed = pd.DataFrame(imputed_data, columns=data_knn.columns)
    
    data_final_knn = pd.DataFrame(scaler.inverse_transform(data_imputed), columns=data_knn.columns)
    
    return data_final_knn



def process_data():
    # load data
    
    print(f"Data Loading : ...")
    
    data = pd.read_csv(DATA_PATH, index_col=0)
    
    print(f"Initial shape : {data.shape}")
    
    # duplicated values
    duplicates_count = data.duplicated().sum()
    
    print(f"Shape before removing duplicates : {duplicates_count}")
    
    data = data.drop_duplicates()
    
    print(f"Shape after removing duplicates : {data.shape}")
    
    # missing values
    cols_with_missing = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    data[cols_with_missing] = data[cols_with_missing].replace(0, np.nan)
    
    missing_values = data.isnull().sum()
    
    print(f"Missing values count after replacing 0 with Nan : {missing_values}")    
    
    data = knn_imputation(data)
    
    # add risk_category column
    
    condition = (data['Glucose'] > 126) & (data['BMI'] > 30) & (data['DiabetesPedigreeFunction'] > 0.5)
    
    data['risk_category'] = condition.astype(int)
    
    print("Risk category distribution:")
    print(data['risk_category'].value_counts())
    
    # outliers manage
    print("Appling final feature transformation : ")
    data = preparation_final_model(data)
    
    # save data
    print(f"Saving processed data to {OUTPUT_PATH} : ")
    data.to_csv(OUTPUT_PATH)
    
    print("Processing data done.")    
    
    
    
def preparation_final_model(df):
    """fonction for log1p and winsorize"""
    from scipy.stats.mstats import winsorize
    
    df_prepare = df.copy()
    
    df_prepare['Insulin'] = np.log1p(df['Insulin'])
    df_prepare['DiabetesPedigreeFunction'] = np.log1p(df['DiabetesPedigreeFunction'])
    df_prepare['Pregnancies'] = winsorize(df['Pregnancies'], limits=[0.05, 0.05]).data
    
    return df_prepare

if __name__ == "__main__":
    process_data()