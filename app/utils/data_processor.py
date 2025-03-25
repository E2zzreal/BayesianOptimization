import pandas as pd
from sklearn.preprocessing import StandardScaler

class DataProcessor:
    """
    A class for data loading, preprocessing and feature scaling.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
    
    def load_data(self, file_path):
        """
        Load data from a file.
        
        Args:
            file_path (str): Path to the data file
            
        Returns:
            pd.DataFrame: Loaded data
        """
        return pd.read_csv(file_path)
    
    def preprocess_data(self, df):
        """
        Perform basic data preprocessing.
        
        Args:
            df (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Processed data
        """
        # Drop rows with missing values
        df = df.dropna()
        return df
    
    def scale_features(self, df, columns):
        """
        Scale specified features using StandardScaler.
        
        Args:
            df (pd.DataFrame): Input data
            columns (list): List of columns to scale
            
        Returns:
            pd.DataFrame: Data with scaled features
        """
        df[columns] = self.scaler.fit_transform(df[columns])
        return df