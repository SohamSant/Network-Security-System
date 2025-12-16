import os
import sys
import certifi
ca = certifi.where()

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import numpy as np
import pymongo
from typing import List
from sklearn.model_selection import train_test_split

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact


class DataIngestion:
    """
    Data Ingestion component for the Network Security ML pipeline.
    Handles fetching data from MongoDB and preparing train/test splits.
    """
    
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        """
        Initialize DataIngestion component with configuration
        
        Args:
            data_ingestion_config: Configuration object containing all paths and parameters
        """
        try:
            self.data_ingestion_config = data_ingestion_config
            logging.info("DataIngestion component initialized successfully")
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def export_collection_as_dataframe(self) -> pd.DataFrame:
        """
        Export MongoDB collection as pandas DataFrame
        
        Returns:
            pd.DataFrame: Data exported from MongoDB collection
        """
        try:
            logging.info("Exporting data from MongoDB collection")
            
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name
            
            # Connect to MongoDB
            mongo_client = pymongo.MongoClient(os.getenv("MONGO_DB_URL"), tlsCAFile=ca)
            
            logging.info(f"Connected to MongoDB. Database: {database_name}, Collection: {collection_name}")
            
            # Access database and collection
            collection = mongo_client[database_name][collection_name]
            
            # Convert collection to DataFrame
            df = pd.DataFrame(list(collection.find()))
            
            # Drop MongoDB's _id column if it exists
            if "_id" in df.columns:
                df = df.drop(columns=["_id"], axis=1)
            
            # Replace 'na' with NaN
            df.replace({"na": np.nan}, inplace=True)
            
            logging.info(f"Data exported successfully. Shape: {df.shape}")
            
            return df
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def export_data_into_feature_store(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Export DataFrame to feature store as CSV file
        
        Args:
            dataframe: DataFrame to be exported
            
        Returns:
            pd.DataFrame: The same dataframe that was saved
        """
        try:
            logging.info("Exporting data to feature store")
            
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            
            # Create directory if it doesn't exist
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            
            # Save DataFrame to CSV
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            
            logging.info(f"Data saved to feature store at: {feature_store_file_path}")
            
            return dataframe
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def split_data_as_train_test(self, dataframe: pd.DataFrame) -> None:
        """
        Split the dataframe into train and test sets and save them
        
        Args:
            dataframe: DataFrame to be split
        """
        try:
            logging.info("Splitting data into train and test sets")
            
            # Perform train-test split
            train_set, test_set = train_test_split(
                dataframe, 
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=42
            )
            
            logging.info(f"Train set shape: {train_set.shape}, Test set shape: {test_set.shape}")
            
            # Create directory for train and test files
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)
            
            # Save train set
            train_set.to_csv(
                self.data_ingestion_config.training_file_path,
                index=False,
                header=True
            )
            
            # Save test set
            test_set.to_csv(
                self.data_ingestion_config.testing_file_path,
                index=False,
                header=True
            )
            
            logging.info(f"Train data saved to: {self.data_ingestion_config.training_file_path}")
            logging.info(f"Test data saved to: {self.data_ingestion_config.testing_file_path}")
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Main method to initiate the data ingestion process
        
        Returns:
            DataIngestionArtifact: Artifact containing paths to train and test files
        """
        try:
            logging.info("="*70)
            logging.info("Starting Data Ingestion process")
            logging.info("="*70)
            
            # Step 1: Export data from MongoDB
            dataframe = self.export_collection_as_dataframe()
            
            # Step 2: Save data to feature store
            dataframe = self.export_data_into_feature_store(dataframe)
            
            # Step 3: Split data into train and test sets
            self.split_data_as_train_test(dataframe)
            
            logging.info("Data Ingestion process completed successfully")
            logging.info("="*70)
            
            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )
            
            logging.info(f"Data Ingestion Artifact: {data_ingestion_artifact}")
            
            return data_ingestion_artifact
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)


if __name__ == "__main__":
    """
    Test the DataIngestion component
    """
    from networksecurity.entity.config_entity import TrainingPipelineConfig
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Create pipeline config
    training_pipeline_config = TrainingPipelineConfig()
    
    # Create data ingestion config
    data_ingestion_config = DataIngestionConfig(training_pipeline_config)
    
    # Initialize and run data ingestion
    data_ingestion = DataIngestion(data_ingestion_config)
    artifact = data_ingestion.initiate_data_ingestion()
    
    print(f"\nData Ingestion Artifact:")
    print(f"Train file: {artifact.trained_file_path}")
    print(f"Test file: {artifact.test_file_path}")
