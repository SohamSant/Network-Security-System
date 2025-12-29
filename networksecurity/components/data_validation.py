import os
import sys
import pandas as pd
from scipy.stats import ks_2samp

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import DataValidationConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from networksecurity.constant.training_pipeline import SCHEMA_FILE_PATH
from networksecurity.utils.main_utils.utils import read_yaml_file, write_yaml_file

class DataValidation:
    """
    Data Validation component for the Network Security ML pipeline.
    Validates data quality and performs drift detection.
    """
    
    def __init__(
        self, 
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        """
        Validate if all required columns are present
        """
        try:
            number_of_columns = len(self._schema_config['columns'])
            logging.info(f"Required number of columns: {number_of_columns}")
            logging.info(f"Dataframe has columns: {len(dataframe.columns)}")
            
            if len(dataframe.columns) == number_of_columns:
                return True
            return False
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def is_numerical_column_exist(self, dataframe: pd.DataFrame) -> bool:
        """
        Validate if all numerical columns from schema exist in dataframe
        """
        try:
            numerical_columns = self._schema_config['numerical_columns']
            dataframe_columns = dataframe.columns

            missing_numerical_columns = []
            for column in numerical_columns:
                if column not in dataframe_columns:
                    missing_numerical_columns.append(column)
            
            if len(missing_numerical_columns) > 0:
                logging.info(f"Missing numerical columns: {missing_numerical_columns}")
                return False
            return True
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def detect_dataset_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame,threshold: float = 0.05) -> bool:
        """
        Detect data drift between base and current datasets using KS test
        """
        try:
            drift_status = False
            report = {}
            
            for column in base_df.columns:
                d1 = base_df[column]
                d2 = current_df[column]
                
                # Perform KS test
                is_same_dist = ks_2samp(d1, d2)
                
                # larger p-value means distributions are likely the same
                # But here we probably want to check if they are DIFFERENT
                # standard approach: if p_value < alpha, we reject H0 (same dist), so they are different (drift)
                
                if is_same_dist.pvalue >= threshold:
                    # Same distribution
                    is_found = False
                else:
                    # Different distribution -> Drift
                    is_found = True
                    drift_status = True 
                
                report.update({
                    column: {
                        "p_value": float(is_same_dist.pvalue),
                        "drift_status": is_found
                    }
                })
            
            drift_report_file_path = self.data_validation_config.drift_report_file_path
            
            # Create directory
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path, exist_ok=True)
            
            write_yaml_file(file_path=drift_report_file_path, content=report)
            
            return drift_status
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path
            
            train_dataframe = DataValidation.read_data(train_file_path)
            test_dataframe = DataValidation.read_data(test_file_path)
            
            validation_status = True
            
            # 1. Validate number of columns
            status = self.validate_number_of_columns(dataframe=train_dataframe)
            if not status:
                validation_status = False
            
            status = self.validate_number_of_columns(dataframe=test_dataframe)
            if not status:
                validation_status = False
                
            # 2. Validate numerical columns
            status = self.is_numerical_column_exist(dataframe=train_dataframe)
            if not status:
                validation_status = False
            
            status = self.is_numerical_column_exist(dataframe=test_dataframe)
            if not status:
                validation_status = False

            # 3. Detect drift
            if validation_status:
                drift_status = self.detect_dataset_drift(base_df=train_dataframe, current_df=test_dataframe)
            
            # Save valid data (if we reached here, structure is valid)
            # Logic: If structural validation fails, we raised exception. 
            # If drift is detected, we typically still save the data but flag it, or maybe reject it. 
            # The prompt isn't specific, but usually validation artifact just reports status.
            # However, previous code copied data to valid/invalid. Let's keep that pattern but refined.
            # Wait, if I raised Exception above, code would stop. 
            # The previous code had a boolean `validation_status` and saved to invalid if false.
            # I should probably NOT raise exception immediately but set status to False to allow 'Invalid' data saving.
            # But for structural errors (missing columns), the code might break later if I proceed. 
            # I will modify the logic to handle status gracefully without crashing if possible, 
            # OR simple assumption: valid structure = valid, drift is just a warning/report usually?
            # Let's assume strict validation: if columns missing = invalid.
            
            # Let's avoid raising Exception for logical validation failures to allow saving to 'invalid' dir?
            # Actually, standard pipeline usually halts on schema mismatch. 
            # I'll stick to: Raise exception if critical schema mismatch? 
            # Or return artifact with validation_status=False.
            
            # Let's revert to boolean tracking to match the 'invalid/valid' dir logic.
            
            # Save valid data flow starts here
            
            dir_path = self.data_validation_config.valid_data_dir if validation_status else self.data_validation_config.invalid_data_dir
            os.makedirs(dir_path, exist_ok=True)
            
            train_filename = os.path.basename(train_file_path)
            test_filename = os.path.basename(test_file_path)

            if validation_status:
                train_dataframe.to_csv(self.data_validation_config.valid_train_file_path, index=False, header=True)
                test_dataframe.to_csv(self.data_validation_config.valid_test_file_path, index=False, header=True)
            else:
                 train_dataframe.to_csv(self.data_validation_config.invalid_train_file_path, index=False, header=True)
                 test_dataframe.to_csv(self.data_validation_config.invalid_test_file_path, index=False, header=True)

            # Artifact
            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=self.data_validation_config.invalid_train_file_path,
                invalid_test_file_path=self.data_validation_config.invalid_test_file_path,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )
            
            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)
