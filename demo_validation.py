import os
import sys
import pandas as pd
from networksecurity.entity.config_entity import DataValidationConfig, TrainingPipelineConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact
from networksecurity.components.data_validation import DataValidation
from networksecurity.constant.training_pipeline import SCHEMA_FILE_PATH
from networksecurity.utils.main_utils.utils import read_yaml_file

def verify_validation():
    try:
        # 1. Setup Mock Data
        os.makedirs("dummy_data", exist_ok=True)
        train_path = "dummy_data/train.csv"
        test_path = "dummy_data/test.csv"
        
        # Load schema to know what columns to create
        schema = read_yaml_file(SCHEMA_FILE_PATH)
        columns = [list(x.keys())[0] for x in schema['columns']]
        
        # Create dummy dataframe with correct columns
        df = pd.DataFrame(columns=columns)
        # Add a dummy row with random values (all 0 for simplicity)
        df.loc[0] = [0] * len(columns)
        
        df.to_csv(train_path, index=False)
        df.to_csv(test_path, index=False)
        
        print("Dummy data created.")

        # 2. Setup Configs & Artifacts
        tp_config = TrainingPipelineConfig()
        dv_config = DataValidationConfig(tp_config)
        
        di_artifact = DataIngestionArtifact(
            trained_file_path=os.path.abspath(train_path),
            test_file_path=os.path.abspath(test_path)
        )
        
        # 3. Init Validation
        dv = DataValidation(di_artifact, dv_config)
        print("DataValidation initialized.")
        
        # 4. Run Validation
        artifact = dv.initiate_data_validation()
        print(f"Validation Artifact: {artifact}")
        print(f"Validation Status: {artifact.validation_status}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_validation()
