import os
import sys
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Starting the data ingestion process.")
        try:
            # Load raw data
            data_file_path = os.path.join("notebook", "data", "stud.csv")
            df = pd.read_csv(data_file_path)

            # Ensure 'artifacts' directory exists
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Raw data saved successfully.")

            # Split into train and test sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)
            logging.info("Train-test split completed successfully.")

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            raise CustomException(f"Error during data ingestion: {e}", sys)


if __name__ == "__main__":
    ingestion = DataIngestion()
    train_data, test_data = ingestion.initiate_data_ingestion()

    # Proceed with data transformation and model training
    from src.components.data_transformation import DataTransformation
    transformer = DataTransformation()
    train_arr, test_arr, preprocessor_file_path = transformer.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))
