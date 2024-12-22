import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import saved_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            # Feature columns
            numerical_columns = ['reading_score', 'writing_score']
            categorical_columns = [
                'gender', 'race_ethnicity', 'parental_level_of_education',
                'lunch', 'test_preparation_course',
            ]

            # Pipelines for numeric and categorical features
            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])
            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder()),
                ("scaler", StandardScaler(with_mean=False))
            ])

            logging.info("Pipelines created for numerical and categorical features.")

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(f"Error in creating transformer object: {e}", sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Load datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Datasets loaded successfully.")

            # Ensure target column is present
            target_column = "math_score"
            if target_column not in train_df.columns or target_column not in test_df.columns:
                raise CustomException(f"Target column '{target_column}' is missing in datasets.")

            # Split into input and target features
            input_train = train_df.drop(columns=[target_column])
            target_train = train_df[target_column]

            input_test = test_df.drop(columns=[target_column])
            target_test = test_df[target_column]

            # Transform features
            preprocessor = self.get_data_transformer_object()
            input_train_transformed = preprocessor.fit_transform(input_train)
            input_test_transformed = preprocessor.transform(input_test)

            # Combine input and target features
            train_arr = np.c_[input_train_transformed, np.array(target_train)]
            test_arr = np.c_[input_test_transformed, np.array(target_test)]

            # Save preprocessor
            saved_object(self.data_transformation_config.preprocessor_obj_file_path, preprocessor)
            logging.info("Preprocessor object saved successfully.")

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(f"Error during data transformation: {e}", sys)
