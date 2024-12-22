import os
import sys
import dill
from sklearn.metrics import r2_score
from src.exception import CustomException


def saved_object(file_path, obj):
    """
    Saves a Python object to the specified file path using dill serialization.

    Args:
        file_path (str): The path where the object will be saved.
        obj (Any): The Python object to save.

    Raises:
        CustomException: If any exception occurs during the saving process.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models):
    """
    Evaluates multiple regression models on the training and test data.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): Test labels.
        models (dict): A dictionary of model names and their instances.

    Returns:
        dict: A dictionary with model names as keys and R2 scores as values.

    Raises:
        CustomException: If any exception occurs during the evaluation process.
    """
    try:
        report = {}
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            test_model_score = r2_score(y_test, y_pred)
            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(f"Error evaluating models: {e}", sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)



