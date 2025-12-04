import pickle
import os

def load_trained_model(file_path):
    """
    Safely loads a pickled model from the specified path.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file not found: {file_path}")
        
    with open(file_path, "rb") as model_file:
        return pickle.load(model_file)
