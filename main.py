from src.preprocessing import get_preprocessed_data
from src.model_training import (
    train_random_forest
)
from src.evaluation import evaluate_model

def main():
    # RandomForest
    features_train, target_train, features_validation, target_validation, features_test, target_test = get_preprocessed_data()
    rf = train_random_forest(features_train, target_train, features_validation, target_validation)
    evaluate_model(rf, features_test, target_test, "RandomForest")

if __name__ == "__main__":
    main()