# main.py

from src.preprocessing import get_preprocessed_data
from src.model_training import train_random_forest
from src.evaluation import evaluate_model
from src.model_training import train_random_forest, train_knn
def main():
    # الحصول على البيانات بعد preprocessing
    features_train, target_train, features_validation, target_validation, features_test, target_test = get_preprocessed_data()

    # تدريب RandomForest
    rf = train_random_forest(features_train, target_train, features_validation, target_validation)
    evaluate_model(rf, features_test, target_test, "RandomForest")

    # تدريب KNN
    knn = train_knn(
        features_train, target_train,
        features_validation, target_validation,
        features_test, target_test
    )

if __name__ == "__main__":
    main()