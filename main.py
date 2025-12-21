from src.preprocessing import get_preprocessed_data
from src.model_training import (
    train_random_forest,
    train_knn,
    train_naive_bayes,
    train_logistic_regression,
    train_decision_tree,
    train_svm
)
from src.evaluation import evaluate_model

def main():
    # preprocessing
    features_train, target_train, features_validation, target_validation, features_test, target_test = get_preprocessed_data()

    # RandomForest
    rf = train_random_forest(features_train, target_train, features_validation, target_validation)
    evaluate_model(rf, features_test, target_test, "RandomForest")

    # KNN
    knn = train_knn(features_train, target_train,features_validation, target_validation)
    evaluate_model(knn, features_test, target_test, "KNN (Test)")

    # Logistic Regression
    lr = train_logistic_regression(features_train, target_train, features_validation, target_validation)
    evaluate_model(lr, features_test, target_test, "LogisticRegression")

    # Naive Bayes
    nb = train_naive_bayes(features_train, target_train, features_validation, target_validation)
    evaluate_model(nb, features_test, target_test, "NaiveBayes")

    # Decision Tree
    dt = train_decision_tree(features_train, target_train, features_validation, target_validation)
    evaluate_model(dt, features_test, target_test, "DecisionTree (Test)")

    # SVM
    svm = train_svm(features_train, target_train, features_validation, target_validation)
    evaluate_model(svm, features_test, target_test, "SVM")


if __name__ == "__main__":
    main()