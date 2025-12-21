import os
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.evaluation import evaluate_model


# RANDOM FOREST MODEL
def train_random_forest(X_train, y_train, X_val, y_val):
    estimators_options = [50, 100, 200]
    depth_options = [5, 10, None]

    best_acc = 0
    best_model = None
    best_params = {}

    print("\n" + "=" * 70)
    print(" TRAINING RANDOM FOREST")
    print("=" * 70)

    for n in estimators_options:
        for d in depth_options:
            clf = RandomForestClassifier(
                n_estimators=n,
                max_depth=d,
                random_state=42
            )

            clf.fit(X_train, y_train)
            val_preds = clf.predict(X_val)
            acc = accuracy_score(y_val, val_preds)

            depth_str = str(d) if d is not None else "None"
            print(f"Trees: {n:<3} | Max Depth: {depth_str:<4} -> Accuracy: {acc:.4f}")

            if acc > best_acc:
                best_acc = acc
                best_model = clf
                best_params = {'n_estimators': n, 'max_depth': d}

    save_path = os.path.join('models', 'best_random_forest.pkl')
    joblib.dump(best_model, save_path)

    print("\n" + "=" * 40)
    print(f"BEST RANDOM FOREST MODEL")
    print(f"Trees={best_params['n_estimators']}, Depth={best_params['max_depth']}")
    print(f"Accuracy={best_acc:.4f}")
    print("=" * 40)

    return best_model


# KNN MODEL
def train_knn(features_train, target_train, features_val, target_val, tune_hyperparameters=True):

    print("\n" + "=" * 70)
    print(" TRAINING K-NEAREST NEIGHBORS")
    print("=" * 70)

    if tune_hyperparameters:
        print("\n Hyperparameter Tuning...")
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11, 15],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        }

        base_model = KNeighborsClassifier()
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(features_train, target_train)

        print(f"   Best parameters: {grid_search.best_params_}")
        print(f"   Best CV score: {grid_search.best_score_:.4f}")

        model = grid_search.best_estimator_
    else:
        model = KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            metric='euclidean'
        )

    model.fit(features_train, target_train)

    save_path = os.path.join("models", "best_KNN.pkl")
    joblib.dump(model, save_path)

    print("\n Evaluating on Validation Set...")
    evaluate_model(model, features_val, target_val, "KNN (Validation)")

    return model


# LOGISTIC REGRESSION MODEL
def train_logistic_regression(X_train, y_train, X_val, y_val):
    print("\n" + "=" * 70)
    print(" TRAINING LOGISTIC REGRESSION")
    print("=" * 70)

    C_values = [0.1, 1.0, 10.0]

    best_acc = 0
    best_model = None
    best_C = None

    for C in C_values:
        model = LogisticRegression(
            C=C,
            max_iter=1000,
            solver="lbfgs"
        )

        model.fit(X_train, y_train)

        val_preds = model.predict(X_val)
        acc = accuracy_score(y_val, val_preds)

        print(f"C={C:<4} -> Validation Accuracy={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_C = C

    save_path = os.path.join("models", "best_logistic_regression.pkl")
    joblib.dump(best_model, save_path)

    print("\n" + "=" * 40)
    print("BEST LOGISTIC REGRESSION MODEL")
    print(f"C={best_C}")
    print(f"Accuracy={best_acc:.4f}")
    print("=" * 40)

    return best_model


# NAIVE BAYES MODEL
def train_naive_bayes(X_train, y_train, X_val, y_val):
    print("\n" + "=" * 70)
    print(" TRAINING NAIVE BAYES")
    print("=" * 70)

    # var_smoothing values to try (smaller values = less smoothing)
    var_smoothing_options = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]

    best_acc = 0
    best_model = None
    best_var_smoothing = None

    for var_smoothing in var_smoothing_options:
        model = GaussianNB(var_smoothing=var_smoothing)

        model.fit(X_train, y_train)

        val_preds = model.predict(X_val)
        acc = accuracy_score(y_val, val_preds)

        print(f"var_smoothing={var_smoothing:<8} -> Validation Accuracy={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_var_smoothing = var_smoothing

    save_path = os.path.join("models", "best_naive_bayes.pkl")
    joblib.dump(best_model, save_path)

    print("\n" + "=" * 40)
    print("BEST NAIVE BAYES MODEL")
    print(f"var_smoothing={best_var_smoothing}")
    print(f"Accuracy={best_acc:.4f}")
    print("=" * 40)

    return best_model


# DECISION TREE MODEL
def train_decision_tree(X_train, y_train, X_val, y_val, tune_hyperparameters=True):

    print("\n" + "=" * 70)
    print(" TRAINING DECISION TREE")
    print("=" * 70)

    if tune_hyperparameters:
        print("\n Hyperparameter Tuning...")

        param_grid = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 4, 6, 8],
            'min_samples_leaf': [1, 2, 4]
        }

        base_model = DecisionTreeClassifier(random_state=42)

        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        print(f"   Best parameters: {grid_search.best_params_}")
        print(f"   Best CV score: {grid_search.best_score_:.4f}")

        model = grid_search.best_estimator_

    else:
        model = DecisionTreeClassifier(
            criterion='gini',
            max_depth=15,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42
        )

    # Train final model
    model.fit(X_train, y_train)

    save_path = os.path.join("models", "best_decision_tree.pkl")
    joblib.dump(model, save_path)

    print("\n Evaluating on Validation Set...")
    evaluate_model(model, X_val, y_val, "DecisionTree (Validation)")

    return model

def train_svm(X_train, y_train, X_val, y_val):
    print("\n" + "=" * 70)
    print(" TRAINING SUPPORT VECTOR MACHINE (SVM)")
    print("=" * 70)

    kernels = ['linear', 'rbf']
    C_values = [0.1, 1, 10]
    gamma_values = ['scale', 0.01, 0.1]

    best_acc = 0
    best_model = None
    best_params = {}

    for kernel in kernels:
        for C in C_values:
            for gamma in gamma_values if kernel == 'rbf' else ['scale']:

                model = SVC(
                    kernel=kernel,
                    C=C,
                    gamma=gamma,
                    probability=True,
                    random_state=42
                )

                model.fit(X_train, y_train)
                val_preds = model.predict(X_val)
                acc = accuracy_score(y_val, val_preds)

                print(
                    f"Kernel={kernel:<6} | C={C:<4} | Gamma={gamma:<6} "
                    f"-> Validation Accuracy={acc:.4f}"
                )

                if acc > best_acc:
                    best_acc = acc
                    best_model = model
                    best_params = {
                        'kernel': kernel,
                        'C': C,
                        'gamma': gamma
                    }

    save_path = os.path.join("models", "best_svm.pkl")
    joblib.dump(best_model, save_path)

    print("\n" + "=" * 40)
    print("BEST SVM MODEL")
    print(f"Kernel={best_params['kernel']}")
    print(f"C={best_params['C']}")
    print(f"Gamma={best_params['gamma']}")
    print(f"Accuracy={best_acc:.4f}")
    print("=" * 40)

    return best_model