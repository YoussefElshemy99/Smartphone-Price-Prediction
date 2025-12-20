import os

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_random_forest(X_train, y_train, X_val, y_val):
    # Define Hyperparameters to vary
    # n_estimators: Number of trees in the forest
    # max_depth: The maximum depth of the tree
    estimators_options = [50, 100, 200]
    depth_options = [5, 10, None]  # None means unlimited depth

    best_acc = 0
    best_model = None
    best_params = {}
    results = []

    # Iterate through all combinations (Grid Search logic)
    for n in estimators_options:
        for d in depth_options:
            # Initialize model
            clf = RandomForestClassifier(n_estimators=n, max_depth=d, random_state=42)

            # Train
            clf.fit(X_train, y_train)

            # Evaluate
            val_preds = clf.predict(X_val)
            acc = accuracy_score(y_val, val_preds)

            # Record result
            depth_str = str(d) if d is not None else "None"
            print(f"Trees: {n:<3} | Max Depth: {depth_str:<4} -> Accuracy: {acc:.4f}")

            results.append({'n_estimators': n, 'max_depth': depth_str, 'accuracy': acc})

            # Save best model
            if acc > best_acc:
                best_acc = acc
                best_model = clf
                best_params = {'n_estimators': n, 'max_depth': d}

    # Save using joblib
    save_path = os.path.join('models', 'best_random_forest.pkl')
    joblib.dump(best_model, save_path)

    print("\n" + "=" * 40)
    print(f"BEST MODEL: Trees={best_params['n_estimators']}, Depth={best_params['max_depth']}")
    print(f"BEST ACCURACY: {best_acc:.4f}")
    print("\n" + "=" * 40)

    return best_model

# KNN--MODEL

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from src.evaluation import evaluate_model  # استخدم نفس evaluation function

def train_knn(features_train, target_train, features_val, target_val, features_test, target_test, tune_hyperparameters=True):
    print("\n" + "=" * 70)
    print("🔍 TRAINING K-NEAREST NEIGHBORS")
    print("=" * 70)

    if tune_hyperparameters:
        print("\n[1/3] Hyperparameter Tuning...")
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11, 15],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        }

        base_model = KNeighborsClassifier()
        grid_search = GridSearchCV(
            base_model, param_grid, cv=5,
            scoring='f1', n_jobs=-1, verbose=1
        )
        grid_search.fit(features_train, target_train)

        best_params = grid_search.best_params_
        print(f"  Best parameters: {best_params}")
        print(f"  Best CV score: {grid_search.best_score_:.4f}")

        model = grid_search.best_estimator_
    else:
        model = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='euclidean')

    model.fit(features_train, target_train)

    evaluate_model(model, features_val, target_val, "KNN (Validation)")

    evaluate_model(model, features_test, target_test, "KNN (Test)")

    return model