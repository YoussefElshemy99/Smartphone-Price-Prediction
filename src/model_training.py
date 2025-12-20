import os

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler



def train_logistic_regression(X_train, y_train, X_val, y_val):
    """
    Train Logistic Regression with different regularization strengths
    and save the best performing model.
    """

    # Logistic Regression is sensitive to feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Hyperparameter options (Regularization strength)
    C_options = [0.01, 0.1, 1.0]

    best_acc = 0
    best_model = None
    best_C = None

    print("\nTraining Logistic Regression models...")

    for C in C_options:
        clf = LogisticRegression(
            C=C,
            max_iter=1000,
            solver="lbfgs",
            multi_class="auto"
        )

        # Train
        clf.fit(X_train_scaled, y_train)

        # Evaluate
        val_preds = clf.predict(X_val_scaled)
        acc = accuracy_score(y_val, val_preds)

        print(f"C: {C:<4} -> Accuracy: {acc:.4f}")

        # Save best model
        if acc > best_acc:
            best_acc = acc
            best_model = clf
            best_C = C

    # Save best model and scaler
    model_path = os.path.join('models', 'best_logistic_regression.pkl')
    scaler_path = os.path.join('models', 'logistic_regression_scaler.pkl')

    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)

    print("\n" + "=" * 40)
    print(f"BEST LOGISTIC REGRESSION MODEL: C={best_C}")
    print(f"BEST ACCURACY: {best_acc:.4f}")
    print("=" * 40)

    return best_model

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
