import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def get_model(problem_type):
    """
    Get all relevant models based on problem type
    
    Args:
        problem_type: 'classification' or 'regression'
    
    Returns:
        Dictionary of model instances
    """
    if problem_type == 'classification':
        return {
            'logistic_regression': LogisticRegression(random_state=42),
            'decision_tree': DecisionTreeClassifier(random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'xgboost': XGBClassifier(random_state=42),
            'knn': KNeighborsClassifier(),
            'svm': SVC(random_state=42),
            'naive_bayes': GaussianNB(),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'neural_network': MLPClassifier(random_state=42, max_iter=1000)
        }
    elif problem_type == 'regression':
        return {
            'linear_regression': LinearRegression(),
            'decision_tree': DecisionTreeRegressor(random_state=42),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'xgboost': XGBRegressor(random_state=42),
            'knn': KNeighborsRegressor(),
            'svm': SVR(),
            'gradient_boosting': GradientBoostingRegressor(random_state=42),
            'neural_network': MLPRegressor(random_state=42, max_iter=1000)
        }
    else:
        raise ValueError("problem_type must be either 'classification' or 'regression'")


def run_ml_pipeline(df, target_column):
    """
    Complete ML pipeline that automatically detects problem type and runs all relevant models
    
    Args:
        df: Input DataFrame (cleaned)
        target_column: Name of target variable
    
    Returns:
        Tuple containing:
        1. Dictionary with model comparisons (JSON-friendly)
        2. Dictionary of feature importance plots
    """
    # Automatically detect problem type
    unique_values = df[target_column].nunique()
    problem_type = 'classification' if unique_values < 10 else 'regression'
    print(f"\nDetected problem type: {problem_type}")
    
    # Get all relevant models
    models = get_model(problem_type)
    model_comparisons = {
        "problem_type": problem_type,
        "models": []
    }
    feature_importance_plots = {}
    
    # Prepare data once for all models
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    
    print("\nTraining all relevant models...")
    print("-" * 50)
    print(f"{'Model':<20} {'Train Score':>15} {'Test Score':>15}")
    print("-" * 50)
    
    # Train and evaluate each model
    for model_name, model in models.items():
        try:
            # Train model
            model.fit(X_train, y_train)
            
            # Get predictions
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            # Calculate metrics
            if problem_type == 'classification':
                train_score = float(accuracy_score(y_train, train_pred))
                test_score = float(accuracy_score(y_test, test_pred))
                metric_name = "accuracy"
            else:
                train_score = float(r2_score(y_train, train_pred))
                test_score = float(r2_score(y_test, test_pred))
                metric_name = "r2_score"
            
            # Store model results
            model_comparisons["models"].append({
                "model_name": model_name,
                "metrics": {
                    "train_score": train_score,
                    "test_score": test_score,
                    "metric_type": metric_name
                }
            })
            
            print(f"{model_name:<20} {train_score:>15.3f} {test_score:>15.3f}")
            
            # Feature importance plot
            if hasattr(model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                plt.figure(figsize=(10, 6))
                sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
                plt.title(f'Top 10 Feature Importance - {model_name}')
                plt.tight_layout()
                
                feature_importance_plots[model_name] = plt
                
                # Add feature importance to JSON
                model_comparisons["models"][-1]["feature_importance"] = feature_importance.head(10).to_dict('records')
            
        except Exception as e:
            print(f"{model_name:<20} Failed to run: {str(e)}")
            model_comparisons["models"].append({
                "model_name": model_name,
                "error": str(e)
            })
    
    # Sort models by test score
    model_comparisons["models"].sort(key=lambda x: x.get("metrics", {}).get("test_score", -1), reverse=True)
    
    # Add best model info
    valid_models = [m for m in model_comparisons["models"] if "error" not in m]
    if valid_models:
        best_model = valid_models[0]
        model_comparisons["best_model"] = {
            "name": best_model["model_name"],
            "test_score": best_model["metrics"]["test_score"]
        }
        print(f"\nBest Model: {best_model['model_name']}")
        print(f"Test Score: {best_model['metrics']['test_score']:.3f}")
    
    return model_comparisons, feature_importance_plots