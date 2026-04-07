"""
Ticket Classification Model Training Module

This module trains and evaluates Logistic Regression models with TF-IDF vectorization
for multi-class ticket classification. It includes hyperparameter tuning and cross-validation.
"""

from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

DATA_PATH = Path("../data") / "tickets.zip"
MODEL_PATH = Path("../models") / "ticket_model.joblib"
METRICS_PATH = Path("../models") / "model_metrics.joblib"

RANDOM_STATE = 42


def build_model(class_weight=None):
    """
    Build a text classification pipeline with TF-IDF and Logistic Regression.
    
    Args:
        class_weight (str, optional): Weight to handle class imbalance. 
                                     Options: None (uniform), 'balanced'
    
    Returns:
        Pipeline: Scikit-learn pipeline with vectorizer and classifier
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True, 
            stop_words="english", 
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )),
        ("clf", LogisticRegression(
            max_iter=2000, 
            class_weight=class_weight,
            random_state=RANDOM_STATE
        ))
    ])


def evaluate(model, X_test, y_test):
    """
    Evaluate model performance on test set.
    
    Args:
        model: Trained scikit-learn model
        X_test: Test features
        y_test: Test labels
    
    Returns:
        dict: Dictionary containing accuracy, macro_f1, weighted_f1, and confusion matrix
    """
    preds = model.predict(X_test)
    report = classification_report(y_test, preds, output_dict=True)
    
    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_test, preds).tolist()
    }


def main():
    """
    Main training pipeline: load data, train models with hyperparameter tuning,
    evaluate on test set, and save best model.
    """
    print("Loading data...")
    try:
        df = pd.read_csv(DATA_PATH, compression="zip").dropna(subset=["Document", "Topic_group"])
        print(f"Loaded {len(df)} samples")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    X = df["Document"].astype(str)
    y = df["Topic_group"].astype(str)
    
    print(f"Class distribution:\n{y.value_counts()}\n")
    
    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples\n")
    
    # Model candidates with hyperparameter tuning
    candidates = [("logreg_default", None), ("logreg_balanced", "balanced")]
    best_model, best_score, best_name = None, -1, None
    results = []
    
    for name, cw in candidates:
        print(f"Training {name}...")
        model = build_model(class_weight=cw)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1_macro")
        print(f"  Cross-validation F1 scores: {cv_scores}")
        print(f"  Mean CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Train on full training set
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)
        
        metrics["cv_mean"] = float(cv_scores.mean())
        metrics["cv_std"] = float(cv_scores.std())
        results.append((name, metrics))
        
        print(f"  Test accuracy: {metrics['accuracy']:.4f}")
        print(f"  Test macro_f1: {metrics['macro_f1']:.4f}")
        print(f"  Test weighted_f1: {metrics['weighted_f1']:.4f}\n")
        
        if metrics["macro_f1"] > best_score:
            best_score = metrics["macro_f1"]
            best_model = model
            best_name = name
    
    # Save best model and metrics
    print("Results Summary:")
    for name, m in results:
        print(f"- {name}: CV_F1={m['cv_mean']:.4f} | Test_Acc={m['accuracy']:.4f} | Test_F1={m['macro_f1']:.4f}")
    
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(results, METRICS_PATH)
    
    print(f"\n✓ Best model: {best_name} (F1: {best_score:.4f})")
    print(f"✓ Saved model to: {MODEL_PATH}")
    print(f"✓ Saved metrics to: {METRICS_PATH}")
    
    # Generate visualizations
    try:
        from analyze import generate_all_visualizations
        generate_all_visualizations(results)
    except ImportError:
        print("Note: Install matplotlib and seaborn to generate visualizations")
        print("  pip install matplotlib seaborn")


if __name__ == "__main__":
    main()