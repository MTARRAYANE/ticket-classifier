"""
Ticket Classification Inference Module

This module provides functionality to make predictions using a trained ticket classifier.
It loads a pre-trained model and predicts ticket categories with confidence scores.
"""

from pathlib import Path
import joblib

MODEL_PATH = Path("../models") / "ticket_model.joblib"


def load_model(model_path=MODEL_PATH):
    """
    Load the trained ticket classification model.
    
    Args:
        model_path (Path): Path to the serialized model file
    
    Returns:
        model: Loaded scikit-learn pipeline
    
    Raises:
        FileNotFoundError: If model file does not exist
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Please train a model first using train.py"
        )
    return joblib.load(model_path)


# Load model once at module import
try:
    model = load_model()
except FileNotFoundError as e:
    print(f"Warning: {e}")
    model = None


def predict_ticket(text, confidence_threshold=0.0):
    """
    Predict ticket category and confidence score.
    
    Args:
        text (str): Ticket text to classify
        confidence_threshold (float): Minimum confidence (0-1). Lower values are more permissive.
    
    Returns:
        dict: Prediction results containing:
            - 'category': Predicted ticket category
            - 'confidence': Confidence score (0-1)
            - 'is_confident': Boolean indicating if confidence exceeds threshold
    
    Raises:
        ValueError: If input text is empty or model is not loaded
    """
    if model is None:
        raise ValueError("Model not loaded. Please train a model first.")
    
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input text must be a non-empty string")
    
    pred = model.predict([text])[0]
    proba = model.predict_proba([text])[0]
    confidence = float(proba.max())
    
    return {
        "category": pred,
        "confidence": round(confidence, 4),
        "is_confident": confidence >= confidence_threshold
    }


if __name__ == "__main__":
    """
    Test the inference module with sample tickets
    """
    test_tickets = [
        "please give me administrative rights to install software on my laptop",
        "my password needs to be reset I cannot login",
        "the network connection is very slow today",
        "can you help me install microsoft office",
        "my printer is not working properly"
    ]
    
    print("Testing Ticket Classifier")
    print("=" * 60)
    
    for i, ticket in enumerate(test_tickets, 1):
        try:
            result = predict_ticket(ticket)
            print(f"\nTicket {i}: {ticket[:50]}...")
            print(f"  Category: {result['category']}")
            print(f"  Confidence: {result['confidence']}")
            print(f"  Confident: {result['is_confident']}")
        except ValueError as e:
            print(f"Error processing ticket {i}: {e}")