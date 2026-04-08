# Ticket Classification System 🎫

## Description
Multi-class text classifier for IT/Support tickets. Classifies ticket text into topic groups using TF-IDF feature extraction and Logistic Regression with hyperparameter optimization.

## Features
- **Machine Learning**: TF-IDF vectorization with n-gram support + Logistic Regression
- **Model Optimization**: Cross-validation and hyperparameter tuning for best performance
- **Evaluation Metrics**: Accuracy, F1-scores, classification reports, and confusion matrices
- **Error Handling**: Robust input validation and error management
- **Comprehensive Logging**: Detailed training logs with cross-validation statistics
- **Data Visualization**: Automatic generation of 6+ performance charts and plots (matplotlib/seaborn)
- **Unit Tests**: Test suite for inference functionality
- **Analysis Tools**: Metrics visualization and export utilities

## Technical Stack
- **ML Framework**: scikit-learn (Pipeline, TfidfVectorizer, LogisticRegression)
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Model Serialization**: joblib
- **Testing**: unittest
- **Python**: 3.7+

## Installation

1. Clone the repository:
```bash
git clone https://github.com/MTARRAYANE/ticket-classifier.git
cd ticket-classifier
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Make Predictions on Tickets
```bash
cd src
python inference.py
```

This script demonstrates inference on sample tickets. You can modify test cases in the script.

**Python API:**
```python
from inference import predict_ticket

result = predict_ticket("please reset my password")
print(result)
# Output: {'category': 'Password Reset', 'confidence': 0.8734, 'is_confident': True}
```

### Train a New Model
```bash
cd src
python train.py
```

This trains multiple models with cross-validation, automatically selects the best one by macro F1 score, and saves it.

**Training Output:**
- `models/ticket_model.joblib` - Best trained model
- `models/model_metrics.joblib` - Full training metrics

### Run Unit Tests
```bash
cd src
python -m unittest test_inference.py -v
```

Tests cover:
- Valid predictions
- Confidence threshold filtering
- Error handling (empty strings, invalid input types)
- Consistency across multiple predictions

### Analyze Model Performance
```bash
cd src
python analyze.py
```

Generates multiple visualizations saved to `models/`:
- **Metrics Comparison** - Bar chart comparing accuracy, macro F1, and weighted F1 across models
- **Confusion Matrix** - Heatmap showing prediction correctness for each class
- **Class Performance** - Per-class precision, recall, and F1 scores
- **Cross-Validation Scores** - Model performance across CV folds with error bars
- **Confidence Distribution** - Histogram of prediction confidence scores
- **Metrics Export** - JSON file with all metrics for external analysis

## Project Structure
```
ticket-classifier/
├── src/
│   ├── train.py              # Training pipeline with CV & hyperparameter tuning
│   ├── inference.py          # Prediction interface with error handling
│   ├── analyze.py            # Evaluation & analysis utilities
│   ├── test_inference.py     # Unit tests for inference
│   └── readme.md             # Training documentation
├── models/                   # Saved model files
│   ├── ticket_model.joblib   # Trained model (excluded from git)
│   └── model_metrics.joblib  # Training metrics (excluded from git)
├── data/                     # Training/test data
│   └── tickets.zip          # Ticket dataset (excluded from git)
├── requirements.txt          # Python dependencies
├── README.md                # This file
└── .gitignore              # Git ignore rules
```

## Model Performance
The model uses:
- **Cross-validation**: 5-fold stratified cross-validation to estimate generalization
- **Class balancing**: Tests both default and balanced class weights
- **TF-IDF Parameters**:
  - Max features: 5000
  - N-grams: 1-2 (unigrams + bigrams)
  - Min document frequency: 2
  - Max document frequency: 80%

## Key Metrics
- **Accuracy**: Overall prediction correctness
- **Macro F1-Score**: Unweighted average of F1 per class (best for imbalanced data)
- **Weighted F1-Score**: Weighted by class support
- **Confusion Matrix**: Per-class prediction breakdown

## Requirements
- Python 3.7+
- pandas >= 1.0.0
- numpy >= 1.17.0
- scikit-learn >= 0.22.0
- joblib >= 0.14.0
- matplotlib >= 3.1.0
- seaborn >= 0.11.0

## Best Practices Implemented
✅ Docstrings for all functions (Google style)  
✅ Cross-validation for robust evaluation  
✅ Hyperparameter tuning with model comparison  
✅ Error handling with informative messages  
✅ Unit tests with edge case coverage  
✅ Console logging with progress indicators  
✅ Metrics export for reproducibility  

## Future Enhancements
- [ ] Add confusion matrix visualization
- [ ] Implement ROC/AUC curves
- [ ] Support for additional classifiers (SVM, Random Forest)
- [ ] Hyperparameter grid search
- [ ] Feature importance analysis
- [ ] Model versioning system
- [ ] REST API for inference

## License
MIT

## Author
[MTAR RAYEN]

