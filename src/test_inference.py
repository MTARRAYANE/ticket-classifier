"""
Unit tests for the inference module

Tests the prediction functionality, error handling, and edge cases.
"""

import unittest
from inference import predict_ticket


class TestTicketPrediction(unittest.TestCase):
    """Test cases for ticket prediction"""
    
    def test_predict_valid_ticket(self):
        """Test prediction on valid ticket text"""
        result = predict_ticket("please reset my password")
        
        self.assertIsInstance(result, dict)
        self.assertIn("category", result)
        self.assertIn("confidence", result)
        self.assertIn("is_confident", result)
        
        # Check confidence is between 0 and 1
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)
    
    def test_predict_with_confidence_threshold(self):
        """Test confidence threshold filtering"""
        result = predict_ticket("my network is slow", confidence_threshold=0.5)
        self.assertIsInstance(result["is_confident"], bool)
    
    def test_predict_empty_string(self):
        """Test error handling for empty input"""
        with self.assertRaises(ValueError):
            predict_ticket("")
    
    def test_predict_whitespace_only(self):
        """Test error handling for whitespace-only input"""
        with self.assertRaises(ValueError):
            predict_ticket("   ")
    
    def test_predict_non_string_input(self):
        """Test error handling for non-string input"""
        with self.assertRaises(ValueError):
            predict_ticket(123)
    
    def test_multiple_predictions_consistent(self):
        """Test that same input produces same prediction"""
        text = "install software please"
        result1 = predict_ticket(text)
        result2 = predict_ticket(text)
        
        self.assertEqual(result1["category"], result2["category"])
        self.assertEqual(result1["confidence"], result2["confidence"])


if __name__ == "__main__":
    unittest.main()
