#!/usr/bin/env python3
"""
test-api-project Test Suite
"""

import unittest
import os
import sys
from unittest.mock import patch, MagicMock

# Test için main modülünü import et
sys.path.insert(0, os.path.dirname(__file__))

class TestTest_Api_Project(unittest.TestCase):
    
    def setUp(self):
        """Test setup"""
        # Mock environment variables
        self.env_patcher = patch.dict(os.environ, {
            'MODEL_NAME': 'microsoft/DialoGPT-medium',
            'PORT': '5000'
        })
        self.env_patcher.start()
    
    def tearDown(self):
        """Test cleanup"""
        self.env_patcher.stop()
    
    @patch('transformers.pipeline')
    def test_model_loading(self, mock_pipeline):
        """Model yükleme testi"""
        mock_pipeline.return_value = MagicMock()
        
        from main import Test_Api_ProjectAI
        ai = Test_Api_ProjectAI()
        
        self.assertIsNotNone(ai.model)
        mock_pipeline.assert_called_once()
    
    @patch('transformers.pipeline')
    def test_text_processing(self, mock_pipeline):
        """Metin işleme testi"""
        mock_model = MagicMock()
        mock_model.return_value = [{'generated_text': 'Test response'}]
        mock_pipeline.return_value = mock_model
        
        from main import Test_Api_ProjectAI
        ai = Test_Api_ProjectAI()
        
        result = ai.process("Test input")
        self.assertEqual(result, "Test response")
    
    def test_environment_variables(self):
        """Environment variable testi"""
        self.assertEqual(os.getenv('MODEL_NAME'), 'microsoft/DialoGPT-medium')
        self.assertEqual(os.getenv('PORT'), '5000')

if __name__ == "__main__":
    unittest.main()
