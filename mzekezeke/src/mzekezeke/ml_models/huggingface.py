"""
HuggingFace Model Integration for Health Predictions

This module provides integration with HuggingFace models for:
- Health text analysis and interpretation
- Medical language understanding
- Symptom analysis and classification
- Health risk prediction using transformer models
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
import numpy as np
import requests
import json

from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    pipeline, TextClassificationPipeline
)
from huggingface_hub import InferenceClient
import torch

from .base import MLModelInterface

logger = logging.getLogger(__name__)


class HuggingFacePredictor(MLModelInterface):
    """
    HuggingFace model integration for health predictions.
    
    Supports both local models and HuggingFace Inference API.
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        use_api: bool = True,
        api_token: Optional[str] = None,
        device: str = "auto"
    ):
        super().__init__(f"HuggingFace-{model_name}", "1.0.0")
        
        self.model_name_hf = model_name
        self.use_api = use_api
        self.api_token = api_token or os.getenv("HUGGINGFACE_API_TOKEN")
        self.device = self._get_device(device)
        
        # Model components
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.inference_client = None
        
        # Supported tasks
        self.supported_tasks = {
            "health_risk_classification",
            "symptom_analysis", 
            "medical_text_analysis",
            "health_outcome_prediction",
            "medical_qa"
        }
        
        self._initialize_model()

    def _get_device(self, device: str) -> str:
        """Determine the best device for model inference."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps" 
            else:
                return "cpu"
        return device

    def _initialize_model(self) -> None:
        """Initialize the HuggingFace model or API client."""
        try:
            if self.use_api and self.api_token:
                # Initialize API client
                self.inference_client = InferenceClient(
                    model=self.model_name_hf,
                    token=self.api_token
                )
                logger.info(f"Initialized HuggingFace API client for {self.model_name_hf}")
            else:
                # Load local model
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_hf)
                
                # Try different model types based on the model name
                if "classification" in self.model_name_hf.lower():
                    self.model = AutoModelForSequenceClassification.from_pretrained(
                        self.model_name_hf
                    )
                    self.pipeline = pipeline(
                        "text-classification",
                        model=self.model,
                        tokenizer=self.tokenizer,
                        device=0 if self.device == "cuda" else -1
                    )
                else:
                    self.model = AutoModel.from_pretrained(self.model_name_hf)
                
                if self.device != "cpu":
                    self.model = self.model.to(self.device)
                
                logger.info(f"Loaded local model {self.model_name_hf} on {self.device}")
                
            self.is_trained = True
            
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFace model: {e}")
            self.is_trained = False

    def validate_input(self, data: List[Any]) -> bool:
        """Validate input data for HuggingFace models."""
        if not data:
            return False
        
        # Check if we have text data or can convert to text
        for item in data:
            if hasattr(item, 'value'):
                if not isinstance(item.value, (str, dict, list)):
                    continue
            elif not isinstance(item, (str, dict, list)):
                return False
        
        return True

    def predict(
        self, 
        data: List[Any], 
        prediction_horizon: timedelta, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Make health predictions using HuggingFace models.
        
        Args:
            data: Health data (can include text descriptions, symptoms, etc.)
            prediction_horizon: Time horizon for prediction
            context: Additional context (task type, patient info, etc.)
            
        Returns:
            Prediction results with confidence scores
        """
        if not self.validate_input(data):
            return {"error": "Invalid input data for HuggingFace prediction"}
        
        if not self.is_trained:
            return {"error": "Model not properly initialized"}
        
        task_type = context.get("task", "health_risk_classification")
        
        try:
            # Convert data to text format
            text_input = self._prepare_text_input(data, context)
            
            if self.use_api and self.inference_client:
                result = self._predict_with_api(text_input, task_type, context)
            else:
                result = self._predict_with_local_model(text_input, task_type, context)
            
            # Add metadata
            result.update({
                "model_used": self.model_name,
                "prediction_type": task_type,
                "timestamp": datetime.now(),
                "prediction_horizon": prediction_horizon,
                "confidence": self.calculate_prediction_confidence(data, result, context)
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {"error": f"Prediction failed: {str(e)}"}

    def _prepare_text_input(self, data: List[Any], context: Dict[str, Any]) -> str:
        """Convert health data to text format for model input."""
        text_parts = []
        
        # Add context information
        if "patient_age" in context:
            text_parts.append(f"Patient age: {context['patient_age']}")
        
        if "symptoms" in context:
            text_parts.append(f"Symptoms: {', '.join(context['symptoms'])}")
        
        # Process data points
        for item in data:
            if hasattr(item, 'value'):
                value = item.value
                metric_type = getattr(item, 'metric_type', 'measurement')
                
                if isinstance(value, str):
                    text_parts.append(value)
                elif isinstance(value, dict):
                    for key, val in value.items():
                        text_parts.append(f"{key}: {val}")
                elif isinstance(value, (int, float)):
                    text_parts.append(f"{metric_type}: {value}")
            else:
                text_parts.append(str(item))
        
        return " ".join(text_parts)

    def _predict_with_api(self, text_input: str, task_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction using HuggingFace Inference API."""
        try:
            if task_type == "health_risk_classification":
                response = self.inference_client.text_classification(text_input)
                return {
                    "predicted_value": response[0]["label"],
                    "classification_scores": {item["label"]: item["score"] for item in response}
                }
            
            elif task_type == "medical_qa":
                question = context.get("question", "What is the health assessment?")
                response = self.inference_client.question_answering(
                    question=question,
                    context=text_input
                )
                return {
                    "predicted_value": response["answer"],
                    "answer_score": response["score"]
                }
            
            else:
                # Generic text classification
                response = self.inference_client.text_classification(text_input)
                return {
                    "predicted_value": response[0]["label"],
                    "scores": response
                }
        
        except Exception as e:
            logger.error(f"API prediction failed: {e}")
            return {"error": f"API prediction failed: {str(e)}"}

    def _predict_with_local_model(self, text_input: str, task_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction using local HuggingFace model."""
        try:
            if self.pipeline:
                # Use pipeline for classification tasks
                results = self.pipeline(text_input)
                if isinstance(results, list):
                    results = results[0]
                
                return {
                    "predicted_value": results.get("label", "unknown"),
                    "classification_score": results.get("score", 0.0)
                }
            else:
                # Use tokenizer and model directly
                inputs = self.tokenizer(
                    text_input,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512
                )
                
                if self.device != "cpu":
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                # Process outputs based on model type
                if hasattr(outputs, "logits"):
                    logits = outputs.logits
                    probabilities = torch.softmax(logits, dim=-1)
                    predicted_class = torch.argmax(probabilities, dim=-1).item()
                    confidence = probabilities[0][predicted_class].item()
                    
                    return {
                        "predicted_value": predicted_class,
                        "confidence_score": confidence,
                        "all_probabilities": probabilities[0].cpu().numpy().tolist()
                    }
                else:
                    # For models without classification head
                    embeddings = outputs.last_hidden_state.mean(dim=1)
                    return {
                        "embeddings": embeddings.cpu().numpy().tolist(),
                        "embedding_size": embeddings.shape[-1]
                    }
        
        except Exception as e:
            logger.error(f"Local model prediction failed: {e}")
            return {"error": f"Local model prediction failed: {str(e)}"}

    def train(self, training_data: List[Any], labels: List[Any]) -> Dict[str, Any]:
        """
        Fine-tune the HuggingFace model (placeholder implementation).
        
        Note: Full fine-tuning implementation would require significant
        additional setup and resources.
        """
        logger.warning("Fine-tuning not implemented in this version")
        return {
            "status": "training_not_implemented",
            "message": "Use pre-trained models or implement fine-tuning separately"
        }

    def analyze_health_text(self, text: str, analysis_type: str = "general") -> Dict[str, Any]:
        """
        Analyze health-related text using the model.
        
        Args:
            text: Health text to analyze
            analysis_type: Type of analysis (symptoms, risk, outcome)
            
        Returns:
            Analysis results
        """
        context = {"task": "medical_text_analysis", "analysis_type": analysis_type}
        
        # Create a mock data point
        data = [type('MockData', (), {'value': text, 'metric_type': 'text'})()]
        
        return self.predict(data, timedelta(hours=24), context)

    def classify_health_risk(self, symptoms: List[str], patient_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify health risk based on symptoms and patient context.
        
        Args:
            symptoms: List of symptoms
            patient_context: Patient information (age, medical history, etc.)
            
        Returns:
            Risk classification results
        """
        # Prepare text input
        text_input = f"Symptoms: {', '.join(symptoms)}"
        if "age" in patient_context:
            text_input += f" Patient age: {patient_context['age']}"
        
        context = {"task": "health_risk_classification", "symptoms": symptoms}
        context.update(patient_context)
        
        data = [type('MockData', (), {'value': text_input, 'metric_type': 'symptoms'})()]
        
        return self.predict(data, timedelta(hours=24), context)

    def get_health_recommendations(self, health_data: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate health recommendations based on data analysis.
        """
        context["task"] = "medical_qa"
        context["question"] = "What health recommendations would you provide based on this data?"
        
        data = [type('MockData', (), {'value': health_data, 'metric_type': 'health_summary'})()]
        
        return self.predict(data, timedelta(days=7), context)

    def update_model(self, new_model_name: str) -> bool:
        """Update to a different HuggingFace model."""
        try:
            old_model_name = self.model_name_hf
            self.model_name_hf = new_model_name
            self._initialize_model()
            
            if self.is_trained:
                logger.info(f"Successfully updated from {old_model_name} to {new_model_name}")
                return True
            else:
                # Revert on failure
                self.model_name_hf = old_model_name
                self._initialize_model()
                return False
                
        except Exception as e:
            logger.error(f"Model update failed: {e}")
            return False 