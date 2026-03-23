"""
AILS - Artificial Intelligence Learning System
Created by Cherry Computer Ltd.

Package initialization for core AILS modules.
"""

__version__ = "1.0.0"
__author__ = "Cherry Computer Ltd."
__email__ = "contact@cherrycomputer.ltd"
__license__ = "MIT"
__description__ = "Artificial Intelligence Learning System Framework"

from src.data.scraper import AILSScraper, DynamicScraper
from src.data.database import AILSDatabaseManager
from src.data.preprocessor import AILSPreprocessor
from src.nlp.sentiment import SentimentAnalyzer
from src.models.neural_network import AILSNeuralNetwork
from src.ethics.bias_detector import AILSBiasDetector
from src.ethics.privacy import PrivacyPreserver

__all__ = [
    "AILSScraper",
    "DynamicScraper",
    "AILSDatabaseManager",
    "AILSPreprocessor",
    "SentimentAnalyzer",
    "AILSNeuralNetwork",
    "AILSBiasDetector",
    "PrivacyPreserver",
]
