"""
Configuration settings for ReelSense system.
"""
import os
from pathlib import Path

class Config:
    """Configuration for ReelSense system"""
    
    # Base paths
    # Assuming this file is in src/reelsense/config.py
    # BASE_DIR should be the root of the project (parent of src)
    # We want to go up 2 levels from this file's directory to get to src/reelsense -> src -> PS_1_ReelSense
    # Wait, __file__ is src/reelsense/config.py. Parent is src/reelsense. Parent.Parent is src.
    # Actually the previous code used relative paths '../data'.
    # If we run from src/main.py, '../data' implies data is at the same level as src.
    # Let's make it robust using pathlib.
    
    # Get the package directory
    PACKAGE_DIR = Path(__file__).parent.absolute()
    # Get the src directory
    SRC_DIR = PACKAGE_DIR.parent
    # Get the project root
    PROJECT_ROOT = SRC_DIR.parent
    
    # Data paths
    DATA_PATH = PROJECT_ROOT / 'data'
    OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'results'
    VIZ_DIR = PROJECT_ROOT / 'outputs' / 'visualizations'
    
    # Make sure directories exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    
    # Model parameters
    K = 10  # Number of recommendations
    N_TEST = 3  # Number of test ratings per user
    
    # Diversity parameters
    LAMBDA_MMR = 0.5  # Balance between relevance and diversity
    
    # Hybrid model weights
    HYBRID_WEIGHTS = {
        'popularity': 0.1,
        'user_cf': 0.25,
        'item_cf': 0.25,
        'svd': 0.25,
        'content': 0.15
    }
    
    # Evaluation
    EVAL_SAMPLE_USERS = 50  # Number of users to evaluate (set to None for all)
