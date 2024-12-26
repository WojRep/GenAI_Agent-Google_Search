from pydantic import BaseModel
from functools import lru_cache
from typing import Dict, Any, Optional
import yaml
from dotenv import load_dotenv
import os
from credentials import get_credentials_manager

# Load .env file
load_dotenv()

class Settings(BaseModel):
    """Application settings"""
    # API Settings
    api_key_name: str = os.getenv("API_KEY_NAME", "X-Auth-Token")
    api_key: str = os.getenv("API_KEY", "")
    
    class Config:
        validate_assignment = True

@lru_cache()
def load_yaml_config() -> Dict[str, Any]:
    """Load YAML configuration"""
    try:
        with open('config.yaml', 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        return {}

@lru_cache()
def get_settings() -> Settings:
    """Get application settings"""
    return Settings()

# Initialize settings, config and credentials manager
settings = get_settings()
config = load_yaml_config()
credentials_manager = get_credentials_manager()