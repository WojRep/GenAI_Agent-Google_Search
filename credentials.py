from typing import Dict, Optional
import os
from dotenv import load_dotenv
from functools import lru_cache

# Load .env file
load_dotenv()

class CredentialsManager:
    def __init__(self):
        self._credentials: Dict[str, Dict[str, str]] = {
            'openai': {
                'api_key': os.getenv('OPENAI_API_KEY', ''),
                'org_id': os.getenv('OPENAI_ORG_ID', '')
            },
            'anthropic': {
                'api_key': os.getenv('ANTHROPIC_API_KEY', '')
            },
            'ollama': {
                'host': os.getenv('OLLAMA_HOST', 'http://localhost:11434'),
                'api_key': os.getenv('OLLAMA_API_KEY', '')
            },
            'google': {
                'cse_key': os.getenv('GOOGLE_CSE_KEY', ''),
                'cse_id': os.getenv('GOOGLE_CSE_ID', '')
            }
        }

    def get_credentials(self, provider: str) -> Dict[str, str]:
        """Get credentials for specific provider"""
        return self._credentials.get(provider, {})
    
    def get_credential(self, provider: str, key: str) -> str:
        """Get specific credential for provider"""
        return self._credentials.get(provider, {}).get(key, '')
    
    def has_required_credentials(self, provider: str) -> bool:
        """Check if all required credentials for a provider are non-empty"""
        creds = self._credentials.get(provider, {})
        if provider == 'openai':
            return bool(creds.get('api_key'))
        elif provider == 'anthropic':
            return bool(creds.get('api_key'))
        elif provider == 'ollama':
            return bool(creds.get('host'))  # api_key is optional for Ollama
        elif provider == 'google':
            return bool(creds.get('cse_key')) and bool(creds.get('cse_id'))
        return False

    def is_provider_available(self, provider: str) -> bool:
        """Check if provider exists and has required credentials"""
        return provider in self._credentials and self.has_required_credentials(provider)

    def get_available_providers(self) -> Dict[str, bool]:
        """Get list of all providers and their availability status"""
        return {provider: self.has_required_credentials(provider) 
                for provider in self._credentials.keys()}

@lru_cache()
def get_credentials_manager() -> CredentialsManager:
    """Get singleton instance of CredentialsManager"""
    return CredentialsManager()
