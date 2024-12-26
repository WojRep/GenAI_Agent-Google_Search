from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import aiohttp
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
import logging
from settings import config
from credentials import get_credentials_manager

logger = logging.getLogger('search_agent')
credentials_manager = get_credentials_manager()

class AIProvider(ABC):
    @abstractmethod
    async def generate_answer(self, prompt: str, search_results: List[str], model: Optional[str] = None) -> tuple[str, str]:
        """Returns tuple of (answer, model_used)"""
        pass

    @property
    @abstractmethod
    def is_available(self) -> bool:
        pass

    @property
    @abstractmethod
    def available_models(self) -> List[str]:
        pass

    def get_model(self, requested_model: Optional[str] = None) -> str:
        """Get appropriate model based on request and configuration"""
        if requested_model and requested_model in self.available_models:
            return requested_model
        return self.config['models']['default']

class OpenAIProvider(AIProvider):
    def __init__(self):
        creds = credentials_manager.get_credentials('openai')
        self.client = AsyncOpenAI(
            api_key=creds['api_key'],
            organization=creds.get('org_id')
        ) if creds['api_key'] else None
        self.config = config['providers']['openai']
    
    @property
    def is_available(self) -> bool:
        return bool(self.client)

    @property
    def available_models(self) -> List[str]:
        return self.config['models']['allowed']
        
    async def generate_answer(self, prompt: str, search_results: List[str], model: Optional[str] = None) -> tuple[str, str]:
        if not self.is_available:
            raise ValueError("OpenAI credentials not configured")
            
        selected_model = self.get_model(model)
        
        combined_results = "\n".join(search_results)
        full_prompt = f"Question: {prompt}\n\nSearch Results:\n{combined_results}\n\nPlease provide a comprehensive answer based on the search results."
        
        try:
            response = await self.client.chat.completions.create(
                model=selected_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides comprehensive answers based on search results."},
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=self.config['max_tokens']
            )
            return response.choices[0].message.content.strip(), selected_model
        except Exception as e:
            logger.error(f"OpenAI error: {str(e)}")
            raise

class AnthropicProvider(AIProvider):
    def __init__(self):
        creds = credentials_manager.get_credentials('anthropic')
        self.client = AsyncAnthropic(
            api_key=creds['api_key']
        ) if creds['api_key'] else None
        self.config = config['providers']['anthropic']
    
    @property
    def is_available(self) -> bool:
        return bool(self.client)

    @property
    def available_models(self) -> List[str]:
        return self.config['models']['allowed']
    
    async def generate_answer(self, prompt: str, search_results: List[str], model: Optional[str] = None) -> tuple[str, str]:
        if not self.is_available:
            raise ValueError("Anthropic credentials not configured")
            
        selected_model = self.get_model(model)
            
        combined_results = "\n".join(search_results)
        full_prompt = f"Question: {prompt}\n\nSearch Results:\n{combined_results}\n\nPlease provide a comprehensive answer based on the search results."
        
        try:
            response = await self.client.messages.create(
                model=selected_model,
                max_tokens=self.config['max_tokens'],
                messages=[{
                    "role": "user",
                    "content": full_prompt
                }]
            )
            return response.content[0].text, selected_model
        except Exception as e:
            logger.error(f"Anthropic error: {str(e)}")
            raise

class OllamaProvider(AIProvider):
    def __init__(self):
        creds = credentials_manager.get_credentials('ollama')
        self.base_url = creds.get('host', 'http://localhost:11434')
        self.api_key = creds.get('api_key', '')  # Optional for Ollama
        self.config = config['providers']['ollama']
    
    @property
    def is_available(self) -> bool:
        return bool(self.base_url)

    @property
    def available_models(self) -> List[str]:
        return self.config['models']['allowed']
    
    async def generate_answer(self, prompt: str, search_results: List[str], model: Optional[str] = None) -> tuple[str, str]:
        if not self.is_available:
            raise ValueError("Ollama host not configured")
            
        selected_model = self.get_model(model)
            
        combined_results = "\n".join(search_results)
        full_prompt = f"Question: {prompt}\n\nSearch Results:\n{combined_results}\n\nPlease provide a comprehensive answer based on the search results."
        
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": selected_model,
                        "prompt": full_prompt,
                        "stream": False,
                        "options": {
                            "num_predict": self.config['max_tokens']
                        }
                    },
                    headers=headers
                ) as response:
                    if response.status != 200:
                        raise Exception(f"Ollama request failed with status {response.status}")
                    
                    result = await response.json()
                    return result['response'], selected_model
                    
            except Exception as e:
                logger.error(f"Ollama error: {str(e)}")
                raise

def get_ai_provider(provider_name: Optional[str] = None) -> AIProvider:
    """Get AI provider instance with validation"""
    if provider_name is None:
        provider_name = config['providers']['default']
    
    providers = {
        'openai': OpenAIProvider,
        'anthropic': AnthropicProvider,
        'ollama': OllamaProvider
    }
    
    if provider_name not in providers:
        raise ValueError(f"Provider {provider_name} doesn't exist")
        
    provider_config = config['providers'].get(provider_name)
    if not provider_config or not provider_config.get('enabled', False):
        raise ValueError(f"Provider {provider_name} is not enabled")
    
    provider = providers[provider_name]()
    if not provider.is_available:
        raise ValueError(f"Provider {provider_name} is not properly configured")
    
    return provider

def get_provider_models(provider_name: str) -> List[str]:
    """Get list of available models for a provider"""
    try:
        provider = get_ai_provider(provider_name)
        return provider.available_models
    except ValueError:
        return []

def get_available_providers() -> Dict[str, Dict[str, Any]]:
    """Get list of all configured providers with their models and availability status"""
    results = {}
    for provider_name in config['providers'].keys():
        if provider_name != 'default':
            try:
                provider = get_ai_provider(provider_name)
                results[provider_name] = {
                    'available': provider.is_available,
                    'models': provider.available_models,
                    'default_model': provider.get_model()
                }
            except ValueError:
                results[provider_name] = {
                    'available': False,
                    'models': [],
                    'default_model': None
                }
    return results