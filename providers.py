from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import aiohttp
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
import logging
import json
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
        
        logger.debug("OpenAI - Full prompt being sent: %s", full_prompt)
        
        try:
            request_data = {
                "model": selected_model,
                "messages": [
                    {"role": "system", "content": config['agent']['prompts']['providers']['openai']},
                    {"role": "user", "content": full_prompt}
                ],
                "max_tokens": self.config['max_tokens']
            }
            
            logger.debug("OpenAI - Request data: %s", json.dumps(request_data, indent=2))
            
            response = await self.client.chat.completions.create(**request_data)
            
            logger.debug("OpenAI - Full API response: %s", response.model_dump_json())
            
            response_content = response.choices[0].message.content.strip()
            logger.debug("OpenAI - Final response content: %s", response_content)
            
            return response_content, selected_model
            
        except Exception as e:
            logger.error("OpenAI error: %s", str(e))
            logger.debug("OpenAI - Full exception details:", exc_info=True)
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
        
        logger.debug("Anthropic - Full prompt being sent: %s", full_prompt)
        
        try:
            request_data = {
                "model": selected_model,
                "max_tokens": self.config['max_tokens'],
                "messages": [{
                    "role": "system",
                    "content": config['agent']['prompts']['providers']['anthropic']
                }, {
                    "role": "user", 
                    "content": full_prompt
                }]
            }
            
            logger.debug("Anthropic - Request data: %s", json.dumps(request_data, indent=2))
            
            response = await self.client.messages.create(**request_data)
            
            logger.debug("Anthropic - Full API response: %s", response.model_dump_json())
            
            response_content = response.content[0].text
            logger.debug("Anthropic - Final response content: %s", response_content)
            
            return response_content, selected_model
            
        except Exception as e:
            logger.error("Anthropic error: %s", str(e))
            logger.debug("Anthropic - Full exception details:", exc_info=True)
            raise

class OllamaProvider(AIProvider):
    def __init__(self):
        creds = credentials_manager.get_credentials('ollama')
        self.base_url = creds.get('host', 'http://localhost:11434')
        self.api_key = creds.get('api_key', '')
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
        
        logger.debug("Ollama - Full prompt being sent: %s", full_prompt)
        
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        
        async with aiohttp.ClientSession() as session:
            try:
                request_data = {
                    "model": selected_model,
                    "prompt": f"<system>{config['agent']['prompts']['providers']['ollama']}</system>\n\n{full_prompt}",
                    "stream": False,
                    "options": {
                        "num_predict": self.config['max_tokens']
                    }
                }
                
                logger.debug("Ollama - Request data: %s", json.dumps(request_data, indent=2))
                
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=request_data,
                    headers=headers
                ) as response:
                    if response.status != 200:
                        error_body = await response.text()
                        logger.debug("Ollama - Error response (status %s): %s", response.status, error_body)
                        raise Exception(f"Ollama request failed with status {response.status}")
                    
                    result = await response.json()
                    logger.debug("Ollama - Full API response: %s", json.dumps(result, indent=2))
                    
                    response_content = result['response']
                    logger.debug("Ollama - Final response content: %s", response_content)
                    
                    return response_content, selected_model
                    
            except Exception as e:
                logger.error("Ollama error: %s", str(e))
                logger.debug("Ollama - Full exception details:", exc_info=True)
                raise

class LMStudioProvider(AIProvider):
    def __init__(self):
        creds = credentials_manager.get_credentials('lmstudio')
        self.base_url = creds.get('host', 'http://localhost:1234/v1')
        self.api_key = creds.get('api_key', '')
        self.config = config['providers']['lmstudio']
    
    @property
    def is_available(self) -> bool:
        return bool(self.base_url)

    @property
    def available_models(self) -> List[str]:
        return self.config['models']['allowed']
    
    async def generate_answer(self, prompt: str, search_results: List[str], model: Optional[str] = None) -> tuple[str, str]:
        if not self.is_available:
            raise ValueError("LM Studio host not configured")
            
        selected_model = self.get_model(model)
            
        combined_results = "\n".join(search_results)
        full_prompt = f"Question: {prompt}\n\nSearch Results:\n{combined_results}\n\nPlease provide a comprehensive answer based on the search results."
        
        logger.debug("LMStudio - Full prompt being sent: %s", full_prompt)
        
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        
        async with aiohttp.ClientSession() as session:
            try:
                request_data = {
                    "model": selected_model,
                    "messages": [
                        {"role": "system", "content": config['agent']['prompts']['providers']['lmstudio']},
                        {"role": "user", "content": full_prompt}
                    ],
                    "max_tokens": self.config['max_tokens'],
                    "temperature": config['agent']['temperature'],
                    "top_p": config['agent']['top_p']
                }
                
                logger.debug("LMStudio - Request data: %s", json.dumps(request_data, indent=2))
                
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    json=request_data,
                    headers=headers
                ) as response:
                    if response.status != 200:
                        error_body = await response.text()
                        logger.debug("LMStudio - Error response (status %s): %s", response.status, error_body)
                        raise Exception(f"LM Studio request failed with status {response.status}")
                    
                    result = await response.json()
                    logger.debug("LMStudio - Full API response: %s", json.dumps(result, indent=2))
                    
                    response_content = result['choices'][0]['message']['content']
                    logger.debug("LMStudio - Final response content: %s", response_content)
                    
                    return response_content, selected_model
                    
            except Exception as e:
                logger.error("LM Studio error: %s", str(e))
                logger.debug("LMStudio - Full exception details:", exc_info=True)
                raise

def get_ai_provider(provider_name: Optional[str] = None) -> AIProvider:
    if provider_name is None:
        provider_name = config['providers']['default']
    
    providers = {
        'openai': OpenAIProvider,
        'anthropic': AnthropicProvider,
        'ollama': OllamaProvider,
        'lmstudio': LMStudioProvider
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
    try:
        provider = get_ai_provider(provider_name)
        return provider.available_models
    except ValueError:
        return []

def get_available_providers() -> Dict[str, Dict[str, Any]]:
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