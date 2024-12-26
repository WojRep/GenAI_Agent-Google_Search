from fastapi import FastAPI, HTTPException, Depends, Security, Request
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import asyncio
import logging
import logging.config
from contextvars import ContextVar
import json
import os
from settings import settings, config, credentials_manager
from providers import get_ai_provider, get_available_providers
from search_optimizer import get_optimized_queries

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Setup logging
class SessionFilter(logging.Filter):
    def filter(self, record):
        record.session_id = session_id.get()
        return True

session_id: ContextVar[str] = ContextVar('session_id', default='NO_SESSION')

# Configure logging
if 'logging' in config:
    logging.config.dictConfig(config['logging'])
else:
    # Basic logging configuration if not defined in config
    logging.basicConfig(level=logging.INFO)

logger = logging.getLogger('search_agent')

app = FastAPI(title="Search Agent API")

# Security
api_key_header = APIKeyHeader(name=settings.api_key_name)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header != settings.api_key:
        logger.warning("Invalid API key attempt", extra={'api_key': api_key_header})
        raise HTTPException(
            status_code=401,
            detail="Invalid API Key"
        )
    return api_key_header



class SearchRequest(BaseModel):
    question: str
    provider: Optional[str] = None
    model: Optional[str] = None

class SearchResponse(BaseModel):
    answer: str
    session_id: str
    provider: str
    model: str

# Google Search client initialization
def get_search_client():
    creds = credentials_manager.get_credentials('google')
    return build("customsearch", "v1", developerKey=creds['cse_key'])

async def search(search_term: str) -> str:
    logger.debug(f"Starting search for term: {search_term}")
    try:
        service = get_search_client()
        optimized_queries = get_optimized_queries(search_term)
        
        logger.debug(f"Optimized queries generated: {json.dumps(optimized_queries, indent=2)}")
        
        all_results = []
        
        async def execute_single_search(query: str) -> List[str]:
            creds = credentials_manager.get_credentials('google')
            search_params = {
                'q': query,
                'cx': creds['cse_id'],
                'num': config.get('search', {}).get('google', {}).get('results_per_query', 10)
            }
            
            logger.debug(f"Executing Google search with parameters: {json.dumps(search_params, indent=2)}")
            
            loop = asyncio.get_event_loop()
            res = await loop.run_in_executor(
                None,
                lambda: service.cse().list(**search_params).execute()
            )
            
            logger.debug(f"Raw Google search results: {json.dumps(res, indent=2)}")
            
            snippets = [result.get('snippet', '') for result in res.get('items', [])]
            logger.debug(f"Extracted snippets: {json.dumps(snippets, indent=2)}")
            
            return snippets

        # Execute keyword search
        logger.info(f"Executing keyword search with query: {optimized_queries['keyword_query']}")
        keyword_results = await execute_single_search(optimized_queries["keyword_query"])
        all_results.extend(keyword_results)
        
        # Execute context searches
        for query in optimized_queries["context_queries"]:
            logger.info(f"Executing context search with query: {query}")
            context_results = await execute_single_search(query)
            all_results.extend(context_results)
        
        unique_results = list(dict.fromkeys(all_results))
        final_results = "\n".join(unique_results)
        logger.debug(f"Final combined search results: {json.dumps(final_results, indent=2)}")
        
        return final_results if final_results else "No results found"
        
    except Exception as e:
        logger.error(f"Search error: {str(e)}", exc_info=True)
        return f"Error during search: {str(e)}"

@app.post("/search", response_model=SearchResponse)
async def search_endpoint(
    request: SearchRequest,
    api_key: str = Depends(get_api_key)
):
    current_session_id = session_id.get()
    logger.info(f"Starting new search request - Session ID: {current_session_id}")
    logger.info(f"Search question: {request.question}")
    
    try:
        # Get the AI provider
        provider = get_ai_provider(request.provider)
        provider_name = request.provider or config.get('providers', {}).get('default', 'openai')
        
        logger.info(f"Using AI provider: {provider_name}")
        
        # Perform search
        logger.info("Initiating search...")
        search_results = await search(request.question)
        
        # Generate answer using selected provider
        logger.info(f"Generating answer with {provider_name}")
        answer, model_used = await provider.generate_answer(
            request.question, 
            [search_results],
            model=request.model
        )
        
        logger.info(f"Request completed successfully using model: {model_used}")
        return SearchResponse(
            answer=answer,
            session_id=current_session_id,
            provider=provider_name,
            model=model_used
        )
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )

@app.get("/providers")
async def get_providers_info(
    api_key: str = Depends(get_api_key)
):
    """Get information about available providers and their models"""
    return get_available_providers()

if __name__ == "__main__":
    import uvicorn
    host = config.get('api', {}).get('host', '0.0.0.0')
    port = config.get('api', {}).get('port', 8000)
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_config=config.get('logging', None)
    )