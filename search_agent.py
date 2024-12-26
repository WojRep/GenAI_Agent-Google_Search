from fastapi import FastAPI, HTTPException, Depends, Security, Request
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from openai import AsyncOpenAI
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import asyncio
import re
import time
import sys
import os
import uuid
import yaml
import logging
import logging.config
from functools import lru_cache
from contextvars import ContextVar
import json
from typing import List, Optional
import asyncio
from search_optimizer import get_optimized_queries
from settings import get_settings 

# Load configuration
def load_config() -> Dict[str, Any]:
    with open('config.yaml', 'r') as file:
        return yaml.safe_load(file)

config = load_config()
settings = get_settings() 

# Setup logging
class SessionFilter(logging.Filter):
    def filter(self, record):
        record.session_id = session_id.get()
        return True

session_id: ContextVar[str] = ContextVar('session_id', default='NO_SESSION')

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Configure logging
logging.config.dictConfig(config['logging'])
logger = logging.getLogger('search_agent')

app = FastAPI(title="Search Agent API")

# Security
api_key_header = APIKeyHeader(name=settings.API_KEY_NAME)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header != settings.API_KEY:
        logger.warning("Invalid API key attempt", extra={'api_key': api_key_header})
        raise HTTPException(
            status_code=401,
            detail="Invalid API Key"
        )
    return api_key_header

# OpenAI client initialization
@lru_cache()
def get_openai_client():
    return AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

# Google Search client initialization
@lru_cache()
def get_search_client():
    return build("customsearch", "v1", developerKey=settings.GOOGLE_CSE_KEY)


# Request/Response Models
class SearchRequest(BaseModel):
    question: str

class SearchResponse(BaseModel):
    answer: str
    session_id: str



async def search(search_term: str) -> str:
    logger.debug(f"Starting search for term: {search_term}")
    try:
        service = get_search_client()
        optimized_queries = get_optimized_queries(search_term)
        
        # Log optimized queries
        logger.debug(f"Optimized queries generated: {json.dumps(optimized_queries, indent=2)}")
        
        all_results = []
        
        async def execute_single_search(query: str) -> List[str]:
            search_params = {
                'q': query,
                'cx': settings.GOOGLE_CSE_ID,
                'num': config['search']['results_per_query']
            }
            
            logger.debug(f"Executing Google search with parameters: {json.dumps(search_params, indent=2)}")
            
            loop = asyncio.get_event_loop()
            res = await loop.run_in_executor(
                None,
                lambda: service.cse().list(**search_params).execute()
            )
            
            # Log complete search results
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
        
        # Log final combined results
        unique_results = list(dict.fromkeys(all_results))
        final_results = "\n".join(unique_results)
        logger.debug(f"Final combined search results: {json.dumps(final_results, indent=2)}")
        
        return final_results if final_results else "No results found"
        
    except Exception as e:
        logger.error(f"Search error: {str(e)}", exc_info=True)
        return f"Error during search: {str(e)}"

async def generate_answer(client: AsyncOpenAI, prompt: str, search_results: List[str]) -> str:
    logger.debug("Starting answer generation with OpenAI")
    combined_results = "\n".join(search_results)
    full_prompt = f"Question: {prompt}\n\nSearch Results:\n{combined_results}\n\nPlease provide a comprehensive answer based on the search results."
    
    # Log the complete prompt being sent to OpenAI
    logger.debug(f"Full prompt being sent to OpenAI: {json.dumps(full_prompt, indent=2)}")
    
    response = await client.chat.completions.create(
        model=settings.MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that provides comprehensive answers based on search results."},
            {"role": "user", "content": full_prompt}
        ],
        max_tokens=settings.MAX_TOKENS,
        n=1,
        stop=None
    )
    
    # Log the complete response from OpenAI
    logger.debug(f"Complete OpenAI response: {json.dumps(response.dict(), indent=2)}")
    
    answer = response.choices[0].message.content.strip()
    logger.debug(f"Final extracted answer: {json.dumps(answer, indent=2)}")
    return answer

@app.post("/search", response_model=SearchResponse)
async def search_endpoint(
    request: SearchRequest,
    api_key: str = Depends(get_api_key)
):
    current_session_id = session_id.get()
    logger.info(f"Starting new search request - Session ID: {current_session_id}")
    logger.info(f"Search question: {request.question}")

    
    try:
        client = get_openai_client()

        
        logger.info("Initiating parallel search ...")
        search_results = await search(request.question)
        
        logger.info("Generating final answer with OpenAI")
        answer = await generate_answer(client, request.question, search_results)
        
        logger.info("Request completed successfully")
        return SearchResponse(answer=answer, session_id=current_session_id)
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting server on {config['api']['host']}:{config['api']['port']}")
    uvicorn.run(
        app, 
        host=config['api']['host'], 
        port=config['api']['port'],
        log_config=config['logging']
    )