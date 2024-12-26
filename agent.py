from typing import List, Optional, Dict, Any
import re
import json
import logging
from pydantic import BaseModel
from providers import get_ai_provider
from settings import config

logger = logging.getLogger('search_agent')

class AgentAction(BaseModel):
    action: str
    action_input: str
    thought: str

class Agent:
    def __init__(self, provider_name: Optional[str] = None, model: Optional[str] = None):
        self.provider = get_ai_provider(provider_name)
        self.model = model
        self.system_prompt = config['agent']['prompts']['system']
        
    def _extract_action(self, text: str) -> AgentAction:
        """Extract action, input and thought from AI response"""

        logger.debug(f"Extracting action from response: {text}")
        # Clean up the text
        text = text.strip()
        
        # More robust pattern matching
        thought_pattern = r"Thought:[\s]*(.+?)[\s]*(?=Action:)"
        action_pattern = r"Action:[\s]*(.+?)[\s]*(?=Action Input:)"
        input_pattern = r'Action Input:[\s]*"(.+?)"'
        
        thought_match = re.search(thought_pattern, text, re.DOTALL)
        action_match = re.search(action_pattern, text, re.DOTALL)
        input_match = re.search(input_pattern, text, re.DOTALL)

        logger.debug(f"Extracting action - Thought match: {thought_match}") 
        logger.debug(f"Extracting action - Action match: {action_match}")
        logger.debug(f"Extracting action - Input match: {input_match}")
        
        if not all([thought_match, action_match, input_match]):
            logger.warning(f"Failed to parse response: {text}")
            # Return a default "Response To Human" action
            return AgentAction(
                thought="Failed to parse the AI response properly",
                action="Response To Human",
                action_input="I apologize, but I encountered an error processing your request. Please try again."
            )
            
        return AgentAction(
            thought=thought_match.group(1).strip(),
            action=action_match.group(1).strip(),
            action_input=input_match.group(1).strip()
        )
    
    async def _get_ai_response(self, messages: List[Dict[str, str]], search_results: List[str]) -> str:
        """Get response from AI provider"""
        try:
            # Format messages properly
            prompt = messages[-1]["content"] if isinstance(messages[-1], dict) else messages[-1]
            response, _ = await self.provider.generate_answer(
                prompt,
                search_results,  # Przekazujemy search_results do providera
                model=self.model
            )
            
            # Check if response has the required components
            if not all(x in response for x in ['Thought:', 'Action:', 'Action Input:']):
                # If missing components, force the response to human format
                formatted_response = f'Thought: Converting direct response to proper format\nAction: Response To Human\nAction Input: "{response}"'
                logger.debug(f"Reformatted response: {formatted_response}")
                return formatted_response
            
            return response
                
        except Exception as e:
            logger.error(f"Error getting AI response: {str(e)}")
            return 'Thought: An error occurred while processing the request\nAction: Response To Human\nAction Input: "I encountered an error while processing your request. Please try again."'

    
    async def process_question(self, question: str, search_func) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question}
        ]
        
        # Get initial search results
        initial_results = await search_func(question)
        
        max_iterations = config.get('agent', {}).get('max_iterations', 5)
        iteration = 0
        
        while iteration < max_iterations:
            logger.debug(f"Agent iteration {iteration + 1}")
            # Get AI response with search results
            response = await self._get_ai_response(messages, [initial_results])

            logger.debug(f"AI Response: {response}")
            
            # Extract action
            try:
                action_data = self._extract_action(response)
            except ValueError as e:
                logger.error(f"Failed to parse AI response: {e}")
                return "Error: Failed to process your request properly"
            
            logger.info(f"Extracted action: {action_data.action}")
            logger.debug(f"Action thought: {action_data.thought}")
            logger.debug(f"Action input: {action_data.action_input}")
            
            # Handle different actions
            if action_data.action == "Search":
                observation = await search_func(action_data.action_input)
                messages.extend([
                    {"role": "assistant", "content": response},
                    {"role": "user", "content": f"Observation: {observation}"}
                ])
            
            elif action_data.action == "Response To Human":
                return action_data.action_input
            
            else:
                logger.warning(f"Unknown action: {action_data.action}")
                return f"Error: Unknown action '{action_data.action}'"
            
            iteration += 1
        
        return "Error: Maximum iterations reached without finding an answer"