"""
Workflow Configuration for General Tools
Contains configuration for API, prompt, and static tools
"""

import re
from typing import Any, Dict, List

# Simplified workflow config - focusing on commonly used tools
# F tool in the original final_tool.py had RAG which we'll exclude from voice agent
WORKFLOW_CONFIG: Dict[str, Any] = {
    "workflow_id": "general_tools_workflow",
    "workflow_name": "General Tools Workflow",
    "description": "Weather, news, jokes, quotes, and other general tools.",
    "integrations": [
        {
            "tool_id": "get_weather_tool",
            "tool_name": "Get Weather",
            "tool_description": "Get the current weather for a given location.",
            "when_to_use": "Call this tool when the user asks for weather details.",
            "type": "api",
            "url": "https://api.open-meteo.com/v1/forecast",
            "method": "GET",
            "geocoding_url": "https://geocoding-api.open-meteo.com/v1/search",
            "input_parameters": [
                {
                    "name": "location",
                    "type": "string",
                    "description": "City or place to get weather for. Example: 'Surat, India'.",
                    "required": True,
                },
            ],
        },
        {
            "tool_id": "crypto_api",
            "tool_name": "Crypto Price",
            "tool_description": "Get current price of a cryptocurrency.",
            "when_to_use": "User asks for crypto prices.",
            "type": "api",
            "url": "https://api.coingecko.com/api/v3/simple/price?ids={coin}&vs_currencies=usd",
            "method": "GET",
            "input_parameters": [
                {"name": "coin", "type": "string", "description": "Cryptocurrency ID (e.g., bitcoin, ethereum)", "required": True}
            ],
        },
        {
            "tool_id": "news_api",
            "tool_name": "News",
            "tool_description": "Get latest news headlines.",
            "when_to_use": "User requests news by category.",
            "type": "api",
            "url": "https://inshortsapi.vercel.app/news?category={category}",
            "method": "GET",
            "input_parameters": [
                {"name": "category", "type": "string", "description": "News category (e.g., technology, sports, business)", "required": True}
            ],
        },
        {
            "tool_id": "quote_api",
            "tool_name": "Random Quote",
            "tool_description": "Get a random motivational or inspirational quote.",
            "when_to_use": "User asks for a quote.",
            "type": "api",
            "url": "https://api.quotable.io/random",
            "method": "GET",
            "input_parameters": [],
        },
        {
            "tool_id": "joke_api",
            "tool_name": "Random Joke",
            "tool_description": "Get a random joke.",
            "when_to_use": "User asks for a joke.",
            "type": "api",
            "url": "https://official-joke-api.appspot.com/random_joke",
            "method": "GET",
            "input_parameters": [],
        },
    ],
}


def convert_workflow_to_user_tools(workflow_config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Convert workflow config format to USER_TOOLS format.
    
    Handles:
    - API tools: Extracts 'url' → 'api_url', preserves 'method' → 'http_method'
    - Prompt tools: Converts {{variable}} → {variable} in 'response_prompt' → 'prompt_template'
    - Static tools: Handles both 'response_text' and 'response_prompt' formats
    - Parameters: Converts 'input_parameters' list to JSON Schema format
    
    Args:
        workflow_config: Dictionary containing workflow configuration with integrations list
        
    Returns:
        Dictionary in USER_TOOLS format ready for tool execution
    """
    user_tools: Dict[str, Dict[str, Any]] = {}
    
    integrations = workflow_config.get("integrations", [])
    
    for integration in integrations:
        # Extract tool identifier
        tool_id = integration.get("tool_id", "")
        tool_name = integration.get("tool_name", "")
        
        # Use tool_id as the key, or sanitize tool_name if tool_id is missing
        tool_key = tool_id if tool_id else tool_name.lower().replace(" ", "_")
        
        # Determine tool type
        tool_type = integration.get("type", "")
        if not tool_type:
            # Infer type from presence of url or response_text/response_prompt
            if integration.get("url"):
                tool_type = "api"
            elif integration.get("response_text"):
                tool_type = "static"
            elif integration.get("response_prompt"):
                tool_type = "prompt"
            else:
                tool_type = "static"
        
        # Build parameters schema
        parameters_schema: Dict[str, Any] = {
            "type": "object",
            "properties": {},
        }
        required_params: List[str] = []
        
        input_params = integration.get("input_parameters", [])
        for param in input_params:
            param_name = param.get("name", "")
            param_type = param.get("type", "string")
            param_desc = param.get("description", "")
            
            properties: Dict[str, Any] = {"type": param_type}
            if param_desc:
                properties["description"] = param_desc
            if "enum" in param:
                properties["enum"] = param["enum"]
            if "default" in param:
                properties["default"] = param["default"]
                
            parameters_schema["properties"][param_name] = properties
            
            if param.get("required", False):
                required_params.append(param_name)
        
        if required_params:
            parameters_schema["required"] = required_params
        
        # Get description
        description = (
            integration.get("tool_description") or 
            integration.get("when_to_use") or 
            tool_name
        )
        
        # Build tool entry
        tool_entry: Dict[str, Any] = {
            "tool_name": tool_key,
            "type": tool_type,
            "description": description,
            "parameters": parameters_schema,
        }
        
        # Add type-specific fields
        if tool_type == "api":
            api_url = integration.get("url", "")
            tool_entry["api_url"] = api_url
            
            # Store method if provided
            if integration.get("method"):
                tool_entry["http_method"] = integration["method"]
            
            # Store geocoding_url if provided
            if integration.get("geocoding_url"):
                tool_entry["geocoding_url"] = integration["geocoding_url"]
                
        elif tool_type == "prompt":
            # Convert {{variable}} to {variable} format
            response_prompt = integration.get("response_prompt", "")
            prompt_template = re.sub(r'\{\{(\w+)\}\}', r'{\1}', response_prompt)
            tool_entry["prompt_template"] = prompt_template
            
        elif tool_type == "static":
            if "response_text" in integration:
                tool_entry["response_text"] = integration["response_text"]
            elif "response_prompt" in integration:
                # Convert {{variable}} to {variable} format
                response_prompt = integration.get("response_prompt", "")
                prompt_template = re.sub(r'\{\{(\w+)\}\}', r'{\1}', response_prompt)
                tool_entry["prompt_template"] = prompt_template
        
        user_tools[tool_key] = tool_entry
    
    return user_tools
