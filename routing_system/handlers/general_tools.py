"""
General Tools Handler
Handles API-based tools like weather, news, crypto, jokes, quotes
"""

import os
import requests
from typing import Any, Dict
from google.genai import types
from dotenv import load_dotenv

# Import configuration
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config.workflow_config import WORKFLOW_CONFIG,convert_workflow_to_user_tools

load_dotenv()


class GeneralToolsHandler:
    """Handles general API-based tools"""
    
    def __init__(self):
        # Convert workflow config to user tools format
        self.user_tools = convert_workflow_to_user_tools(WORKFLOW_CONFIG)
    
    def get_tools(self):
        """Returns all tool declarations for general tools"""
        function_declarations = []
        
        for tool_def in self.user_tools.values():
            function_declarations.append(
                types.FunctionDeclaration(
                    name=tool_def["tool_name"],
                    description=tool_def["description"],
                    parameters=types.Schema(
                        type=tool_def["parameters"]["type"].upper(),
                        properties={
                            param_name: types.Schema(
                                type=param_props["type"].upper(),
                                description=param_props.get("description", ""),
                            )
                            for param_name, param_props in tool_def["parameters"].get("properties", {}).items()
                        },
                        required=tool_def["parameters"].get("required", []),
                    ),
                )
            )
        
        return types.Tool(function_declarations=function_declarations)
    
    def handle(self, tool_name: str, **args) -> Dict[str, Any]:
        """
        Handles execution of general tools.
        
        Args:
            tool_name: Name of the tool to execute
            **args: Tool arguments
            
        Returns:
            Dictionary with tool result
        """
        tool_def = self.user_tools.get(tool_name)
        if not tool_def:
            return {"error": f"Tool '{tool_name}' not found"}
        
        tool_type = tool_def.get("type", "")
        
        if tool_type == "api":
            return self._call_api(tool_def, args)
        else:
            return {"error": f"Unsupported tool type: {tool_type}"}
    
    def _call_api(self, tool: Dict[str, Any], args: Dict[str, Any]) -> Any:
        """Execute API tool call"""
        print(f"[GENERAL_TOOLS] Calling API {tool['tool_name']} with args {args}")
        
        try:
            url = tool["api_url"].format(**args)
            method = tool.get("http_method", "GET").upper()
            
            if method == "GET":
                resp = requests.get(url, timeout=10)
            elif method == "POST":
                resp = requests.post(url, json=args, timeout=10)
            else:
                resp = requests.request(method, url, json=args, timeout=10)
            
            return resp.json()
        except Exception as exc:
            return {"error": str(exc)}
