"""
Master Router
Classifies user intent and routes to appropriate handler
"""

import os
import json
from dotenv import load_dotenv
from google import genai

# Import handlers
from .navigation import NavigationHandler
from .medical_extraction import MedicalExtractionHandler
from .general_tools import GeneralToolsHandler

load_dotenv()

# Initialize the client
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


class MasterRouter:
    """Master router that classifies queries and routes to appropriate handlers"""
    
    def __init__(self):
        self.navigation_handler = NavigationHandler()
        self.medical_handler = MedicalExtractionHandler()
        self.general_handler = GeneralToolsHandler()
    
    def classify_intent(self, user_query: str):
        """
        Classifies the user query to determine intent.
        Returns: {"intent": "navigation"|"medical"|"general_tools"|"general", "confidence": float}
        """
        prompt = f"""You are a query classification assistant. Analyze the user query and determine the intent.

**User Query:** "{user_query}"

**Classification Task:**
Determine if this query is:
1. **navigation** - User wants to navigate to a page/website (e.g., "go to home", "open settings", "visit Google")
2. **medical** - User is asking about medical procedures, biopsies, or anatomical sites (e.g., "biopsy from antrum", "polyp in sigmoid")
3. **general_tools** - User asks for weather, news, crypto prices, jokes, quotes (e.g., "what's the weather?", "tell me a joke", "bitcoin price")
4. **general** - General conversation or unclear intent

**Output Format:**
Return ONLY a valid JSON object:
{{
  "intent": "navigation" | "medical" | "general_tools" | "general",
  "confidence": 0.0 to 1.0
}}
"""
        
        try:
            response = client.models.generate_content(
                model='gemini-2.0-flash',
                contents=prompt,
                config={
                    "temperature": 0.3,
                    "response_mime_type": "application/json"
                }
            )
            return json.loads(response.text)
        except Exception as e:
            return {"intent": "general", "confidence": 0.0, "error": str(e)}
    
    def route(self, user_query: str, tool_name: str = None, **args):
        """
        Routes query to appropriate handler.
        
        Args:
            user_query: The user's query
            tool_name: Optional tool name for direct routing
            **args: Tool arguments
            
        Returns:
            Structured response from the appropriate handler
        """
        # If tool_name is provided, route directly
        if tool_name:
            if tool_name == "navigate_to_page":
                return self.navigation_handler.handle(**args)
            elif tool_name == "extract_medical_info":
                return self.medical_handler.handle(**args)
            else:
                # Assume it's a general tool
                return self.general_handler.handle(tool_name, **args)
        
        # Otherwise, classify and route
        classification = self.classify_intent(user_query)
        intent = classification.get("intent", "general")
        confidence = classification.get("confidence", 0.0)
        
        print(f"\n[MASTER_ROUTER] Intent: {intent} (confidence: {confidence:.2f})")
        
        if intent == "navigation" and confidence > 0.6:
            # Extract page name and route to navigation
            print(f"[MASTER_ROUTER] Routing to navigation handler")
            return {
                "intent": "navigation",
                "query": user_query,
                "message": "Please specify which page to navigate to."
            }
        
        elif intent == "medical" and confidence > 0.6:
            # Route to medical extraction
            print(f"[MASTER_ROUTER] Routing to medical extraction")
            result = self.medical_handler.handle(user_query)
            return {
                "intent": "medical",
                "query": user_query,
                "data": result
            }
        
        elif intent == "general_tools" and confidence > 0.6:
            # Route to general tools
            print(f"[MASTER_ROUTER] Routing to general tools")
            return {
                "intent": "general_tools",
                "query": user_query,
                "message": "Please specify which tool to use (weather, news, crypto, joke, quote)."
            }
        
        else:
            # General conversation
            return {
                "intent": "general",
                "query": user_query,
                "response": "I can help you with:\\n1. Navigation (e.g., 'go to settings')\\n2. Medical queries (e.g., 'biopsy from antrum')\\n3. General tools (weather, news, jokes, etc.)\\n\\nHow can I assist you?"
            }
