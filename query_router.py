import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# Initialize the client
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# Import handlers
import navigation
import medical_extraction

def classify_query(user_query):
    """
    Classifies the user query to determine intent: navigation or medical extraction.
    Returns the classification and confidence.
    """
    prompt = f"""You are a query classification assistant. Analyze the user query and determine the intent.

**User Query:** "{user_query}"

**Classification Task:**
Determine if this query is:
1. **navigation** - User wants to navigate to a page/website (e.g., "go to home", "open settings", "visit Google")
2. **medical** - User is asking about medical procedures, biopsies, or anatomical sites (e.g., "biopsy from antrum", "polyp in sigmoid")
3. **general** - General conversation or unclear intent

**Output Format:**
Return ONLY a valid JSON object:
{{
  "intent": "navigation" | "medical" | "general",
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


def route_query(user_query):
    """
    Main routing function that classifies the query and routes to the appropriate handler.
    Returns structured response from the appropriate function.
    """
    # Classify the query
    classification = classify_query(user_query)
    intent = classification.get("intent", "general")
    confidence = classification.get("confidence", 0.0)
    
    print(f"\n[ROUTER] Intent: {intent} (confidence: {confidence:.2f})")
    
    # Route based on intent
    if intent == "navigation" and confidence > 0.6:
        # Extract page name from query (simple approach)
        # The navigation handler will do the intelligent matching
        result = {
            "intent": "navigation",
            "query": user_query,
            "action": "Routing to navigation handler..."
        }
        print(f"[ROUTER] Routing to navigation handler")
        return result
        
    elif intent == "medical" and confidence > 0.6:
        # Route to medical extraction
        print(f"[ROUTER] Routing to medical extraction")
        medical_result = medical_extraction.extract_medical_info(user_query)
        return {
            "intent": "medical",
            "query": user_query,
            "data": medical_result
        }
        
    else:
        # General conversation or low confidence
        return {
            "intent": "general",
            "query": user_query,
            "response": "I can help you with navigation (e.g., 'go to settings') or medical queries (e.g., 'biopsy from antrum'). Please ask a specific question."
        }


def get_query_router_tool():
    """
    Returns the tool declaration for the query router.
    This allows the voice agent to use the router as a function call.
    """
    return types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="route_query",
                description="Analyzes user queries and routes them to the appropriate handler (navigation or medical extraction).",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "user_query": types.Schema(
                            type="STRING",
                            description="The user's query to analyze and route.",
                        ),
                    },
                    required=["user_query"],
                ),
            )
        ]
    )


def handle_query_routing(user_query):
    """
    Handler function that will be called by the voice agent.
    Returns the routing result.
    """
    return route_query(user_query)


# Test the router
if __name__ == "__main__":
    test_queries = [
        "go to the home page",
        "biopsy from antrum",
        "open settings",
        "polyp in sigmoid colon",
        "what's the weather today?",
        "take me to Google",
        "sample from oesophagus for culture"
    ]
    
    print("=" * 60)
    print("TESTING QUERY ROUTER")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = route_query(query)
        print(f"Result: {json.dumps(result, indent=2)}")
        print("-" * 60)
