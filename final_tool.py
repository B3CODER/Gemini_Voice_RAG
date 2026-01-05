import json
import os
import re
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ---------------------------------------------------------
# ENV + CLIENT SETUP
# ---------------------------------------------------------
load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in environment")

client = genai.Client(api_key=GEMINI_API_KEY)

MODEL_NAME = "gemini-2.0-flash"
SUMMARY_MODEL_NAME = "gemini-2.5-flash-lite"

# ---------------------------------------------------------
# SYSTEM INSTRUCTIONS
# ---------------------------------------------------------
SYSTEM_INSTRUCTION = """You are a tool-calling assistant.

You have access to several tools (functions). For each user query:
- For general queries like greetings ("hi", "hello"), introductions, or questions about your capabilities ("what can you do"), respond directly without calling any tools.
- Decide which tools are relevant to the user's query.
- If multiple tools are referred and all the tools parmeters are passed then only call the tools else when we have all the tools parameters then we call all the tools.
- Call tools via function_call when needed.
- If a tool requires parameters, try to infer them from the conversation; otherwise call it with empty/missing values and the backend will handle asking the user.
- If no tools are relevant to a specific question or task, then **always** call rag tool.

Do NOT fabricate tool outputs. Only refer to information that comes from tools or from the conversation.
""".strip()

SUMMARY_SYSTEM_INSTRUCTION = """
You are a summarizer for executed tool results.

Input:
- A JSON list named "tool_results".
- Each item has:
  - "name": tool name
  - "type": one of {"api", "prompt", "static"}
  - "content": the tool output

Rules:
1. For items where "type" == "static":
   - DO NOT change or rewrite the text.
   - DO NOT summarize it.
   - Repeat the content EXACTLY as provided. No added words.

2. For items where "type" == "api" or "prompt":
   - Present the information naturally and directly to the user.
   - If it's a joke, quote, story, or similar content: present it as-is without explanation.
   - If it's data or factual information: format it clearly for the user.
   - DO NOT add meta-commentary like "I received..." or "The API returned...".

3. Combine all results into ONE final answer:
   - Present all information in a natural, conversational way.
   - Separate different topics with a blank line.
   - Maintain the flow as if you're directly answering the user's question.

4. Output format:
   - Only natural-language text.
   - No JSON, no code blocks.
   - No unnecessary preambles or explanations about where the data came from.
""".strip()

MODEL_CONFIG = types.GenerateContentConfig(
    tools=[
        types.Tool(
            function_declarations=[]  # filled below
        )
    ],
    system_instruction=SYSTEM_INSTRUCTION,
)

SUMMARY_CONFIG = types.GenerateContentConfig(
    system_instruction=SUMMARY_SYSTEM_INSTRUCTION,
)

# ---------------------------------------------------------
# WORKFLOW CONFIG (FROM FRONTEND)
# ---------------------------------------------------------
# This config will be provided by the frontend in production.
# It contains all tool definitions in a standardized format that
# gets converted to USER_TOOLS format via convert_workflow_to_user_tools().
#
# The workflow config supports:
# - API tools: External API calls with URL templates
# - Prompt tools: LLM-based dynamic responses
# - Static tools: Templated or fixed text responses
# - RAG tools: Knowledge base retrieval
# ---------------------------------------------------------
WORKFLOW_CONFIG: Dict[str, Any] = {
    "workflow_id": "weather_and_food_workflow",
    "workflow_name": "Weather + Food Workflow",
    "description": "Example workflow with API, prompt, and static tools.",
    "integrations": [
        {
            "tool_id": "food_suggestion_tool",
            "tool_name": "Food Suggestion",
            "tool_description": (
                "Suggest good eating options for a user given their location and time of day."
            ),
            "when_to_use": "Use this when the user asks what to eat.",
            "type": "prompt",
            "response_prompt": (
                "You are a local food expert.\n"
                "User location: {{location}}\n"
                "Time of day: {{time_of_day}}\n"
                "User info: {{user_info}}\n\n"
                "Suggest 3–5 specific things they can eat right now.\n"
                "- Prefer local dishes and realistic options.\n"
                "- 1–2 sentences per item.\n"
                "Respond in bullet points."
            ),
            "input_parameters": [
                {
                    "name": "location",
                    "type": "string",
                    "description": "City / area where the user is. Example: 'Surat, India'.",
                    "required": True,
                },
                {
                    "name": "time_of_day",
                    "type": "string",
                    "description": "Time of day (breakfast, lunch, evening, late night...).",
                    "required": True,
                },
            ],
        },
        {
            "tool_id": "get_weather_tool",
            "tool_name": "Get Weather",
            "tool_description": "Get the current weather for a given location.",
            "when_to_use": "Call this tool when the user asks for weather details.",
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
                {
                    "name": "unit",
                    "type": "string",
                    "description": "Temperature unit: 'metric' (C) or 'imperial' (F).",
                    "required": False,
                    "enum": ["metric", "imperial"],
                    "default": "metric",
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
                {"name": "coin", "type": "string", "description": "Cryptocurrency ID", "required": True}
            ],
        },
        {
            "tool_id": "news_api",
            "tool_name": "News (Inshorts)",
            "tool_description": "Get latest news headlines (summary included).",
            "when_to_use": "User requests news by category.",
            "type": "api",
            "url": "https://inshortsapi.vercel.app/news?category={category}",
            "method": "GET",
            "input_parameters": [
                {"name": "category", "type": "string", "description": "News category", "required": True}
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
        {
            "tool_id": "cloth_combination",
            "tool_name": "Cloth Combination",
            "tool_description": "Give clothing suggestion.",
            "when_to_use": "User asks what to wear.",
            "type": "static",
            "response_text": "Okay, you look good in any clothes, but I suggest you wear light color clothes today.",
            "input_parameters": [],
        },
        {
            "tool_id": "schedule_meeting",
            "tool_name": "Schedule Meeting",
            "tool_description": "Schedules a meeting with the provided details.",
            "when_to_use": "User asks to schedule or create a meeting.",
            "type": "static",
            "response_prompt": "Okay, I have scheduled a {{duration_minutes}} minute meeting with {{attendee_email}} on '{{topic}}'.",
            "input_parameters": [
                {"name": "attendee_email", "type": "string", "description": "Attendee email", "required": True},
                {"name": "topic", "type": "string", "description": "Meeting topic", "required": True},
                {"name": "duration_minutes", "type": "integer", "description": "Duration in minutes", "required": True},
            ],
        },
        {
            "tool_id": "motivation_tool",
            "tool_name": "Motivation",
            "tool_description": "Give a motivational message using a provided topic.",
            "when_to_use": "User asks for motivation on a topic.",
            "type": "prompt",
            "response_prompt": "Give a short motivational message about: {{topic}}",
            "input_parameters": [
                {"name": "topic", "type": "string", "description": "Topic to motivate about", "required": True}
            ],
        },
        {
            "tool_id": "rag_tool",
            "tool_name": "RAG Tool",
            "tool_description": "Retrieves information from internal knowledge base using RAG.",
            "when_to_use": "When user asks questions about internal documents or knowledge base.",
            "type": "rag",
            "input_parameters": [
                {"name": "user_query", "type": "string", "description": "User's query", "required": True}
            ],
        },
    ],
}

# ---------------------------------------------------------
# WORKFLOW CONFIG TO USER_TOOLS CONVERTER
# ---------------------------------------------------------
def convert_workflow_to_user_tools(workflow_config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Convert workflow config format (from frontend) to USER_TOOLS format (for backend).
    
    Handles:
    - API tools: Extracts 'url' → 'api_url', preserves 'method' → 'http_method'
    - Prompt tools: Converts {{variable}} → {variable} in 'response_prompt' → 'prompt_template'
    - Static tools: Handles both 'response_text' and 'response_prompt' formats
    - RAG tools: Preserves type and parameters
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


# Convert workflow config to USER_TOOLS
USER_TOOLS = convert_workflow_to_user_tools(WORKFLOW_CONFIG)

# ---------------------------------------------------------
# BUILD FUNCTION DECLARATIONS FOR GEMINI
# ---------------------------------------------------------
FUNCTION_DECLS: List[Dict[str, Any]] = [
    {
        "name": tool_def["tool_name"],
        "description": tool_def["description"],
        "parameters": tool_def["parameters"],
    }
    for tool_def in USER_TOOLS.values()
]

# patch the tool declarations into MODEL_CONFIG
MODEL_CONFIG.tools[0].function_declarations = FUNCTION_DECLS

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def call_custom_api(tool: Dict[str, Any], args: Dict[str, Any]) -> Any:
    print(f"[DEBUG] Calling API {tool['tool_name']} with args {args}")
    url = tool["api_url"].format(**args)
    method = tool.get("http_method", "GET").upper()
    try:
        if method == "GET":
            resp = requests.get(url, timeout=10)
        elif method == "POST":
            resp = requests.post(url, json=args, timeout=10)
        else:
            resp = requests.request(method, url, json=args, timeout=10)
        return resp.json()
    except Exception as exc:
        return {"error": str(exc)}


def handle_prompt_tool(
    tool: Dict[str, Any],
    args: Dict[str, Any],
    *,
    client: genai.Client,
) -> str:
    prompt = tool["prompt_template"].format(**args)
    print(f"[DEBUG] Calling PROMPT tool='{tool['tool_name']}' with prompt: {prompt!r}")
    resp = client.models.generate_content(model=SUMMARY_MODEL_NAME, contents=prompt)
    return (resp.text or "").strip()

def parse_fc_args(fc: Any) -> Dict[str, Any]:
    return dict(getattr(fc, "args", {}))


def extract_value_from_history(param: str, history: List[types.Content]) -> Optional[str]:
    """
    Try to infer a value for param from user messages in history.
    Very simple heuristic: look for the param name in text and take the trailing substring.
    """
    param_lower = param.lower()
    for item in reversed(history):
        if getattr(item, "role", None) != "user":
            continue

        text_parts: List[str] = []
        for part in getattr(item, "parts", []):
            t = getattr(part, "text", None)
            if t:
                text_parts.append(t)

        content = " ".join(text_parts).lower()
        if not content:
            continue

        if param_lower in content:
            tail = content.split(param_lower, 1)[-1].strip()
            if tail:
                return tail
    return None


# ---------------------------------------------------------
# EXECUTE TOOL CALLS
# ---------------------------------------------------------
def execute_tool_calls(
    response: Any,
    history: List[types.Content],
    client: genai.Client,
    user_input: str
) -> Optional[Dict[str, Any]]:
    """
    Parses function_call parts from response, executes tools, and returns:

    - {"status": "pending_json", "payload": {tool_name: {missing, args}}}
    - {"status": "static_only", "text": "..."}  # all static tools, no second LLM
    - {"status": "mixed", "summary_items": [...]}  # API / prompt / static mixed
    - None if no tool_calls.
    """
    candidates = getattr(response, "candidates", [])
    if not candidates:
        return None

    model_content = getattr(candidates[0], "content", None)
    if not model_content:
        return None

    tool_calls: List[Any] = []
    for part in getattr(model_content, "parts", []):
        fc = getattr(part, "function_call", None)
        if fc:
            tool_calls.append(fc)

    if not tool_calls:
        return None

    pending_params: Dict[str, Dict[str, Any]] = {}
    static_texts: List[str] = []
    summary_items: List[Dict[str, Any]] = []
    all_static = True

    for fc in tool_calls:
        tool_name = getattr(fc, "name", "")
        args = parse_fc_args(fc)
        print(f"[MODEL_TOOL_CALL] name={tool_name} args={json.dumps(args, ensure_ascii=False)}")

        tool_def = USER_TOOLS.get(tool_name)
        if not tool_def:
            all_static = False
            continue

        tool_type = tool_def.get("type", "")
        if tool_type != "static":
            all_static = False

        required_params = tool_def["parameters"].get("required", [])
        final_args: Dict[str, Any] = {}
        missing: List[str] = []

        # fill required params
        for param in required_params:
            if args.get(param):
                final_args[param] = args[param]
                continue

            auto_value = extract_value_from_history(param, history)
            if auto_value:
                final_args[param] = auto_value
                continue

            final_args[param] = ""
            missing.append(param)

        if missing:
            pending_params[tool_name] = {"missing": missing, "args": final_args}
            continue

        # execute
        if tool_type == "api":
            result = call_custom_api(tool_def, final_args)
        elif tool_type == "prompt":
            result = handle_prompt_tool(tool_def, final_args, client=client)
        elif tool_type == "rag":
            user_query = user_input if user_input else ""

            # CALL YOUR SEPARATE RAG EXECUTOR
            from rag_executor import run_rag
            rag_result = run_rag(user_query)

            return {
                "status": "rag_mode",
                "result": rag_result
            }
        elif tool_type == "static":
            if "prompt_template" in tool_def:
                result = tool_def["prompt_template"].format(**final_args)
            else:
                result = tool_def.get("response_text", "")
            text = result if isinstance(result, str) else str(result)
            static_texts.append(text)
        else:
            result = {"error": "Unsupported tool type."}

        summary_items.append(
            {
                "name": tool_name,
                "type": tool_type,
                "content": result,
            }
        )

    # missing params → let frontend handle asking the user
    if pending_params:
        return {"status": "pending_json", "payload": pending_params}

    # all tools static -> just join static texts, no summarizer
    if tool_calls and all_static and static_texts:
        combined = "\n\n".join(static_texts)
        return {"status": "static_only", "text": combined}

    # mixed / dynamic tools
    if summary_items:
        return {"status": "mixed", "summary_items": summary_items}

    return None

def trim_history_blocks(history, max_users=5):
    blocks = []
    current_block = []

    for item in history:
        if getattr(item, "role", None) == "user":
            if current_block:
                blocks.append(current_block)
            current_block = [item]
        else:
            current_block.append(item)

    if current_block:
        blocks.append(current_block)

    # keep only last N user blocks
    blocks = blocks[-max_users:]

    # flatten blocks back to list
    new_history = []
    for block in blocks:
        new_history.extend(block)

    return new_history


def save_history_to_json(history, filename="chat_history.json"):
    cleaned = []

    for item in history:
        entry = {
            "role": item.role,
            "parts": []
        }

        for part in item.parts:

            # 1️⃣ If part contains a function call (Gemini)
            if hasattr(part, "function_call") and part.function_call:
                entry["parts"].append({
                    "function_call": {
                        "name": part.function_call.name,
                        "args": part.function_call.args or {}
                    }
                })

            # 2️⃣ If part contains normal text
            elif part.text:
                entry["parts"].append({
                    "text": part.text
                })

            # 3️⃣ Fallback for any unknown structure
            else:
                entry["parts"].append({
                    "text": str(part)
                })

        cleaned.append(entry)

    # Save formatted JSON
    with open(filename, "w") as f:
        json.dump(cleaned, f, indent=2)


# ---------------------------------------------------------
# CHATBOT LOOP
# ---------------------------------------------------------
def chatbot(client: genai.Client) -> None:
    print("Chatbot ready! Type 'exit' to quit.")
    history: List[types.Content] = []

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break

        user_content = types.Content(
            role="user",
            parts=[types.Part(text=user_input)],
        )
        history.append(user_content)
        save_history_to_json(history)

        # main model call (routing + possible text)
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=history,
            config=MODEL_CONFIG,
        )
        print("#" * 90)
        print(response)

        candidate_content = None
        candidates = getattr(response, "candidates", [])
        if candidates:
            candidate_content = getattr(candidates[0], "content", None)

        if candidate_content:
            history.append(candidate_content)
            save_history_to_json(history)
        elif response.text:
            history.append(types.Content(role="model", parts=[types.Part(text=response.text)]))
            save_history_to_json(history)

        # execute tools
        tool_results = execute_tool_calls(response, history, client, user_input)
        response_text = (response.text or "").strip()

        # no tool calls -> just echo model text
        if not tool_results:
            if response_text:
                print(response_text)
            history = trim_history_blocks(history, max_users=5)
            continue

        status = tool_results.get("status")

        # missing params: for now just show model text; frontend can use tool_results["payload"]
        if status == "pending_json":
            if response_text:
                print(response_text)
            history = trim_history_blocks(history, max_users=5)
            continue

        # all static tools: no second LLM
        if status == "static_only":
            static_text = (tool_results.get("text") or "").strip()
            if static_text:
                print(static_text)
                history.append(types.Content(role="model", parts=[types.Part(text=static_text)]))
                history = trim_history_blocks(history, max_users=5)
                continue
        
        if status== "rag_mode":
            print("----- RAG RESULT START -----")
            print(tool_results["result"])
            print("----- RAG RESULT END -----")

            history.append(types.Content(role="model", parts=[types.Part(text=str(tool_results["result"]))]))
            continue

        # mixed / dynamic tools: call summarizer
        if status == "mixed":
            summary_items = tool_results.get("summary_items", [])

            if summary_items:
                summary_prompt = (
                    f"tool_results = {json.dumps(summary_items, ensure_ascii=False, indent=2)}"
                )

                final_response = client.models.generate_content(
                    model=SUMMARY_MODEL_NAME,
                    contents=summary_prompt,
                    config=SUMMARY_CONFIG,
                )
                summary_text = (final_response.text or "").strip()
                if summary_text:
                    print("=" * 40)
                    print(summary_text)
                    history.append(types.Content(role="model", parts=[types.Part(text=summary_text)]))
                    save_history_to_json(history)

                    history = trim_history_blocks(history, max_users=5)
                    continue

        # fallback: show original model text
        if response_text:
            print(response_text)
        history = trim_history_blocks(history, max_users=5)


if __name__ == "__main__":
    chatbot(client)