import os
import asyncio
from dotenv import load_dotenv

load_dotenv()
import base64
import io
import traceback
import json
import requests

import pyaudio

from google import genai
from google.genai import types

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

MODEL = "models/gemini-2.5-flash-native-audio-preview-09-2025"

client = genai.Client(
    http_options={"api_version": "v1beta"},
    api_key=os.environ.get("GEMINI_API_KEY"),
)

# ============================================================================
# UNIFIED TOOL DEFINITIONS
# ============================================================================
# Define tools with all configuration in ONE place!
# Each tool includes: declaration (for AI) + endpoint config (for execution)

UNIFIED_TOOLS = [
    {
        # Tool Declaration (what the AI sees)
        "declaration": types.FunctionDeclaration(
            name="create_user",
            description="Creates a new user in the system using POST method",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "name": types.Schema(type=types.Type.STRING, description="User's full name"),
                    "email": types.Schema(type=types.Type.STRING, description="User's email address"),
                    "age": types.Schema(type=types.Type.INTEGER, description="User's age"),
                },
                required=["name", "email"],
            ),
        ),
        # Endpoint Configuration (how to execute)
        "endpoint": {
            "method": "POST",
            "base_url": "https://jsonplaceholder.typicode.com/users",
            "id_field": None,
            "success_message": "User '{name}' created successfully"
        }
    },
     
    {
        # Tool Declaration for create_order
        "declaration": types.FunctionDeclaration(
            name="create_order",
            description="Creates a new order in the system using POST method",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "product": types.Schema(type=types.Type.STRING, description="Product name"),
                    "quantity": types.Schema(type=types.Type.INTEGER, description="Quantity to order"),
                    "price": types.Schema(type=types.Type.NUMBER, description="Price per unit (optional)"),
                },
                required=["product", "quantity"],
            ),
        ),
        # Endpoint Configuration
        "endpoint": {
            "method": "POST",
            "base_url": "https://jsonplaceholder.typicode.com/posts",
            "id_field": None,
            "success_message": "Order for '{product}' (quantity: {quantity}) created successfully"
        }
    },
     
    {
        "declaration": types.FunctionDeclaration(
            name="update_user",
            description="Updates an existing user in the system using PATCH method",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "user_id": types.Schema(type=types.Type.STRING, description="User ID to update"),
                    "name": types.Schema(type=types.Type.STRING, description="New name (optional)"),
                    "email": types.Schema(type=types.Type.STRING, description="New email (optional)"),
                    "age": types.Schema(type=types.Type.INTEGER, description="New age (optional)"),
                },
                required=["user_id"],
            ),
        ),
        "endpoint": {
            "method": "PATCH",
            "base_url": "https://jsonplaceholder.typicode.com/users",
            "id_field": "user_id",
            "success_message": "User {user_id} updated successfully"
        }
    },
    {
        "declaration": types.FunctionDeclaration(
            name="update_order",
            description="Updates an existing order in the system using PATCH method",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "order_id": types.Schema(type=types.Type.STRING, description="Order ID to update"),
                    "quantity": types.Schema(type=types.Type.INTEGER, description="New quantity (optional)"),
                    "status": types.Schema(type=types.Type.STRING, description="New status (optional)"),
                },
                required=["order_id"],
            ),
        ),
        "endpoint": {
            "method": "PATCH",
            "base_url": "https://jsonplaceholder.typicode.com/posts",
            "id_field": "order_id",
            "success_message": "Order {order_id} updated successfully"
        }
    },
]

# Build tools list for the AI (extract declarations)
tools = [types.Tool(function_declarations=[tool["declaration"]]) for tool in UNIFIED_TOOLS]

# Build endpoint mapping for execution (extract configs)
TOOL_ENDPOINTS = {tool["declaration"].name: tool["endpoint"] for tool in UNIFIED_TOOLS}


CONFIG = types.LiveConnectConfig(
    response_modalities=[
        "AUDIO",
    ],
    tools=tools,  # Add the tools to the config
    system_instruction="""You must ALWAYS respond in English only, regardless of what language the user speaks in.
    
You have access to the following tools:
1. create_user (POST): Creates a new user with name, email, and optionally age
2. create_order (POST): Creates a new order with product name, quantity, and optionally price
3. update_user (PATCH): Updates an existing user by user_id
4. update_order (PATCH): Updates an existing order by order_id

When the user asks to create or add something new, use the POST tools (create_user or create_order).
When the user asks to update or modify existing data, use the PATCH tools (update_user or update_order).

Always confirm what action you're taking before executing the tool.""",
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Alnilam")
        ),
        # language_code="en-US"
    ),
    context_window_compression=types.ContextWindowCompressionConfig(
        trigger_tokens=25600,
        sliding_window=types.SlidingWindow(target_tokens=12800),
    ),
    input_audio_transcription=types.AudioTranscriptionConfig(),
    output_audio_transcription=types.AudioTranscriptionConfig(),
)

pya = pyaudio.PyAudio()


def dynamic_api_call(function_name: str, **kwargs) -> dict:
    """
    Dynamic function that handles any POST or PATCH API call.
    
    Args:
        function_name: Name of the function/tool being called
        **kwargs: Parameters passed from the AI model
    
    Returns:
        dict: Response with success status, message, data, and method used
    """
    # Get the configuration for this function
    if function_name not in TOOL_ENDPOINTS:
        return {
            "success": False,
            "message": f"Unknown function: {function_name}",
            "method": "UNKNOWN"
        }
    
    config = TOOL_ENDPOINTS[function_name]
    method = config["method"]
    base_url = config["base_url"]
    id_field = config["id_field"]
    
    # Build the URL
    if method == "PATCH" and id_field:
        # For PATCH, append the ID to the URL
        resource_id = kwargs.pop(id_field, None)
        if not resource_id:
            return {
                "success": False,
                "message": f"Missing required field: {id_field}",
                "method": method
            }
        url = f"{base_url}/{resource_id}"
    else:
        # For POST, use the base URL
        url = base_url
    
    # Build the payload from all remaining kwargs
    payload = {k: v for k, v in kwargs.items() if v is not None}
    
    # Make the HTTP request
    try:
        if method == "POST":
            response = requests.post(url, json=payload, timeout=10)
        elif method == "PATCH":
            response = requests.patch(url, json=payload, timeout=10)
        else:
            return {
                "success": False,
                "message": f"Unsupported HTTP method: {method}",
                "method": method
            }
        
        response.raise_for_status()
        result = response.json()
        
        # Format the success message with actual values
        success_msg = config["success_message"].format(**kwargs)
        
        return {
            "success": True,
            "message": success_msg,
            "data": result,
            "method": method
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"API call failed: {str(e)}",
            "method": method
        }


# Dynamic handler mapping - automatically handles all tools in TOOL_ENDPOINTS
class DynamicFunctionHandler:
    """Dynamically routes function calls to the generic API handler"""
    
    def __getitem__(self, function_name):
        """Allow dictionary-style access"""
        if function_name in TOOL_ENDPOINTS:
            return lambda **kwargs: dynamic_api_call(function_name, **kwargs)
        return None
    
    def __contains__(self, function_name):
        """Support 'in' operator"""
        return function_name in TOOL_ENDPOINTS


# Use the dynamic handler instead of a static dictionary
FUNCTION_HANDLERS = DynamicFunctionHandler()


class AudioLoop:
    def __init__(self):
        self.audio_in_queue = None
        self.session = None

        self.send_text_task = None
        self.receive_audio_task = None
        self.receive_audio_task = None
        self.play_audio_task = None
        self.play_audio_task = None
        self.last_speaker = None
        self.ai_speaking = False

    async def send_text(self):
        while True:
            text = await asyncio.to_thread(
                input,
                "message > ",
            )
            if text.lower() == "q":
                break
            await self.session.send(input=text or ".", end_of_turn=True)

    async def listen_audio(self):
        mic_info = pya.get_default_input_device_info()
        self.audio_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )
        if __debug__:
            kwargs = {"exception_on_overflow": False}
        else:
            kwargs = {}
        while True:
            data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
            if not self.ai_speaking:
                await self.session.send(input={"data": data, "mime_type": "audio/pcm"})

    async def receive_audio(self):
        "Background task to reads from the websocket and write pcm chunks to the output queue"
        while True:
            turn = self.session.receive()
            async for response in turn:
                if data := response.data:
                    self.audio_in_queue.put_nowait(data)
                
                # Handle tool calls
                if tool_call := response.tool_call:
                    print(f"\n\n[Tool Call Detected: {tool_call.function_calls[0].name}]")
                    
                    function_responses = []
                    for fc in tool_call.function_calls:
                        function_name = fc.name
                        function_args = fc.args
                        function_id = fc.id  # Get the ID from the function call
                        
                        print(f"  Function: {function_name}")
                        print(f"  Arguments: {json.dumps(function_args, indent=2)}")
                        
                        # Execute the function
                        if function_name in FUNCTION_HANDLERS:
                            handler = FUNCTION_HANDLERS[function_name]
                            try:
                                result = handler(**function_args)
                                print(f"  Result: {result['message']}")
                                print(f"  Method Used: {result['method']}\n")
                                
                                # Create function response with required id field
                                function_responses.append(
                                    types.FunctionResponse(
                                        name=function_name,
                                        response=result,
                                        id=function_id  # Include the ID
                                    )
                                )
                            except Exception as e:
                                print(f"  Error executing {function_name}: {str(e)}\n")
                                function_responses.append(
                                    types.FunctionResponse(
                                        name=function_name,
                                        response={
                                            "success": False,
                                            "message": f"Error: {str(e)}"
                                        },
                                        id=function_id  # Include the ID even for errors
                                    )
                                )
                        else:
                            print(f"  Unknown function: {function_name}\n")
                    
                    # Send function responses back to the model using send_tool_response
                    if function_responses:
                        await self.session.send_tool_response(
                            function_responses=function_responses
                        )
                
                # Handle transcriptions
                if (
                    server_content := response.server_content
                ) and server_content.input_transcription:
                    if self.last_speaker != "User":
                        print("\nUser: ", end="")
                        self.last_speaker = "User"
                    print(server_content.input_transcription.text, end="", flush=True)
                if (
                    server_content := response.server_content
                ) and server_content.output_transcription:
                    if self.last_speaker != "Model":
                        print("\nModel: ", end="")
                        self.last_speaker = "Model"
                    print(server_content.output_transcription.text, end="", flush=True)

            # If you interrupt the model, it sends a turn_complete.
            # For interruptions to work, we need to stop playback.
            # So empty out the audio queue because it may have loaded
            # much more audio than has played yet.
            while not self.audio_in_queue.empty():
                self.audio_in_queue.get_nowait()

    async def play_audio(self):
        stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        while True:
            bytestream = await self.audio_in_queue.get()
            self.ai_speaking = True
            await asyncio.to_thread(stream.write, bytestream)
            self.ai_speaking = False

    async def run(self):
        try:
            async with (
                client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session

                self.audio_in_queue = asyncio.Queue()

                send_text_task = tg.create_task(self.send_text())
                tg.create_task(self.listen_audio())
                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())

                await send_text_task
                raise asyncio.CancelledError("User requested exit")

        except asyncio.CancelledError:
            pass
        except ExceptionGroup as EG:
            self.audio_stream.close()
            traceback.print_exception(EG)


if __name__ == "__main__":
    main = AudioLoop()
    asyncio.run(main.run())