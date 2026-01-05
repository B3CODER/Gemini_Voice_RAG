# Voice Agent Routing System

A modular routing system for voice-controlled applications with three integrated components:

1. **Navigation** - Page and website navigation
2. **Medical Extraction** - Clinical query information extraction
3. **General Tools** - Weather, news, crypto, jokes, quotes, and more

## 📁 Folder Structure

```
routing_system/
├── handlers/               # Core routing logic
│   ├── __init__.py
│   ├── master_router.py    # Main intent classifier and router
│   ├── navigation.py       # Navigation handler
│   ├── medical_extraction.py  # Medical info extraction
│   └── general_tools.py    # API tools (weather, news, etc.)
├── tools/                  # Tool declarations for voice agent
│   ├── __init__.py
│   └── tool_declarations.py
├── config/                 # Configuration files
│   ├── __init__.py
│   ├── medical_sites.py    # Valid GIT sites
│   └── workflow_config.py  # General tools config
├── main.py                 # Voice agent entry point
└── README.md              # This file
```

## 🚀 Quick Start

### Running the Voice Agent (Audio Only)

```bash
cd /home/desk0014/Desktop/Backup/Voice_agent_routing/routing_system
python3 main.py
```

**Note**: This is an audio-only voice agent. No video, camera, or screen capture functionality.

### Voice Commands Examples

**Navigation:**
- "Go to the home page"
- "Open Gemini"
- "Take me to settings"

**Medical Extraction:**
- "Biopsy from antrum"
- "Polyp in sigmoid colon"
- "Sample from oesophagus for culture"

**General Tools:**
- "What's the weather in Surat?"
- "Tell me a joke"
- "What's the bitcoin price?"
- "Give me the latest news"
- "Random quote"

## 🔧 Components

### 1. Master Router (`handlers/master_router.py`)

Classifies user intent and routes to appropriate handlers:
- Uses Gemini 2.0 Flash for intent classification
- Routes to: navigation, medical, general_tools, or general conversation
- Confidence threshold: 60%

### 2. Navigation Handler (`handlers/navigation.py`)

Handles page/website navigation:
- Available pages: home, gemini, profile, settings, dashboard
- Opens URLs in default browser
- Returns structured success/error responses

### 3. Medical Extraction Handler (`handlers/medical_extraction.py`)

Extracts structured medical information:
- **Organ**: Upper GIT / Lower GIT
- **Site**: Exact anatomical site from valid lists
- **Procedure Types**: Biopsy, Polypectomy, etc.
- **Test Types**: Histopathology, Microbiology, etc.

### 4. General Tools Handler (`handlers/general_tools.py`)

Handles API-based tools:
- **Weather**: Current weather for any location
- **Crypto**: Cryptocurrency prices (Bitcoin, Ethereum, etc.)
- **News**: Latest headlines by category
- **Jokes**: Random jokes
- **Quotes**: Motivational quotes

## 📋 Configuration

### Medical Sites (`config/medical_sites.py`)

Defines valid anatomical sites for Upper and Lower GIT procedures.

### Workflow Config (`config/workflow_config.py`)

Defines general tool integrations:
- API endpoints
- Request methods
- Parameters
- Descriptions

## 🛠️ Tool Declarations

All tools are declared in `tools/tool_declarations.py` and integrated into the voice agent configuration.

## 📊 Response Formats

### Navigation Response
```json
{
  "result": "Successfully navigated to home page (https://google.com)"
}
```

### Medical Extraction Response
```json
{
  "organ": "Upper GIT",
  "site": "Antrum",
  "procedure_types": ["Biopsy"],
  "test_types": ["Histopathology"]
}
```

### General Tools Response
```json
{
  "temperature": 25.5,
  "condition": "Partly Cloudy",
  "location": "Surat, India"
}
```

## 🧪 Testing

Test individual handlers:

```python
from handlers.navigation import NavigationHandler
from handlers.medical_extraction import MedicalExtractionHandler
from handlers.general_tools import GeneralToolsHandler

# Test navigation
nav = NavigationHandler()
result = nav.handle("home")

# Test medical extraction
med = MedicalExtractionHandler()
result = med.handle("biopsy from antrum")

# Test general tools
gen = GeneralToolsHandler()
result = gen.handle("get_weather_tool", location="Surat")
```

## 📝 Environment Setup

Required environment variables:
```bash
GEMINI_API_KEY=your_api_key_here
```

## 🔌 Integration

The system is designed to work with:
- Google Gemini Live API (audio streaming)
- PyAudio for real-time voice input/output
- Various public APIs (weather, crypto, news, etc.)

**Audio-Only Design**: This voice agent focuses purely on audio streaming without any video/camera/screen capture capabilities.

## 🎯 Key Features

- **Modular Design**: Each component is independent and reusable
- **Intelligent Routing**: Automatic intent classification
- **Structured Responses**: Consistent JSON responses
- **Domain-Specific Instructions**: Each handler has focused prompts
- **Easy Extension**: Add new tools by updating `workflow_config.py`

## 📄 License

This routing system is part of a voice agent project for medical and general assistance.
