"""
Tool Declarations
Centralizes all tool declarations for the voice agent
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from handlers.navigation import NavigationHandler
from handlers.medical_extraction import MedicalExtractionHandler
from handlers.general_tools import GeneralToolsHandler


def get_navigation_tool():
    """Get navigation tool declaration"""
    return NavigationHandler.get_tool()


def get_medical_tool():
    """Get medical extraction tool declaration"""
    return MedicalExtractionHandler.get_tool()


def get_general_tools():
    """Get general tools declarations"""
    handler = GeneralToolsHandler()
    return handler.get_tools()


def get_all_tools():
    """
    Get all tool declarations for the voice agent.
    Returns a list of Tool objects.
    """
    return [
        get_navigation_tool(),
        get_medical_tool(),
        get_general_tools(),
    ]
