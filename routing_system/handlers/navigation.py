"""
Navigation Handler
Handles page and website navigation
"""

from google.genai import types
import webbrowser

PAGES = {
    "home": "https://google.com",
    "gemini": "https://gemini.google.com",
    "profile": "/profile",
    "settings": "/settings",
    "dashboard": "/dashboard",
}


class NavigationHandler:
    """Handles navigation to different pages and websites"""
    
    @staticmethod
    def get_tool():
        """Returns the tool declaration for navigation"""
        return types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name="navigate_to_page",
                    description="Navigate to a specific page in the application.",
                    parameters=types.Schema(
                        type="OBJECT",
                        properties={
                            "page_name": types.Schema(
                                type="STRING",
                                description=f"The name of the page to navigate to. Available pages: {', '.join(PAGES.keys())}.",
                            ),
                        },
                        required=["page_name"],
                    ),
                )
            ]
        )
    
    @staticmethod
    def handle(page_name):
        """
        Handles the navigation logic.
        Returns the result of the navigation action.
        """
        normalized_name = page_name.lower().strip()
        if normalized_name in PAGES:
            url = PAGES[normalized_name]
            print(f"\n[NAVIGATION] Navigating to: {url}")
            
            # Actually open the URL in the default browser
            try:
                webbrowser.open(url)
                return {"result": f"Successfully navigated to {normalized_name} page ({url})"}
            except Exception as e:
                return {"error": f"Failed to open browser: {str(e)}"}
        else:
            print(f"\n[NAVIGATION] Failed to navigate. Page not found: {page_name}")
            return {"error": f"Page '{page_name}' not found. Available pages: {', '.join(PAGES.keys())}"}
