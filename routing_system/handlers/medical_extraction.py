"""
Medical Extraction Handler
Extracts structured medical information from clinical queries
"""

import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Import configuration
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config.medical_sites import VALID_SITES_UPPER_GIT, VALID_SITES_LOWER_GIT, ALL_SITES

load_dotenv()

# Initialize the client
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


class MedicalExtractionHandler:
    """Handles medical information extraction from clinical queries"""
    
    @staticmethod
    def get_tool():
        """Returns the tool declaration for medical extraction"""
        return types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name="extract_medical_info",
                    description="Extracts structured medical information (organ, site, procedure, test type) from clinical queries about biopsies, procedures, and anatomical sites.",
                    parameters=types.Schema(
                        type="OBJECT",
                        properties={
                            "user_query": types.Schema(
                                type="STRING",
                                description="The clinical query describing a medical procedure (e.g., 'biopsy from antrum', 'polyp in sigmoid colon').",
                            ),
                        },
                        required=["user_query"],
                    ),
                )
            ]
        )
    
    @staticmethod
    def handle(user_query):
        """
        Extracts medical information from the user query.
        Returns structured JSON with organ, site, procedure, and test types.
        """
        prompt = f"""
You are an expert medical coding assistant. Your goal is to extract structured data from clinical queries with 100% accuracy based on strict constraints.

**Input Query:** "{user_query}"

**Your Task:**
Identify the following 4 fields from the query:
1.  **Organ**: MUST be either "Upper GIT" or "Lower GIT".
2.  **Site**: MUST be an EXACT match from the provided Valid Sites lists.
3.  **Procedure**: The clinical procedure (e.g., Biopsy, Polypectomy).
4.  **Test Type**: The lab test (e.g., Histopathology, Microbiology).

**Constraint: Valid Sites**
*   **Upper GIT**: {json.dumps(VALID_SITES_UPPER_GIT)}
*   **Lower GIT**: {json.dumps(VALID_SITES_LOWER_GIT)}

**Mapping Logic (Critical):**
*   **Synonyms**: Map vague terms to the most specific valid site.
    *   "Stomach" -> "Gastric" (unless a specific part like "Antrum" is named).
    *   "Gullet" -> "Oesophagus".
    *   "Lg Bowel" -> If unspecified, check context. If "sigmoid" mentioned -> "Sigmoid colon".
*   **Hierarchy**: If a specific site is mentioned (e.g., "Antrum"), use that. Do NOT use the generic "Gastric" if a more specific valid site exists.
*   **Organ Determination**: The Organ is determined SOLELY by which list the identified Site belongs to.

**Few-Shot Examples:**
*   Query: "Biopsy from antrum"
    -> Site: "Antrum" (Found in Upper GIT list)
    -> Organ: "Upper GIT"
    -> Procedure: "Biopsy"
    -> Test: "Histopathology"

*   Query: "Polyp in sigmoid"
    -> Site: "Sigmoid colon" (Closest match in Lower GIT list)
    -> Organ: "Lower GIT"
    -> Procedure: "Polypectomy"
    -> Test: "Histopathology"

*   Query: "Oesophageal sample for culture"
    -> Site: "Oesophagus"
    -> Organ: "Upper GIT"
    -> Procedure: "Biopsy" (implied by sample)
    -> Test: "Microbiology/Culture"

**Output Format:**
Return ONLY a valid JSON object.

{{
  "organ": "Upper GIT" | "Lower GIT",
  "site": "Exact String from Valid List",
  "procedure_types": ["Procedure 1", ...],
  "test_types": ["Test 1", ...]
}}
"""
        
        try:
            response = client.models.generate_content(
                model='gemini-2.0-flash',
                contents=prompt,
                config={
                    "temperature": 0.7,
                    "response_mime_type": "application/json"
                }
            )
            
            return json.loads(response.text)
        
        except Exception as e:
            return {"error": str(e)}
