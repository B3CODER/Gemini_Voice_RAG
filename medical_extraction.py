import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()

# Initialize the client
# Assuming the API key is set in the environment or handled by the system
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# --- 1. Define Valid Sites from Images ---
# These lists are transcribed directly from the provided UI images.

VALID_SITES_UPPER_GIT = [
    "Oesophagus",
    "Cardio-oesophageal junction",
    "Fundus",
    "Gastric",
    "Body",
    "Antrum",
    "Pylorus",
    "Duodenum 1st part",
    "Duodenum 2nd part",
    "Small Bowel Biopsy"
]

VALID_SITES_LOWER_GIT = [
    "Caecum",
    "Ascending colon",
    "Hepatic flexure",
    "Transverse colon",
    "Splenic flexure",
    "Descending colon",
    "Sigmoid colon",
    "Rectum",
    "Ileo-caecal valve",
    "Terminal ileum",
    "Random colon, no rectum",
    "Random colon with rectum"
]

# Combine for the prompt context
ALL_SITES = {
    "Upper GIT": VALID_SITES_UPPER_GIT,
    "Lower GIT": VALID_SITES_LOWER_GIT
}

def get_medical_tool():
    """
    Returns the tool declaration for medical extraction.
    This allows the voice agent to use medical extraction as a function call.
    """
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


def handle_medical_extraction(user_query):
    """
    Handler function that will be called by the voice agent.
    Returns the medical extraction result.
    """
    return extract_medical_info(user_query)


def extract_medical_info(user_query):
      
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
                "temperature": 0.7, # Low temperature for deterministic output
                "response_mime_type": "application/json"
            }
        )
        
        return json.loads(response.text)

    except Exception as e:
        return {"error": str(e)}

# --- Main Execution Block for Testing ---
if __name__ == "__main__":
    # test_queries = [
    #     "Take a biopsy from the antrum",
    #     "Polyp found in the sigmoid colon, remove it",
    #     "Sample from the oesophagus for culture",
    #     "Random colon biopsy",
    #     "stomach biopsy",
    #     "tissue from caecum"
    # ]

    while True:
        user_query = input("Query: ")
        if user_query.lower() in ['quit', 'exit', 'q']:
            break
        
        if user_query.strip():
            result = extract_medical_info(user_query)
            print(f"Result: {json.dumps(result, indent=2)}")
            print("-" * 30)
