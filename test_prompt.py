
import os
import json
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

PROMPT_TEMPLATE = """

You are an advanced AI assistant with deep expertise in medical terminology and clinical practices. Your primary function is to accurately analyze a user's medical query and identify all corresponding medical procedures and test types based on your internal knowledge.

**Instructions:**

1.  **Analyze the Query:** Carefully examine the user's query to understand the clinical context. Identify all explicit and implicit medical procedures and test types mentioned. Be aware that medical queries can be complex and may use synonyms or descriptive language rather than formal medical terms.
2.  **Identify Procedures:** Based on your analysis, determine the specific medical procedure(s) being performed. For example, a query like "Antrum biopsy label" clearly indicates the procedure is a "Biopsy".
3.  **Infer Test Types from Context:** Based on the identified procedure and the language in the query, infer the most likely diagnostic test(s) that would be ordered. For instance, a "Biopsy" specimen is almost always sent for "Histopathology". A query mentioning "gastric sample for culture" implies a "Gastric culture" test.
4.  **Handle Multiple Types:** A single query may refer to multiple procedures or test types. You must identify and list all relevant types. For example, if a query mentions both a "polyp removal" and a "biopsy," you should identify "Polypectomy" and "Biopsy" as the procedures.
5.  **Format the Output:** Present the results in a clear, structured format. Create two distinct lists: one for "Procedure Types" and one for "Test Types". If no relevant type is found for a category, state "None identified".

---

**TASK**

**User Query:**
`{{user_query}}`

**Output:**

*   **Procedure Types:**
    *   [List of identified procedure names]
*   **Test Types:**
    *   [List of identified test type names]

"""


def process_query(user_query: str):
    # Replace placeholders in the prompt template
    prompt = PROMPT_TEMPLATE
    prompt = prompt.replace("{{user_query}}", user_query)
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
        }
    )
    
    return response.text


if __name__ == "__main__":
     
    while True:
        user_query = input("Query: ")
        if user_query.lower() in ['quit', 'exit', 'q']:
            break
        
        if user_query.strip():
            print(process_query(user_query))
            print()
