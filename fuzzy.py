
from rapidfuzz import process, fuzz

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

def fuzzy_search(query, choices=VALID_SITES_UPPER_GIT, threshold=60):
    # returns (best_match, score)
    match = process.extractOne(
        query,
        choices,
        scorer=fuzz.WRatio  # good all-purpose fuzzy match
    )
    if match and match[1] >= threshold:
        return match  # (match_string, score, index)
    return None


# -------------------------------
# TEST CASES
# -------------------------------

queries = [
    "duodenum two"
]

print("\n=== FUZZY SEARCH TEST RESULTS ===\n")

for q in queries:
    result = fuzzy_search(q)
    if result:
        print(f"Query: {q:<20} → Match: {result[0]:<25} (score={result[1]})")


    else:
        print(f"Query: {q:<20} → No good match found")
