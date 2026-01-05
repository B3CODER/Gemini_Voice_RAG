from rapidfuzz import process, fuzz

# ---------------------------------------
# SITE LISTS
# ---------------------------------------

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
    "Terminal ileum",
    "Cecum",
    "Ascending colon",
    "Transverse colon",
    "Descending colon",
    "Sigmoid colon",
    "Rectum",
    "Anal canal"
]

ALL_SITES = VALID_SITES_UPPER_GIT + VALID_SITES_LOWER_GIT


# ---------------------------------------
# LOOKUP TABLE
# ---------------------------------------

SITE_MAPPING = {
    # Upper GIT
    "Oesophagus": {"specimen_type": "biopsy", "procedure_type": "endoscopy", "test_type": "histopathology"},
    "Cardio-oesophageal junction": {"specimen_type": "biopsy", "procedure_type": "endoscopy", "test_type": "histopathology"},
    "Fundus": {"specimen_type": "biopsy", "procedure_type": "endoscopy", "test_type": "histopathology"},
    "Gastric": {"specimen_type": "biopsy", "procedure_type": "endoscopy", "test_type": "histopathology"},
    "Body": {"specimen_type": "biopsy", "procedure_type": "endoscopy", "test_type": "histopathology"},
    "Antrum": {"specimen_type": "biopsy", "procedure_type": "endoscopy", "test_type": "histopathology"},
    "Pylorus": {"specimen_type": "biopsy", "procedure_type": "endoscopy", "test_type": "histopathology"},
    "Duodenum 1st part": {"specimen_type": "biopsy", "procedure_type": "endoscopy", "test_type": "histopathology"},
    "Duodenum 2nd part": {"specimen_type": "biopsy", "procedure_type": "endoscopy", "test_type": "histopathology"},
    "Small Bowel Biopsy": {"specimen_type": "biopsy", "procedure_type": "endoscopy", "test_type": "histopathology"},

    # Lower GIT
    "Terminal ileum": {"specimen_type": "biopsy", "procedure_type": "colonoscopy", "test_type": "histopathology"},
    "Cecum": {"specimen_type": "biopsy", "procedure_type": "colonoscopy", "test_type": "histopathology"},
    "Ascending colon": {"specimen_type": "biopsy", "procedure_type": "colonoscopy", "test_type": "histopathology"},
    "Transverse colon": {"specimen_type": "biopsy", "procedure_type": "colonoscopy", "test_type": "histopathology"},
    "Descending colon": {"specimen_type": "biopsy", "procedure_type": "colonoscopy", "test_type": "histopathology"},
    "Sigmoid colon": {"specimen_type": "biopsy", "procedure_type": "colonoscopy", "test_type": "histopathology"},
    "Rectum": {"specimen_type": "biopsy", "procedure_type": "colonoscopy", "test_type": "histopathology"},
    "Anal canal": {"specimen_type": "biopsy", "procedure_type": "colonoscopy", "test_type": "histopathology"}
}


# ---------------------------------------
# FUZZY SEARCH
# ---------------------------------------

def fuzzy_match_site(query, threshold=60):
    match = process.extractOne(query, ALL_SITES, scorer=fuzz.WRatio)
    if match and match[1] >= threshold:
        return match   # matched site name
    return None


# ---------------------------------------
# DETERMINE REGION (Upper or Lower GIT)
# ---------------------------------------

def get_git_region(site):
    if site in VALID_SITES_UPPER_GIT:
        return "Upper GIT"
    elif site in VALID_SITES_LOWER_GIT:
        return "Lower GIT"
    return "Unknown"


# ---------------------------------------
# MAIN PIPELINE FUNCTION
# ---------------------------------------

def get_final_output(query):
    matched_site = fuzzy_match_site(query)

    if not matched_site:
        return {"error": "No matching organ site found"}

    region = get_git_region(matched_site)
    mapping = SITE_MAPPING.get(matched_site, {})

    return {
        "site": matched_site,
        "git_region": region,
        "specimen_type": mapping.get("specimen_type"),
        "procedure_type": mapping.get("procedure_type"),
        "test_type": mapping.get("test_type")
    }


# ---------------------------------------
# TEST CASES
# ---------------------------------------

queries = [
    "print label for gastic antrum",
    "gatrc antrm",
    "sigmiod clon biopsy",
    "lower area rectum biopsy",
    "fundas sample",
    "duodnum two"
]

for q in queries:
    print("\nQuery:", q)
    print(get_final_output(q))
