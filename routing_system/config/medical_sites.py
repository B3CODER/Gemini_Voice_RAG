"""
Medical Sites Configuration
Valid anatomical sites for Upper and Lower GIT procedures
"""

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

# Combine for easy access
ALL_SITES = {
    "Upper GIT": VALID_SITES_UPPER_GIT,
    "Lower GIT": VALID_SITES_LOWER_GIT
}
