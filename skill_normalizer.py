"""
Skill Normalization Module
--------------------------
Takes a raw comma-separated skill string and returns a deduplicated list
of canonical skill names using alias mapping and multi-word phrase protection.
"""

import re
from typing import List

# ── Multi-word phrases to protect before splitting ──────────────────────────
MULTI_WORD_PHRASES = [
    "competitive programming",
    "data structures",
    "data structure",
    "data visualization",
    "deep learning",
    "feature engineering",
    "html css",
    "machine learning",
    "natural language processing",
    "node js",
    "ci cd",
    "power bi",
    "rest api",
    "spring boot",
    "ui ux",
]

# ── Canonical skill alias map ───────────────────────────────────────────────
# Keys are all possible raw (lowercased, cleaned) tokens; values are canonical names.
SKILL_ALIASES = {
    # Python ecosystem
    "python":            "python",
    "pyhton":            "python",      # common typo
    "pythn":             "python",
    "py":                "python",
    "python3":           "python",
    "pandas":            "pandas",
    "numpy":             "numpy",
    "scipy":             "scipy",
    "matplotlib":        "data_visualization",   # canonical → data_visualization
    "seaborn":           "data_visualization",
    "data_viz":          "data_visualization",
    "scikit-learn":      "machine_learning",
    "sklearn":           "machine_learning",      # canonical → machine_learning
    "scikit_learn":      "machine_learning",
    "flask":             "flask",
    "django":            "django",
    "fastapi":           "fastapi",

    # Machine Learning / AI
    "ml":                "machine_learning",
    "machine_learning":  "machine_learning",
    "machinelearning":   "machine_learning",
    "deep_learning":     "deep_learning",
    "deeplearning":      "deep_learning",
    "deep-learning":     "deep_learning",
    "dl":                "deep_learning",
    "nlp":               "natural_language_processing",
    "natural_language_processing": "natural_language_processing",
    "computer_vision":   "computer_vision",
    "cv":                "computer_vision",
    "tensorflow":        "tensorflow",
    "tf":                "tensorflow",
    "pytorch":           "pytorch",
    "torch":             "pytorch",
    "keras":             "keras",
    "feature_engineering": "feature_engineering",
    "bert":              "bert",
    "xgboost":           "xgboost",
    "regression":        "regression",
    "clustering":        "clustering",
    "statistics":        "statistics",

    # Data
    "sql":               "sql",
    "mysql":             "mysql",
    "postgresql":        "postgresql",
    "postgres":          "postgresql",
    "mongodb":           "mongodb",
    "mongo":             "mongodb",
    "data_structures":   "data_structures",
    "data_structure":    "data_structures",
    "dsa":               "data_structures",
    "data_visualization": "data_visualization",

    # Web / Frontend
    "html":              "html",
    "css":               "css",
    "html_css":          "html_css",
    "htmlcss":           "html_css",
    "javascript":        "javascript",
    "javascrpit":        "javascript",    # common typo
    "js":                "javascript",
    "typescript":        "typescript",
    "typescrpit":        "typescript",    # common typo
    "ts":                "typescript",
    "react":             "react",
    "reactjs":           "react",
    "reacts":            "react",         # common typo
    "react.js":          "react",
    "angular":           "angular",
    "vue":               "vue",
    "vuejs":             "vue",
    "vue.js":            "vue",
    "node_js":           "node_js",
    "nodejs":            "node_js",
    "node.js":           "node_js",
    "express":           "express",
    "expressjs":         "express",
    "graphql":           "graphql",
    "redux":             "redux",
    "tailwind":          "tailwind",
    "jest":              "jest",
    "ui_ux":             "ui_ux",

    # Java ecosystem
    "java":              "java",
    "spring_boot":       "spring_boot",
    "springboot":        "spring_boot",
    "spring":            "spring",

    # Mobile
    "android":           "android",
    "kotlin":            "kotlin",
    "swift":             "swift",
    "firebase":          "firebase",

    # DevOps / Cloud
    "docker":            "docker",
    "kubernetes":        "kubernetes",
    "kubernates":        "kubernetes",    # common typo
    "k8s":               "kubernetes",
    "microservices":     "microservices",
    "aws":               "aws",
    "azure":             "azure",
    "gcp":               "gcp",
    "ci_cd":             "ci_cd",
    "git":               "git",
    "github":            "github",
    "linux":             "linux",

    # Other languages
    "c":                 "c",
    "c++":               "cpp",
    "cpp":               "cpp",
    "c#":                "csharp",
    "csharp":            "csharp",
    "go":                "go",
    "golang":            "go",
    "rust":              "rust",
    "r":                 "r",
    "scala":             "scala",

    # BI / Analytics
    "power_bi":          "power_bi",
    "powerbi":           "power_bi",
    "tableau":           "tableau",
    "excel":             "excel",

    # Design
    "figma":             "figma",

    # Misc
    "rest_api":          "rest_api",
    "restapi":           "rest_api",
    "rest":              "rest_api",
    "competitive_programming": "competitive_programming",
    "cp":                "competitive_programming",
    "algorithms":        "algorithms",
    "algoritms":         "algorithms",    # common typo
    "algo":              "algorithms",
}


def normalize_skills(raw: str) -> List[str]:
    """
    Normalize a raw comma-separated skill string into canonical skill tokens.

    Steps
    -----
    1. Lowercase the entire string
    2. Protect multi-word phrases by replacing spaces with underscores
    3. Split on commas and strip whitespace
    4. Clean each token (collapse hyphens/spaces → underscores, strip junk)
    5. Map through SKILL_ALIASES; discard unknowns
    6. Deduplicate while preserving order

    Parameters
    ----------
    raw : str
        Comma-separated skill string, e.g. "Pyhton, MachineLearning, SQL"

    Returns
    -------
    list[str]
        Deduplicated list of canonical skill names.
    """
    # ── Step 1: Lowercase ───────────────────────────────────────────────
    lowered = raw.lower()
    print(f"  [1] Lowercased   → {lowered!r}")

    # ── Step 2: Protect multi-word phrases ──────────────────────────────
    protected = lowered
    for phrase in sorted(MULTI_WORD_PHRASES, key=len, reverse=True):
        # Match the phrase even if separated by hyphens or extra spaces
        pattern = re.escape(phrase).replace(r"\ ", r"[\s\-]+")
        protected = re.sub(pattern, phrase.replace(" ", "_"), protected)
    print(f"  [2] Protected    → {protected!r}")

    # ── Step 3: Split on commas and strip ───────────────────────────────
    tokens = [t.strip() for t in protected.split(",") if t.strip()]
    print(f"  [3] Split tokens → {tokens}")

    # ── Step 4: Clean each token ────────────────────────────────────────
    cleaned = []
    for tok in tokens:
        tok = re.sub(r"[\s\-/]+", "_", tok)  # spaces / hyphens / slashes → underscore
        tok = re.sub(r"[^a-z0-9_+#.]", "", tok)  # drop stray characters
        tok = tok.strip("_")
        if tok:
            cleaned.append(tok)
    print(f"  [4] Cleaned      → {cleaned}")

    # ── Step 5: Alias mapping (discard unknowns) ────────────────────────
    mapped = [SKILL_ALIASES[tok] for tok in cleaned if tok in SKILL_ALIASES]
    print(f"  [5] Alias-mapped → {mapped}")

    # ── Step 6: Deduplicate (preserve order) ────────────────────────────
    seen = set()
    deduped = []
    for skill in mapped:
        if skill not in seen:
            seen.add(skill)
            deduped.append(skill)
    print(f"  [6] Deduplicated → {deduped}")

    return deduped


# ── Demo / Test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_input = "Pyhton, MachineLearning, SQL, pandas, numpy, Deep-learning"
    print(f"Raw input: {test_input!r}\n")

    result = normalize_skills(test_input)

    print(f"\n✅ Final output: {result}")
    print(f"   Expected:     ['python', 'machine_learning', 'sql', 'pandas', 'numpy', 'deep_learning']")
    assert result == ["python", "machine_learning", "sql", "pandas", "numpy", "deep_learning"], \
        f"❌ Mismatch! Got {result}"
    print("   ✔ Assertion passed.")
