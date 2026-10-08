"""
Configuration module for AI-Driven Resume Screening and Skill Validation System.
Loads configuration from environment variables and sets defaults with built-in .env parser fallback.
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Load .env file with fallback if python-dotenv is not installed
env_file = BASE_DIR / ".env"
try:
    from dotenv import load_dotenv
    load_dotenv(env_file)
except ImportError:
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip()
                    if k and k not in os.environ:
                        os.environ[k] = v

# Database
DATABASE_PATH = os.getenv("DATABASE_PATH", str(BASE_DIR / "database" / "screening_system.db"))
if not os.path.isabs(DATABASE_PATH):
    DATABASE_PATH = str(BASE_DIR / DATABASE_PATH)

# Models
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
SPACY_MODEL = os.getenv("SPACY_MODEL", "en_core_web_sm")

# File Upload Settings
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", 10))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS = set(os.getenv("ALLOWED_EXTENSIONS", "pdf,docx,txt").lower().split(","))

# Scoring Weights (Configurable, Transparent)
WEIGHT_SEMANTIC_SKILL_MATCH = float(os.getenv("WEIGHT_SEMANTIC_SKILL_MATCH", 0.35))
WEIGHT_SKILL_EVIDENCE = float(os.getenv("WEIGHT_SKILL_EVIDENCE", 0.25))
WEIGHT_EXPERIENCE_RELEVANCE = float(os.getenv("WEIGHT_EXPERIENCE_RELEVANCE", 0.20))
WEIGHT_EDUCATION_RELEVANCE = float(os.getenv("WEIGHT_EDUCATION_RELEVANCE", 0.10))
WEIGHT_PROJECTS_CERTS = float(os.getenv("WEIGHT_PROJECTS_CERTS", 0.10))

# Normalize weights to ensure they sum to 1.0
_TOTAL_WEIGHT = (
    WEIGHT_SEMANTIC_SKILL_MATCH
    + WEIGHT_SKILL_EVIDENCE
    + WEIGHT_EXPERIENCE_RELEVANCE
    + WEIGHT_EDUCATION_RELEVANCE
    + WEIGHT_PROJECTS_CERTS
)
if _TOTAL_WEIGHT > 0:
    SCORING_WEIGHTS = {
        "semantic_skill_match": WEIGHT_SEMANTIC_SKILL_MATCH / _TOTAL_WEIGHT,
        "skill_evidence": WEIGHT_SKILL_EVIDENCE / _TOTAL_WEIGHT,
        "experience_relevance": WEIGHT_EXPERIENCE_RELEVANCE / _TOTAL_WEIGHT,
        "education_relevance": WEIGHT_EDUCATION_RELEVANCE / _TOTAL_WEIGHT,
        "projects_certs": WEIGHT_PROJECTS_CERTS / _TOTAL_WEIGHT,
    }
else:
    SCORING_WEIGHTS = {
        "semantic_skill_match": 0.35,
        "skill_evidence": 0.25,
        "experience_relevance": 0.20,
        "education_relevance": 0.10,
        "projects_certs": 0.10,
    }

# Skill Matching Thresholds
SEMANTIC_MATCH_HIGH_THRESHOLD = 0.75
SEMANTIC_MATCH_PARTIAL_THRESHOLD = 0.55

# Evidence Confidence Levels & Weights
EVIDENCE_LEVEL_WEIGHTS = {
    0: 0.0,
    1: 0.30,
    2: 0.70,
    3: 1.00,
}

EVIDENCE_LEVEL_LABELS = {
    0: "Missing",
    1: "Low (Mentioned Only)",
    2: "Moderate (Supported by Project/Course)",
    3: "High (Substantial Experience/Project)",
}

# Recommendation Thresholds (on a 0-100 scale)
RECOMMENDATION_THRESHOLDS = {
    "HIGH_RELEVANCE": 80.0,
    "RELEVANT": 65.0,
    "MODERATE_RELEVANCE": 50.0,
    "LOW_RELEVANCE": 0.0,
}

# Anonymization
ANONYMIZATION_PREFIX = os.getenv("ANONYMIZATION_PREFIX", "ANON")
ENABLE_PII_MASKING = os.getenv("ENABLE_PII_MASKING", "true").lower() in ("true", "1", "yes")

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
