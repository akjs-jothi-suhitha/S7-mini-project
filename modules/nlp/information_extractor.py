"""
Information Extractor Module.
Extracts structured candidate details (skills, education, experience, certifications, projects, tools)
from resume text to build a CandidateProfile.
"""

import re
from typing import List, Dict, Any, Optional
from modules.nlp.skill_extractor import SkillExtractor
from modules.nlp.text_preprocessor import TextPreprocessor
from modules.models.candidate import RawCandidateDoc, CandidateProfile
from modules.utils.logging_utils import get_logger

logger = get_logger(__name__)


class InformationExtractor:
    """Extracts structured entities and profile attributes from candidate resumes."""

    # Common degree patterns
    _DEGREE_PATTERNS = [
        r"\b(?:B\.?Tech|B\.?E\.?|B\.?Sc|B\.?S\.?|BCA|BBA|Bachelor of [A-Za-z\s]+)\b",
        r"\b(?:M\.?Tech|M\.?E\.?|M\.?Sc|M\.?S\.?|MCA|MBA|Master of [A-Za-z\s]+)\b",
        r"\b(?:Ph\.?D\.?|Doctor of Philosophy)\b",
        r"\b(?:Diploma in [A-Za-z\s]+)\b",
    ]

    # Year pattern (1990 - 2030)
    _YEAR_PATTERN = re.compile(r"\b(19[89][0-9]|20[0-3][0-9])\b")

    # Experience duration patterns
    _EXP_YEARS_PATTERNS = [
        re.compile(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)(?:\s+of)?\s+(?:experience|exp)", re.IGNORECASE),
        re.compile(r"(?:experience|exp)[\s:]+(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)", re.IGNORECASE),
        re.compile(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\s+in\b", re.IGNORECASE),
    ]

    @classmethod
    def extract_profile(cls, raw_doc: RawCandidateDoc) -> CandidateProfile:
        """
        Parses candidate text and returns a comprehensive CandidateProfile.

        Args:
            raw_doc: RawCandidateDoc instance with cleaned and anonymized text.

        Returns:
            CandidateProfile dataclass instance.
        """
        text = raw_doc.cleaned_text or raw_doc.raw_text
        sections = TextPreprocessor.extract_sections(text)

        # 1. Candidate Name (from PII detector if available, else anonymized)
        detected_names = raw_doc.pii_entities.get("names", [])
        candidate_name = detected_names[0] if detected_names else raw_doc.anonymized_id

        # 2. Extract Skills
        skills = SkillExtractor.extract_skills(text)

        # 3. Extract Education
        education = cls._extract_education(sections.get("education", text), text)

        # 4. Extract Experience
        exp_years, exp_entries = cls._extract_experience(sections.get("experience", text), text)

        # 5. Extract Certifications
        certifications = cls._extract_certifications(sections.get("certifications", text), text)

        # 6. Extract Projects
        projects = cls._extract_projects(sections.get("projects", text), text)

        # 7. Extract Technical Tools (subset of extracted skills categorized as Tools/DevOps/Databases/Frameworks)
        technical_tools = [
            s["normalized_skill"]
            for s in skills
            if s["category"] in ("Frameworks & Libraries", "Cloud & DevOps", "Databases & Storage")
        ]

        # 8. Keywords
        extracted_keywords = list(set([s["normalized_skill"] for s in skills]))

        return CandidateProfile(
            candidate_id=raw_doc.candidate_id,
            anonymized_id=raw_doc.anonymized_id,
            candidate_name=candidate_name,
            skills=skills,
            education=education,
            experience_years=exp_years,
            experience_entries=exp_entries,
            certifications=certifications,
            projects=projects,
            technical_tools=technical_tools,
            cleaned_text=raw_doc.cleaned_text,
            anonymized_text=raw_doc.anonymized_text,
            extracted_keywords=extracted_keywords,
        )

    @classmethod
    def _extract_education(cls, edu_text: str, full_text: str) -> List[Dict[str, str]]:
        """Extracts degrees, institutions, and graduation years."""
        education_records = []
        target_text = edu_text if edu_text else full_text
        lines = target_text.splitlines()

        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue

            found_degree = None
            for pattern in cls._DEGREE_PATTERNS:
                match = re.search(pattern, line_str, re.IGNORECASE)
                if match:
                    found_degree = match.group(0).strip()
                    break

            if found_degree:
                year_match = cls._YEAR_PATTERN.search(line_str)
                grad_year = year_match.group(0) if year_match else ""

                # Extract institution if present (e.g. from comma or at/from separator)
                inst = ""
                parts = re.split(r"[,|•\-–—]|(?:\bat\b|\bfrom\b)", line_str, flags=re.IGNORECASE)
                for part in parts:
                    p = part.strip()
                    if p and not re.search(cls._DEGREE_PATTERNS[0], p, re.IGNORECASE) and not cls._YEAR_PATTERN.search(p):
                        if any(kw in p.lower() for kw in ["university", "college", "institute", "school", "academy", "campus"]):
                            inst = p
                            break
                if not inst and len(parts) > 1:
                    inst = parts[1].strip()

                education_records.append({
                    "degree": found_degree,
                    "institution": inst or "Recognized Institution",
                    "year": grad_year,
                })

        if not education_records:
            # Fallback scan in entire text
            for pattern in cls._DEGREE_PATTERNS:
                for m in re.finditer(pattern, full_text, re.IGNORECASE):
                    education_records.append({
                        "degree": m.group(0).strip(),
                        "institution": "Recognized Institution",
                        "year": "",
                    })

        # Deduplicate
        unique_records = []
        seen = set()
        for rec in education_records:
            key = f"{rec['degree'].lower()}_{rec['year']}"
            if key not in seen:
                seen.add(key)
                unique_records.append(rec)

        return unique_records

    @classmethod
    def _extract_experience(cls, exp_text: str, full_text: str) -> tuple[float, List[str]]:
        """Extracts total estimated years of experience and work entry snippets."""
        years = 0.0

        # Check explicit mention of experience years
        for pattern in cls._EXP_YEARS_PATTERNS:
            match = pattern.search(full_text)
            if match:
                try:
                    years = float(match.group(1))
                    break
                except ValueError:
                    pass

        # If not explicitly stated, estimate from year ranges (e.g., 2019 - 2023)
        if years == 0.0:
            date_range_pattern = re.compile(r"\b(20[0-2][0-9]|199[0-9])\s*[-–—to]+\s*(20[0-2][0-9]|Present|Current)\b", re.IGNORECASE)
            total_months = 0
            for match in date_range_pattern.finditer(exp_text or full_text):
                start_year = int(match.group(1))
                end_str = match.group(2).lower()
                end_year = 2024 if "present" in end_str or "current" in end_str else int(end_str)
                diff = max(0, end_year - start_year)
                total_months += diff * 12
            if total_months > 0:
                years = round(total_months / 12.0, 1)

        # Extract experience bullet entries
        entries = []
        lines = (exp_text or full_text).splitlines()
        for line in lines:
            line_str = line.strip()
            if len(line_str) > 20 and any(kw in line_str.lower() for kw in ["developed", "implemented", "managed", "led", "designed", "engineered", "built", "maintained", "worked"]):
                entries.append(line_str)

        return years, entries[:8]

    @classmethod
    def _extract_certifications(cls, cert_text: str, full_text: str) -> List[str]:
        """Extracts candidate certifications."""
        cert_keywords = [
            "certified", "certification", "certificate", "aws", "azure", "gcp",
            "google cloud", "coursera", "udemy", "deeplearning.ai", "scrum master",
            "pmp", "cisco", "ccna", "comptia", "tensorflow developer", "oracle certified"
        ]
        certs = []
        target = cert_text if cert_text else full_text
        for line in target.splitlines():
            line_str = line.strip()
            if 10 <= len(line_str) <= 120 and any(kw in line_str.lower() for kw in cert_keywords):
                # Filter out raw headers
                if line_str.lower() not in ["certifications", "certificates", "courses"]:
                    certs.append(line_str)

        # Deduplicate
        unique_certs = list(dict.fromkeys(certs))
        return unique_certs[:6]

    @classmethod
    def _extract_projects(cls, proj_text: str, full_text: str) -> List[str]:
        """Extracts candidate project descriptions and titles."""
        projects = []
        target = proj_text if proj_text else full_text
        lines = target.splitlines()

        for line in lines:
            line_str = line.strip()
            if len(line_str) > 25 and any(kw in line_str.lower() for kw in ["project", "system", "application", "platform", "model", "pipeline", "built", "developed", "created", "designed"]):
                projects.append(line_str)

        unique_projs = list(dict.fromkeys(projects))
        return unique_projs[:6]
