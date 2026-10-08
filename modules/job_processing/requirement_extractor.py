"""
Requirement Extractor Module.
Extracts structured requirements (required skills, preferred skills, education, experience, tools, responsibilities)
from job description text.
"""

import re
import uuid
from typing import List, Dict, Any, Optional
from modules.nlp.skill_extractor import SkillExtractor
from modules.nlp.skill_taxonomy import SkillTaxonomy
from modules.models.job import JobRequirements
from modules.utils.logging_utils import get_logger

logger = get_logger(__name__)


class RequirementExtractor:
    """Extracts structured requirements and skills from job descriptions."""

    _EXP_PATTERN = re.compile(
        r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)(?:\s+of)?\s+(?:experience|exp|relevant experience)",
        re.IGNORECASE,
    )

    _DEGREE_PATTERNS = [
        r"\b(?:Bachelor'?s?|B\.?Tech|B\.?E\.?|B\.?Sc|B\.?S\.?|BCA|Master'?s?|M\.?Tech|M\.?E\.?|M\.?Sc|M\.?S\.?|MCA|Ph\.?D\.?|Degree in [A-Za-z\s]+)\b"
    ]

    @classmethod
    def extract_requirements(
        cls,
        text: str,
        job_id: Optional[str] = None,
        job_title: Optional[str] = None,
    ) -> JobRequirements:
        """
        Parses clean job description text into structured JobRequirements.

        Args:
            text: Normalized job description text.
            job_id: Optional identifier.
            job_title: Optional title.

        Returns:
            JobRequirements instance.
        """
        jid = job_id or f"JOB-{uuid.uuid4().hex[:6].upper()}"
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        # 1. Infer Job Title if not provided
        title = job_title
        if not title and lines:
            # Check first 2 lines for title-like string
            for line in lines[:2]:
                if len(line) <= 60 and not any(kw in line.lower() for kw in ["about", "company", "description", "overview", "requirements"]):
                    title = line
                    break
        if not title:
            title = "Target Position"

        # 2. Partition into Required vs Preferred vs Responsibilities
        req_section, pref_section, resp_section, edu_section = cls._partition_sections(text)

        # 3. Extract Skills
        all_skills = SkillExtractor.extract_skills(text)
        all_canonical_skills = [s["normalized_skill"] for s in all_skills]

        # Partition skills based on sections
        preferred_skills = []
        if pref_section:
            pref_extracted = SkillExtractor.extract_skills(pref_section)
            preferred_skills = [s["normalized_skill"] for s in pref_extracted]

        required_skills = [s for s in all_canonical_skills if s not in preferred_skills]
        if not required_skills and preferred_skills:
            # If all fell into preferred, promote them to required
            required_skills = preferred_skills
            preferred_skills = []

        # 4. Extract Experience requirement
        exp_years = 0.0
        exp_details = []
        exp_matches = cls._EXP_PATTERN.findall(text)
        if exp_matches:
            try:
                exp_years = float(exp_matches[0])
            except ValueError:
                exp_years = 0.0

        for line in lines:
            if cls._EXP_PATTERN.search(line):
                exp_details.append(line)

        # 5. Extract Education requirements
        education_reqs = []
        target_edu = edu_section if edu_section else text
        for pat in cls._DEGREE_PATTERNS:
            for m in re.finditer(pat, target_edu, re.IGNORECASE):
                education_reqs.append(m.group(0).strip())
        education_reqs = list(dict.fromkeys(education_reqs))

        # 6. Extract Tools (subset of required skills)
        tools = [
            s["normalized_skill"]
            for s in all_skills
            if s["category"] in ("Frameworks & Libraries", "Cloud & DevOps", "Databases & Storage")
        ]

        # 7. Extract Responsibilities
        responsibilities = []
        if resp_section:
            resp_lines = [l.strip() for l in resp_section.splitlines() if len(l.strip()) > 15]
            responsibilities = resp_lines[:8]
        else:
            # Look for lines with action verbs
            for line in lines:
                if len(line) > 20 and any(line.lower().startswith(w) for w in ["develop", "design", "build", "collaborate", "lead", "maintain", "manage", "create", "implement", "work with"]):
                    responsibilities.append(line)
            responsibilities = responsibilities[:8]

        return JobRequirements(
            job_id=jid,
            title=title,
            required_skills=required_skills,
            preferred_skills=preferred_skills,
            education=education_reqs,
            experience_years=exp_years,
            experience_details=exp_details,
            tools=tools,
            responsibilities=responsibilities,
            raw_text=text,
            cleaned_text=text,
            job_keywords=all_canonical_skills,
        )

    @classmethod
    def _partition_sections(cls, text: str) -> tuple[str, str, str, str]:
        """Separates job text into required, preferred, responsibilities, and education blocks."""
        lines = text.splitlines()
        req_lines = []
        pref_lines = []
        resp_lines = []
        edu_lines = []

        current_mode = "general"

        for line in lines:
            line_str = line.strip()
            line_lower = line_str.lower()

            if any(k in line_lower for k in ["preferred", "nice to have", "good to have", "bonus", "plus"]):
                current_mode = "pref"
            elif any(k in line_lower for k in ["requirement", "must have", "qualifications", "what you need", "what we look for"]):
                current_mode = "req"
            elif any(k in line_lower for k in ["responsibilit", "what you will do", "role overview", "day to day"]):
                current_mode = "resp"
            elif any(k in line_lower for k in ["education", "academic"]):
                current_mode = "edu"

            if current_mode == "pref":
                pref_lines.append(line_str)
            elif current_mode == "req":
                req_lines.append(line_str)
            elif current_mode == "resp":
                resp_lines.append(line_str)
            elif current_mode == "edu":
                edu_lines.append(line_str)

        return (
            "\n".join(req_lines),
            "\n".join(pref_lines),
            "\n".join(resp_lines),
            "\n".join(edu_lines),
        )
