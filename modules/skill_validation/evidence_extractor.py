"""
Evidence Extractor Module.
Finds contextual evidence snippets for skills across Projects, Experience, and Certifications sections.
"""

import re
from typing import List, Dict, Optional, Tuple
from modules.nlp.text_preprocessor import TextPreprocessor
from modules.models.candidate import CandidateProfile
from modules.models.result import EvidenceItem
from config import EVIDENCE_LEVEL_LABELS


class EvidenceExtractor:
    """Extracts contextual evidence sentences for given skills from candidate resume."""

    @classmethod
    def find_evidence_for_skill(
        cls,
        skill_name: str,
        candidate_profile: CandidateProfile,
    ) -> Tuple[int, str, str, str]:
        """
        Searches resume sections for contextual usage of the skill.

        Args:
            skill_name: Canonical skill name.
            candidate_profile: Structured candidate profile.

        Returns:
            Tuple of:
                - evidence_level: int (0 to 3)
                - confidence_label: str ("Missing", "Low", "Moderate", "High")
                - source_section: str ("Projects", "Experience", "Certifications", "Skills", "None")
                - snippet: str (context sentence/description)
        """
        search_terms = [skill_name.lower()]
        # Add basic aliases
        if skill_name.lower() == "python":
            search_terms.append("py")
        elif skill_name.lower() == "natural language processing":
            search_terms.append("nlp")
        elif skill_name.lower() == "machine learning":
            search_terms.append("ml")
        elif skill_name.lower() == "deep learning":
            search_terms.append("dl")
        elif skill_name.lower() == "artificial intelligence":
            search_terms.append("ai")

        def term_in_text(target: str) -> bool:
            t_lower = f" {target.lower()} "
            for term in search_terms:
                if len(term) <= 2:
                    pat = rf"(?<![a-zA-Z0-9_#+.-]){re.escape(term)}(?![a-zA-Z0-9_#+.-])"
                else:
                    pat = rf"\b{re.escape(term)}\b"
                if re.search(pat, t_lower):
                    return True
            return False

        # 1. Search Work Experience (Substantial evidence -> Level 3)
        for entry in candidate_profile.experience_entries:
            if term_in_text(entry):
                return 3, EVIDENCE_LEVEL_LABELS[3], "Experience", entry

        # 2. Search Projects (Level 3 or 2 depending on description depth)
        for proj in candidate_profile.projects:
            if term_in_text(proj):
                # If project has rich context (>40 chars and action verbs) -> Level 3, else Level 2
                level = 3 if len(proj) >= 40 else 2
                return level, EVIDENCE_LEVEL_LABELS[level], "Projects", proj

        # 3. Search Certifications (Level 2)
        for cert in candidate_profile.certifications:
            if term_in_text(cert):
                return 2, EVIDENCE_LEVEL_LABELS[2], "Certifications", cert

        # 4. Search general sentences across full resume
        sentences = TextPreprocessor.segment_sentences(candidate_profile.cleaned_text)
        for sent in sentences:
            if term_in_text(sent) and len(sent) > 30:
                if any(kw in sent.lower() for kw in ["developed", "built", "implemented", "engineered", "created", "designed", "trained", "deployed"]):
                    return 3, EVIDENCE_LEVEL_LABELS[3], "Experience/Projects", sent
                elif len(sent) > 50:
                    return 2, EVIDENCE_LEVEL_LABELS[2], "Contextual Mention", sent

        # 5. Check if present in candidate skills list only (Level 1)
        for s in candidate_profile.skills:
            if s["normalized_skill"].lower() == skill_name.lower() or term_in_text(s["skill_name"]):
                return 1, EVIDENCE_LEVEL_LABELS[1], "Skills List", f"Listed in skills section: {skill_name}"

        # 6. No evidence found (Level 0)
        return 0, EVIDENCE_LEVEL_LABELS[0], "None", "No supporting evidence found in resume."
