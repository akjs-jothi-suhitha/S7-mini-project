"""
Skill Extractor Module.
Extracts technical skills and tools from resume or job text using the Skill Taxonomy and NLP pattern matching.
"""

import re
from typing import List, Dict, Set, Any
from modules.nlp.skill_taxonomy import SkillTaxonomy


class SkillExtractor:
    """Extracts, categorizes, and normalizes skills from text."""

    @classmethod
    def extract_skills(cls, text: str) -> List[Dict[str, Any]]:
        """
        Extracts all technical skills found in text with their canonical names and categories.

        Args:
            text: Resume or job text.

        Returns:
            List of dictionaries with keys: 'skill_name', 'normalized_skill', 'category'.
        """
        if not text:
            return []

        searchable_terms = SkillTaxonomy.get_all_searchable_terms()
        # Sort terms by length descending to match multi-word phrases first (e.g. 'natural language processing' before 'r')
        sorted_terms = sorted(searchable_terms.keys(), key=lambda x: len(x), reverse=True)

        found_canonical: Set[str] = set()
        extracted_skills: List[Dict[str, Any]] = []

        text_lower = f" {text.lower()} "

        # Helper to construct safe regex pattern for terms
        for term in sorted_terms:
            canon = searchable_terms[term]
            if canon in found_canonical:
                continue

            # Handle special symbols in terms like c++, c#, .net, ci/cd, node.js
            escaped_term = re.escape(term)

            # Strict boundaries: ensure word boundary on letters/digits
            # For short 1-2 char terms (e.g. 'c', 'r', 'go', 'js', 'ts', 'ai', 'ml', 'db'), require strict word boundary
            if len(term) <= 2:
                pattern = rf"(?<![a-zA-Z0-9_#+.-]){escaped_term}(?![a-zA-Z0-9_#+.-])"
            elif term.endswith("+") or term.endswith("#"):
                pattern = rf"(?<![a-zA-Z0-9_#+.-]){escaped_term}(?![a-zA-Z0-9_#+.-])"
            else:
                pattern = rf"\b{escaped_term}\b"

            if re.search(pattern, text_lower):
                found_canonical.add(canon)
                category = SkillTaxonomy.get_category(canon)
                extracted_skills.append({
                    "skill_name": term.title() if len(term) > 3 else term.upper(),
                    "normalized_skill": canon,
                    "category": category,
                })

        return extracted_skills

    @classmethod
    def extract_skill_names(cls, text: str) -> List[str]:
        """Convenience method returning just the list of canonical skill names."""
        skills = cls.extract_skills(text)
        return [s["normalized_skill"] for s in skills]
