"""
Semantic Matcher Module.
Executes semantic skill matching and requirement comparison between candidate profiles and job requirements.
"""

from typing import List, Dict, Tuple, Any
import numpy as np
from config import (
    SEMANTIC_MATCH_HIGH_THRESHOLD,
    SEMANTIC_MATCH_PARTIAL_THRESHOLD,
)
from modules.semantic_matching.embedding_model import EmbeddingModel
from modules.semantic_matching.similarity import compute_cosine_similarity, batch_cosine_similarity
from modules.nlp.skill_taxonomy import SkillTaxonomy
from modules.models.candidate import CandidateProfile
from modules.models.job import JobRequirements
from modules.models.result import SkillMatchItem
from modules.utils.logging_utils import get_logger

logger = get_logger(__name__)


class SemanticSkillMatcher:
    """Matches job skills against candidate profile using taxonomy and semantic embeddings."""

    def __init__(self):
        self.embedding_model = EmbeddingModel()

    def match_skills(
        self,
        candidate_profile: CandidateProfile,
        job_requirements: JobRequirements,
    ) -> Tuple[List[SkillMatchItem], List[SkillMatchItem], List[SkillMatchItem], float]:
        """
        Matches candidate skills against required and preferred job skills.

        Args:
            candidate_profile: Extracted candidate profile.
            job_requirements: Extracted job requirements.

        Returns:
            Tuple of:
                - matched_skills: List[SkillMatchItem]
                - partial_skills: List[SkillMatchItem]
                - missing_skills: List[SkillMatchItem]
                - semantic_skill_score: float (0.0 to 100.0)
        """
        job_skills = list(dict.fromkeys(job_requirements.required_skills + job_requirements.preferred_skills))
        if not job_skills:
            # If no specific skills extracted, fallback to job keywords
            job_skills = job_requirements.job_keywords or ["General Software Engineering"]

        candidate_skill_names = [s["normalized_skill"] for s in candidate_profile.skills]

        matched_items: List[SkillMatchItem] = []
        partial_items: List[SkillMatchItem] = []
        missing_items: List[SkillMatchItem] = []

        # Pre-embed candidate skills if present
        cand_embeddings = (
            self.embedding_model.encode(candidate_skill_names)
            if candidate_skill_names
            else np.zeros((0, 384))
        )

        total_match_weight = 0.0

        for job_skill in job_skills:
            canon_job_skill = SkillTaxonomy.normalize(job_skill)

            # 1. Exact canonical or synonym match
            if canon_job_skill in candidate_skill_names or any(
                SkillTaxonomy.normalize(cs).lower() == canon_job_skill.lower()
                for cs in candidate_skill_names
            ):
                item = SkillMatchItem(
                    job_skill=canon_job_skill,
                    status="matched",
                    matched_candidate_skill=canon_job_skill,
                    similarity_score=1.0,
                )
                matched_items.append(item)
                total_match_weight += 1.0
                continue

            # 2. Semantic embedding matching if not exact match
            best_sim = 0.0
            best_cand_skill = None

            if len(cand_embeddings) > 0:
                job_vec = self.embedding_model.encode(canon_job_skill)
                sims = batch_cosine_similarity(job_vec, cand_embeddings)[0]
                best_idx = int(np.argmax(sims))
                best_sim = float(sims[best_idx])
                best_cand_skill = candidate_skill_names[best_idx]

            # Determine category based on thresholds
            if best_sim >= SEMANTIC_MATCH_HIGH_THRESHOLD:
                item = SkillMatchItem(
                    job_skill=canon_job_skill,
                    status="matched",
                    matched_candidate_skill=best_cand_skill,
                    similarity_score=round(best_sim, 3),
                )
                matched_items.append(item)
                total_match_weight += best_sim
            elif best_sim >= SEMANTIC_MATCH_PARTIAL_THRESHOLD:
                item = SkillMatchItem(
                    job_skill=canon_job_skill,
                    status="partially_matched",
                    matched_candidate_skill=best_cand_skill,
                    similarity_score=round(best_sim, 3),
                )
                partial_items.append(item)
                total_match_weight += best_sim * 0.6  # partial credit
            else:
                item = SkillMatchItem(
                    job_skill=canon_job_skill,
                    status="missing",
                    matched_candidate_skill=None,
                    similarity_score=round(best_sim, 3),
                )
                missing_items.append(item)

        # Calculate normalized score (0 - 100)
        max_possible = len(job_skills)
        semantic_score = (total_match_weight / max_possible) * 100.0 if max_possible > 0 else 0.0
        semantic_score = round(min(100.0, max(0.0, semantic_score)), 2)

        return matched_items, partial_items, missing_items, semantic_score
