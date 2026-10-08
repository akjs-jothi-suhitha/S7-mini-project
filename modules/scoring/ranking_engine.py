"""
Ranking Engine Module.
Ranks screened candidates by score and generates sorted leaderboards.
"""

from typing import List
from modules.models.result import CandidateScreeningResult


class RankingEngine:
    """Ranks candidates based on multi-factor scores."""

    @classmethod
    def rank_candidates(
        cls,
        results: List[CandidateScreeningResult],
    ) -> List[CandidateScreeningResult]:
        """
        Sorts candidates by overall score (descending) and assigns sequential integer ranks.

        Args:
            results: List of candidate screening results.

        Returns:
            Ranked list of candidate screening results.
        """
        if not results:
            return []

        # Sort descending by overall_score, tiebreak with semantic score and evidence score
        sorted_results = sorted(
            results,
            key=lambda r: (
                r.overall_score,
                r.score_breakdown.semantic_skill_match_score if r.score_breakdown else 0.0,
                r.score_breakdown.skill_evidence_score if r.score_breakdown else 0.0,
            ),
            reverse=True,
        )

        for i, item in enumerate(sorted_results):
            item.rank = i + 1

        return sorted_results
