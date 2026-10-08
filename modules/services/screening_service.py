"""
Screening Service Orchestrator Module.
Coordinates the entire end-to-end resume screening, skill validation, bias reduction,
scoring, explainability, ranking, and database persistence pipeline.
"""

import time
import uuid
from typing import List, Union, BinaryIO, Dict, Any, Optional
from pathlib import Path
import io

from modules.document_processing.document_processor import DocumentProcessor
from modules.document_processing.file_validator import ValidationError
from modules.job_processing.job_parser import JobParser
from modules.job_processing.requirement_extractor import RequirementExtractor
from modules.nlp.information_extractor import InformationExtractor
from modules.semantic_matching.matcher import SemanticSkillMatcher
from modules.skill_validation.skill_validator import SkillValidator
from modules.fairness.bias_reduction import BiasReductionPipeline
from modules.fairness.fairness_metrics import FairnessAuditor
from modules.scoring.scoring_engine import ScoringEngine
from modules.scoring.ranking_engine import RankingEngine
from modules.explainability.explanation_engine import ExplanationEngine
from modules.models.candidate import RawCandidateDoc, CandidateProfile
from modules.models.job import JobRequirements
from modules.models.result import (
    CandidateScreeningResult,
    BatchScreeningResult,
)
import database
from modules.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ScreeningService:
    """Central orchestrator for the AI-Driven Resume Screening and Skill Validation System."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path
        # Initialize database tables
        database.init_db(self.db_path)
        self.semantic_matcher = SemanticSkillMatcher()

    def process_screening(
        self,
        job_input: Union[str, Path, BinaryIO, bytes, io.BytesIO],
        resume_files: List[Union[str, Path, BinaryIO, bytes, io.BytesIO]],
        job_filename: str = "",
        resume_filenames: Optional[List[str]] = None,
        job_title: Optional[str] = None,
    ) -> BatchScreeningResult:
        """
        Executes the end-to-end recruitment screening workflow.

        Steps:
            1. Parse Job Description
            2. Extract Job Requirements
            3. Ingest and Process Resumes (with PII Masking)
            4. Extract Candidate Information
            5. Perform Semantic Skill Matching
            6. Perform Skill Evidence Validation
            7. Apply Bias Reduction Checks
            8. Calculate Multi-Factor Candidate Scores
            9. Generate Explainable Recommendations
            10. Rank Candidates
            11. Perform Fairness Audit
            12. Persist to SQLite Database

        Args:
            job_input: Job description text or file.
            resume_files: List of uploaded resume files.
            job_filename: Optional name of job file.
            resume_filenames: Optional list of resume filenames.
            job_title: Optional manual title for the job.

        Returns:
            BatchScreeningResult containing ranked candidates and fairness audit metrics.
        """
        start_time = time.time()
        logger.info("Screening process initiated for %d candidate resume(s).", len(resume_files))

        if not resume_files:
            raise ValidationError("Please provide at least one resume to screen.")

        # 1 & 2. Process Job Description
        job_clean_text = JobParser.parse(job_input, filename=job_filename)
        job_requirements = RequirementExtractor.extract_requirements(
            job_clean_text, job_title=job_title
        )

        # Save Job Description to Database
        try:
            database.save_job_description(
                job_id=job_requirements.job_id,
                title=job_requirements.title,
                raw_text=job_requirements.raw_text,
                required_skills=job_requirements.required_skills,
                preferred_skills=job_requirements.preferred_skills,
                education_req=", ".join(job_requirements.education),
                experience_req=job_requirements.experience_years,
                tools_req=job_requirements.tools,
                responsibilities=job_requirements.responsibilities,
                db_path=self.db_path,
            )
        except Exception as e:
            logger.error("Failed to save job description to DB: %s", str(e))

        candidate_results: List[CandidateScreeningResult] = []

        # 3 to 10. Process each Resume
        for idx, resume_file in enumerate(resume_files):
            fname = (
                resume_filenames[idx]
                if resume_filenames and idx < len(resume_filenames)
                else getattr(resume_file, "name", f"resume_{idx+1}.pdf")
            )

            try:
                # Ingest, clean, detect PII, anonymize
                raw_doc: RawCandidateDoc = DocumentProcessor.process_file(
                    resume_file, filename=fname
                )

                # Extract Structured Candidate Profile
                profile: CandidateProfile = InformationExtractor.extract_profile(raw_doc)

                # Save Candidate & Profile to SQLite
                try:
                    database.save_candidate(
                        candidate_id=profile.candidate_id,
                        anonymized_id=profile.anonymized_id,
                        raw_filename=fname,
                        pii_detected_types=list(raw_doc.pii_entities.keys()),
                        db_path=self.db_path,
                    )
                    database.save_candidate_profile(
                        candidate_id=profile.candidate_id,
                        anonymized_id=profile.anonymized_id,
                        education=", ".join([e.get("degree", "") for e in profile.education]),
                        experience_years=profile.experience_years,
                        certifications=profile.certifications,
                        projects=profile.projects,
                        technical_tools=profile.technical_tools,
                        cleaned_text=profile.cleaned_text,
                        anonymized_text=profile.anonymized_text,
                        db_path=self.db_path,
                    )
                    database.save_candidate_skills(
                        candidate_id=profile.candidate_id,
                        skills=profile.skills,
                        db_path=self.db_path,
                    )
                except Exception as e:
                    logger.error("Database save candidate error: %s", str(e))

                # Semantic Skill Matching
                matched, partial, missing, semantic_score = self.semantic_matcher.match_skills(
                    profile, job_requirements
                )

                # Skill Evidence Validation (Novelty component)
                (
                    enriched_matched,
                    enriched_partial,
                    enriched_missing,
                    evidence_items,
                    strong_ev,
                    weak_ev,
                    evidence_score,
                ) = SkillValidator.validate_skills(
                    matched, partial, missing, profile
                )

                # Bias Reduction: Sanitize attributes for scoring
                _ = BiasReductionPipeline.sanitize_for_scoring(profile)

                # Calculate Candidate Score & Recommendation
                overall_score, breakdown, recommendation = ScoringEngine.calculate_score(
                    candidate_profile=profile,
                    job_requirements=job_requirements,
                    semantic_skill_score=semantic_score,
                    evidence_score=evidence_score,
                )

                # Generate Data-Driven Explainability
                explanation = ExplanationEngine.generate_explanation(
                    candidate_profile=profile,
                    job_requirements=job_requirements,
                    matched_skills=enriched_matched,
                    partial_skills=enriched_partial,
                    missing_skills=enriched_missing,
                    evidence_items=evidence_items,
                    overall_score=overall_score,
                    recommendation=recommendation,
                )

                # PII Summary counts
                pii_counts = {k: len(v) for k, v in raw_doc.pii_entities.items() if v}

                result_item = CandidateScreeningResult(
                    candidate_id=profile.candidate_id,
                    anonymized_id=profile.anonymized_id,
                    job_id=job_requirements.job_id,
                    candidate_name=profile.candidate_name,
                    overall_score=overall_score,
                    score_breakdown=breakdown,
                    matched_skills=enriched_matched,
                    partial_skills=enriched_partial,
                    missing_skills=enriched_missing,
                    evidence_items=evidence_items,
                    strong_evidence=strong_ev,
                    weak_evidence=weak_ev,
                    relevant_experience=profile.experience_entries,
                    relevant_projects=profile.projects,
                    recommendation=recommendation,
                    explanation=explanation,
                    pii_summary=pii_counts,
                )
                candidate_results.append(result_item)
                logger.info(
                    "Processed candidate %s: Score=%.1f, Recommendation=%s",
                    profile.anonymized_id,
                    overall_score,
                    recommendation,
                )

            except Exception as e:
                logger.error("Error processing file '%s': %s", fname, str(e))
                raise

        # 11. Candidate Ranking
        ranked_candidates = RankingEngine.rank_candidates(candidate_results)

        # 12. Fairness Audit across batch
        fairness_metrics = FairnessAuditor.audit_batch_fairness(ranked_candidates)

        # Persist Screening Results & Fairness Audit to SQLite
        batch_id = f"BATCH-{uuid.uuid4().hex[:6].upper()}"
        for cand in ranked_candidates:
            try:
                database.save_screening_result(
                    candidate_id=cand.candidate_id,
                    anonymized_id=cand.anonymized_id,
                    job_id=job_requirements.job_id,
                    rank=cand.rank,
                    overall_score=cand.overall_score,
                    semantic_score=cand.score_breakdown.semantic_skill_match_score if cand.score_breakdown else 0.0,
                    evidence_score=cand.score_breakdown.skill_evidence_score if cand.score_breakdown else 0.0,
                    experience_score=cand.score_breakdown.experience_relevance_score if cand.score_breakdown else 0.0,
                    education_score=cand.score_breakdown.education_relevance_score if cand.score_breakdown else 0.0,
                    projects_score=cand.score_breakdown.projects_certs_score if cand.score_breakdown else 0.0,
                    recommendation=cand.recommendation,
                    explanation=cand.explanation,
                    matched_skills=[m.__dict__ for m in cand.matched_skills],
                    missing_skills=[m.__dict__ for m in cand.missing_skills],
                    partial_skills=[m.__dict__ for m in cand.partial_skills],
                    evidence_details=[e.__dict__ for e in cand.evidence_items],
                    db_path=self.db_path,
                )
            except Exception as e:
                logger.error("Failed to save screening result for candidate: %s", str(e))

        try:
            database.save_fairness_audit(
                batch_id=batch_id,
                metric_name="batch_fairness_summary",
                metric_value=fairness_metrics.get("selection_rate_pct", 0.0),
                details=fairness_metrics,
                db_path=self.db_path,
            )
        except Exception as e:
            logger.error("Failed to save fairness audit to DB: %s", str(e))

        elapsed = time.time() - start_time
        logger.info("Screening pipeline completed in %.2f seconds for %d candidates.", elapsed, len(ranked_candidates))

        return BatchScreeningResult(
            job_id=job_requirements.job_id,
            job_title=job_requirements.title,
            total_candidates=len(ranked_candidates),
            ranked_candidates=ranked_candidates,
            fairness_metrics=fairness_metrics,
        )
