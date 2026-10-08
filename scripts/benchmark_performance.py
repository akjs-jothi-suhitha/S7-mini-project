"""
Performance Benchmark Module for AI-Driven Resume Screening System.
Measures empirical execution times for all architectural pipeline components.
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.document_processing.document_processor import DocumentProcessor
from modules.job_processing.job_parser import JobParser
from modules.job_processing.requirement_extractor import RequirementExtractor
from modules.nlp.information_extractor import InformationExtractor
from modules.semantic_matching.matcher import SemanticSkillMatcher
from modules.semantic_matching.embedding_model import EmbeddingModel
from modules.skill_validation.skill_validator import SkillValidator
from modules.scoring.scoring_engine import ScoringEngine
from modules.scoring.ranking_engine import RankingEngine
from modules.explainability.explanation_engine import ExplanationEngine
from modules.services.screening_service import ScreeningService


def run_benchmark() -> Dict[str, Any]:
    data_dir = PROJECT_ROOT / "data"
    job_path = data_dir / "sample_job_descriptions" / "senior_ml_engineer.docx"
    resume_a_path = data_dir / "sample_resumes" / "candidate_a_alex_turner.docx"
    resume_b_path = data_dir / "sample_resumes" / "candidate_b_priya_sharma.docx"
    resume_c_path = data_dir / "sample_resumes" / "candidate_c_john_doe.docx"

    metrics = {}

    # 1. Job Description Parsing Time
    t0 = time.perf_counter()
    job_text = JobParser.parse(job_path)
    t_job_parse = (time.perf_counter() - t0) * 1000.0
    metrics["job_description_parsing_ms"] = round(t_job_parse, 3)

    # 2. Job Requirement Extraction Time
    t0 = time.perf_counter()
    job_req = RequirementExtractor.extract_requirements(job_text, job_title="Senior ML Engineer")
    t_job_req = (time.perf_counter() - t0) * 1000.0
    metrics["job_requirement_extraction_ms"] = round(t_job_req, 3)

    # 3. Document Ingestion, Text Extraction, Cleaning & PII Masking Time
    t0 = time.perf_counter()
    raw_doc_a = DocumentProcessor.process_file(resume_a_path)
    t_doc_proc = (time.perf_counter() - t0) * 1000.0
    metrics["document_processing_and_pii_masking_ms"] = round(t_doc_proc, 3)

    # 4. NLP Information Extraction Time (Profile, Skills, Education, Experience)
    t0 = time.perf_counter()
    profile_a = InformationExtractor.extract_profile(raw_doc_a)
    t_nlp = (time.perf_counter() - t0) * 1000.0
    metrics["nlp_information_extraction_ms"] = round(t_nlp, 3)

    # 5. Semantic Embedding Generation Time
    emb_model = EmbeddingModel()
    t0 = time.perf_counter()
    _ = emb_model.encode(["Machine Learning", "Natural Language Processing", "PyTorch", "AWS"])
    t_embed = (time.perf_counter() - t0) * 1000.0
    metrics["semantic_embedding_generation_ms"] = round(t_embed, 3)

    # 6. Semantic Skill Matching Time
    matcher = SemanticSkillMatcher()
    t0 = time.perf_counter()
    matched, partial, missing, sem_score = matcher.match_skills(profile_a, job_req)
    t_sem_match = (time.perf_counter() - t0) * 1000.0
    metrics["semantic_skill_matching_ms"] = round(t_sem_match, 3)

    # 7. Skill Evidence Validation Time (4-tier search across projects/experience/certs)
    t0 = time.perf_counter()
    (
        en_matched,
        en_partial,
        en_missing,
        ev_items,
        strong,
        weak,
        ev_score,
    ) = SkillValidator.validate_skills(matched, partial, missing, profile_a)
    t_evidence = (time.perf_counter() - t0) * 1000.0
    metrics["skill_evidence_validation_ms"] = round(t_evidence, 3)

    # 8. Scoring & Explainability Generation Time
    t0 = time.perf_counter()
    overall_score, breakdown, rec = ScoringEngine.calculate_score(profile_a, job_req, sem_score, ev_score)
    explanation = ExplanationEngine.generate_explanation(
        profile_a, job_req, en_matched, en_partial, en_missing, ev_items, overall_score, rec
    )
    t_scoring_xai = (time.perf_counter() - t0) * 1000.0
    metrics["scoring_and_xai_generation_ms"] = round(t_scoring_xai, 3)

    # 9. Single Candidate Total In-Memory Processing Time
    metrics["single_candidate_total_ms"] = round(
        t_doc_proc + t_nlp + t_sem_match + t_evidence + t_scoring_xai, 3
    )

    # 10. Multi-Candidate Batch End-to-End Execution Time (including SQLite persistence)
    service = ScreeningService()
    resumes = [resume_a_path, resume_b_path, resume_c_path]

    t0 = time.perf_counter()
    batch_res = service.process_screening(
        job_input=job_path,
        resume_files=resumes,
        job_title="Senior Machine Learning Engineer",
    )
    t_batch = (time.perf_counter() - t0) * 1000.0

    metrics["multi_candidate_batch_total_ms"] = round(t_batch, 3)
    metrics["batch_candidate_count"] = len(resumes)
    metrics["average_time_per_candidate_in_batch_ms"] = round(t_batch / len(resumes), 3)

    return metrics


if __name__ == "__main__":
    results = run_benchmark()
    print("==================================================")
    print("  AI RESUME SCREENING - PERFORMANCE BENCHMARK     ")
    print("==================================================")
    for k, v in results.items():
        print(f"{k:45}: {v}")
    print("==================================================")
