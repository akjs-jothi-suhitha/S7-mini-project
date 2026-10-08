"""
Database management module for the AI-Driven Resume Screening System.
Uses SQLite for persistent storage of candidates, profiles, jobs, skills, and screening results.
"""

import sqlite3
import json
from pathlib import Path
from contextlib import contextmanager
from typing import Dict, List, Optional, Any, Generator
from config import DATABASE_PATH
from modules.utils.logging_utils import get_logger

logger = get_logger(__name__)


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Creates and returns a SQLite database connection with row factory enabled."""
    target_path = db_path or DATABASE_PATH
    db_file = Path(target_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_file))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def db_session(db_path: Optional[str] = None) -> Generator[sqlite3.Connection, None, None]:
    """Context manager that yields a connection and guarantees it is closed on exit."""
    conn = get_connection(db_path)
    try:
        yield conn
    finally:
        conn.close()


def init_db(db_path: Optional[str] = None) -> None:
    """Initializes the database schema using database/schema.sql."""
    schema_path = Path(__file__).resolve().parent / "database" / "schema.sql"
    if not schema_path.exists():
        schema_path = Path(__file__).resolve().parent.parent / "database" / "schema.sql"

    with open(schema_path, "r", encoding="utf-8") as f:
        schema_sql = f.read()

    with db_session(db_path) as conn:
        conn.executescript(schema_sql)
        conn.commit()
    logger.info("Database initialized successfully at %s", db_path or DATABASE_PATH)


def save_candidate(
    candidate_id: str,
    anonymized_id: str,
    raw_filename: str = "",
    pii_detected_types: Optional[List[str]] = None,
    db_path: Optional[str] = None,
) -> None:
    """Saves basic candidate metadata."""
    pii_str = json.dumps(pii_detected_types or [])
    with db_session(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO candidates (candidate_id, anonymized_id, raw_filename, pii_detected_types)
            VALUES (?, ?, ?, ?)
            """,
            (candidate_id, anonymized_id, raw_filename, pii_str),
        )
        conn.commit()


def save_candidate_profile(
    candidate_id: str,
    anonymized_id: str,
    education: str = "",
    experience_years: float = 0.0,
    certifications: Optional[List[str]] = None,
    projects: Optional[List[str]] = None,
    technical_tools: Optional[List[str]] = None,
    cleaned_text: str = "",
    anonymized_text: str = "",
    db_path: Optional[str] = None,
) -> None:
    """Saves candidate extracted profile data."""
    certs_str = json.dumps(certifications or [])
    projects_str = json.dumps(projects or [])
    tools_str = json.dumps(technical_tools or [])

    with db_session(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO candidate_profiles (
                candidate_id, anonymized_id, education, experience_years,
                certifications, projects, technical_tools, cleaned_text, anonymized_text
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                candidate_id,
                anonymized_id,
                education,
                experience_years,
                certs_str,
                projects_str,
                tools_str,
                cleaned_text,
                anonymized_text,
            ),
        )
        conn.commit()


def save_candidate_skills(
    candidate_id: str,
    skills: List[Dict[str, str]],
    db_path: Optional[str] = None,
) -> None:
    """Saves candidate skills."""
    with db_session(db_path) as conn:
        conn.execute("DELETE FROM candidate_skills WHERE candidate_id = ?", (candidate_id,))
        for skill in skills:
            conn.execute(
                """
                INSERT INTO candidate_skills (candidate_id, skill_name, normalized_skill, category)
                VALUES (?, ?, ?, ?)
                """,
                (
                    candidate_id,
                    skill.get("skill_name", ""),
                    skill.get("normalized_skill", ""),
                    skill.get("category", "General"),
                ),
            )
        conn.commit()


def save_job_description(
    job_id: str,
    title: str,
    raw_text: str,
    required_skills: Optional[List[str]] = None,
    preferred_skills: Optional[List[str]] = None,
    education_req: str = "",
    experience_req: float = 0.0,
    tools_req: Optional[List[str]] = None,
    responsibilities: Optional[List[str]] = None,
    db_path: Optional[str] = None,
) -> None:
    """Saves a job description and its parsed requirements."""
    req_skills_str = json.dumps(required_skills or [])
    pref_skills_str = json.dumps(preferred_skills or [])
    tools_str = json.dumps(tools_req or [])
    resp_str = json.dumps(responsibilities or [])

    with db_session(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO job_descriptions (
                job_id, title, raw_text, required_skills, preferred_skills,
                education_req, experience_req, tools_req, responsibilities
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                title,
                raw_text,
                req_skills_str,
                pref_skills_str,
                education_req,
                experience_req,
                tools_str,
                resp_str,
            ),
        )
        conn.commit()


def save_screening_result(
    candidate_id: str,
    anonymized_id: str,
    job_id: str,
    rank: int,
    overall_score: float,
    semantic_score: float,
    evidence_score: float,
    experience_score: float,
    education_score: float,
    projects_score: float,
    recommendation: str,
    explanation: str,
    matched_skills: Optional[List[Any]] = None,
    missing_skills: Optional[List[Any]] = None,
    partial_skills: Optional[List[Any]] = None,
    evidence_details: Optional[List[Any]] = None,
    db_path: Optional[str] = None,
) -> None:
    """Saves a candidate's screening and ranking result."""
    with db_session(db_path) as conn:
        conn.execute(
            """
            INSERT INTO screening_results (
                candidate_id, anonymized_id, job_id, rank, overall_score,
                semantic_score, evidence_score, experience_score, education_score,
                projects_score, recommendation, explanation,
                matched_skills_json, missing_skills_json, partial_skills_json,
                evidence_details_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                candidate_id,
                anonymized_id,
                job_id,
                rank,
                overall_score,
                semantic_score,
                evidence_score,
                experience_score,
                education_score,
                projects_score,
                recommendation,
                explanation,
                json.dumps(matched_skills or []),
                json.dumps(missing_skills or []),
                json.dumps(partial_skills or []),
                json.dumps(evidence_details or []),
            ),
        )
        conn.commit()


def get_screening_results_for_job(job_id: str, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieves all screening results for a given job, ordered by rank."""
    with db_session(db_path) as conn:
        cursor = conn.execute(
            """
            SELECT * FROM screening_results
            WHERE job_id = ?
            ORDER BY rank ASC, overall_score DESC
            """,
            (job_id,),
        )
        rows = cursor.fetchall()
        results = []
        for row in rows:
            res = dict(row)
            res["matched_skills"] = json.loads(res.get("matched_skills_json") or "[]")
            res["missing_skills"] = json.loads(res.get("missing_skills_json") or "[]")
            res["partial_skills"] = json.loads(res.get("partial_skills_json") or "[]")
            res["evidence_details"] = json.loads(res.get("evidence_details_json") or "[]")
            results.append(res)
        return results


def save_fairness_audit(
    batch_id: str,
    metric_name: str,
    metric_value: float,
    details: Optional[Dict[str, Any]] = None,
    db_path: Optional[str] = None,
) -> None:
    """Saves fairness audit log entry."""
    with db_session(db_path) as conn:
        conn.execute(
            """
            INSERT INTO fairness_audit (batch_id, metric_name, metric_value, details_json)
            VALUES (?, ?, ?, ?)
            """,
            (batch_id, metric_name, metric_value, json.dumps(details or {})),
        )
        conn.commit()
