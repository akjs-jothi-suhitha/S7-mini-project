-- Database Schema for AI-Driven Resume Screening and Skill Validation System

CREATE TABLE IF NOT EXISTS candidates (
    candidate_id TEXT PRIMARY KEY,
    anonymized_id TEXT UNIQUE NOT NULL,
    raw_filename TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    pii_detected_types TEXT
);

CREATE TABLE IF NOT EXISTS candidate_profiles (
    candidate_id TEXT PRIMARY KEY,
    anonymized_id TEXT NOT NULL,
    education TEXT,
    experience_years REAL DEFAULT 0.0,
    certifications TEXT,
    projects TEXT,
    technical_tools TEXT,
    cleaned_text TEXT,
    anonymized_text TEXT,
    extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(candidate_id) REFERENCES candidates(candidate_id)
);

CREATE TABLE IF NOT EXISTS job_descriptions (
    job_id TEXT PRIMARY KEY,
    title TEXT,
    raw_text TEXT NOT NULL,
    required_skills TEXT,
    preferred_skills TEXT,
    education_req TEXT,
    experience_req REAL DEFAULT 0.0,
    tools_req TEXT,
    responsibilities TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS candidate_skills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id TEXT NOT NULL,
    skill_name TEXT NOT NULL,
    normalized_skill TEXT NOT NULL,
    category TEXT,
    FOREIGN KEY(candidate_id) REFERENCES candidates(candidate_id)
);

CREATE TABLE IF NOT EXISTS screening_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id TEXT NOT NULL,
    anonymized_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    rank INTEGER,
    overall_score REAL NOT NULL,
    semantic_score REAL NOT NULL,
    evidence_score REAL NOT NULL,
    experience_score REAL NOT NULL,
    education_score REAL NOT NULL,
    projects_score REAL NOT NULL,
    recommendation TEXT NOT NULL,
    explanation TEXT NOT NULL,
    matched_skills_json TEXT,
    missing_skills_json TEXT,
    partial_skills_json TEXT,
    evidence_details_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(candidate_id) REFERENCES candidates(candidate_id),
    FOREIGN KEY(job_id) REFERENCES job_descriptions(job_id)
);

CREATE TABLE IF NOT EXISTS fairness_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value REAL,
    details_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
