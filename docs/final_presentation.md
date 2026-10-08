# FINAL PROJECT PRESENTATION & ACADEMIC DEFENSE GUIDE

## AI-DRIVEN RESUME SCREENING AND SKILL VALIDATION SYSTEM WITH BIAS REDUCTION TECHNIQUES

---

### 1. Project Title
**AI-Driven Resume Screening and Skill Validation System with Bias Reduction Techniques**

---

### 2. Introduction
Traditional recruitment workflows struggle with high volumes of unstructured applicant resumes. Manual screening is slow, error-prone, and susceptible to unconscious human bias. Keyword-based applicant tracking systems (ATS) often fail because they lack semantic comprehension and cannot verify whether a candidate actually applied the claimed skills in real projects or professional experience. This system introduces an end-to-end AI-powered recruitment pipeline designed for fair, transparent, and explainable hiring decisions.

---

### 3. Problem Statement
1. **Keyword Rigidity:** Conventional ATS systems rely on exact string matching, missing qualified candidates who express identical concepts using different terminology.
2. **Resume Embellishment & Keyword Stuffing:** Candidates often list skills they have never applied in practice, receiving unmerited high scores in traditional systems.
3. **Unconscious Bias:** Demographic identifiers (name, gender, age, address) introduce cognitive bias into initial screening.
4. **Black-Box Decision Making:** Modern ML models often output opaque scores without providing recruiters with auditable explanations.

---

### 4. Existing System
- Legacy Keyword-Based Filtering (Boolean Search / Exact Substring Matching).
- First-generation ATS systems counting word frequencies.
- Unstructured manual resume reviews by recruitment staff.

---

### 5. Existing Limitations
- Zero semantic contextual understanding.
- Inability to distinguish between a skill mentioned in a list versus a skill used in major projects.
- Demographic data exposed throughout screening, perpetuating hiring disparity.
- No explainability or transparency in candidate rejection.

---

### 6. Proposed System
An intelligent, explainable AI screening architecture that:
1. Ingests PDF and DOCX resumes.
2. Anonymizes PII to create bias-mitigated representations.
3. Uses **Sentence-BERT** embeddings and cosine similarity for deep semantic matching.
4. Performs **Skill Evidence Validation** across 4 confidence tiers (Missing, Low, Moderate, High).
5. Computes transparent 5-pillar scores.
6. Generates data-driven natural language explanations.
7. Displays interactive candidate comparisons and Fairlearn fairness audit metrics in a Streamlit recruiter dashboard.

---

### 7. Project Aim
Develop an AI-powered recruitment system that intelligently screens resumes, validates candidate skills against verifiable contextual evidence, minimizes hiring bias, and assists recruiters in selecting suitable candidates through fair, transparent, and explainable evaluation.

---

### 8. Objectives
- Support PDF and DOCX resume parsing and text normalization.
- Implement an automated PII detector and anonymizer for demographic bias reduction.
- Build a curated technical Skill Taxonomy with comprehensive alias mappings.
- Implement Sentence-BERT semantic similarity for candidate-job requirement alignment.
- Implement a 4-tier skill evidence validation engine.
- Formulate a transparent multi-factor weighted scoring model.
- Generate explainable AI (XAI) candidate rationales.
- Provide a responsive Streamlit recruiter dashboard with Plotly analytics and SQLite persistence.

---

### 9. Architecture & System Flow

```
START
  ↓
Resume (PDF/DOCX) + Job Description
  ↓
Document Processing (Validation, Parsing, Cleaning, PII Anonymization)
  ↓
NLP Information Extraction (Profile, Education, Experience, Certs, Projects)
  ↓
Job Requirement Extraction (Required vs Preferred Skills, Experience Req)
  ↓
Semantic Skill Matching (Sentence-BERT & Cosine Similarity)
  ↓
Skill Evidence Validation (4-Tier Contextual Evidence Scoring)
  ↓
Bias Reduction (Merit Attribute Isolation & PII Shielding)
  ↓
Candidate Score Generation (5-Pillar Weighted Formulation)
  ↓
Explainable Recommendation (Data-Driven XAI Breakdown)
  ↓
Candidate Ranking (Score Sorting Descending)
  ↓
SQLite Persistence & Recruiter Dashboard
  ↓
END
```

---

### 10. Core Methodology
- **Anonymized Ingestion:** Extract raw text, sanitize PII, assign anonymized identifiers (`ANON-XXXX`).
- **Semantic Representation:** Generate dense vector embeddings for job requirements and candidate profiles.
- **Evidence Cross-Referencing:** Sentence-level search across Work Experience, Projects, and Certifications.
- **Fairness Audit:** Evaluate batch score distributions and demographic parity using Fairlearn.

---

### 11. Resume Processing Module
- **PDF Extraction:** PyMuPDF (`fitz`) / `pdfplumber` / `PyPDF2` fallback.
- **DOCX Extraction:** `python-docx` / standard library OpenXML fallback.
- **Text Cleaning:** Unicode NFKC normalization, bullet standardization, preserving technical terms (`C++`, `.NET`, `Node.js`, `Scikit-learn`, `CI/CD`).

---

### 12. NLP Information Extraction Module
- Extracts canonical skills and technical tools.
- Extracts education degrees (B.Tech, MS, PhD) and graduation years.
- Calculates professional years of experience from explicit patterns and date intervals.
- Segments projects and professional certifications.

---

### 13. Semantic Matching Engine (Novelty 1)
- Model: Pretrained Sentence-BERT (`all-MiniLM-L6-v2`) with cosine similarity.
- Solves keyword mismatches (e.g. "Natural Language Processing" vs "NLP with transformers").
- Classifies skills into:
  - **Matched:** Exact match or similarity $\ge 0.75$.
  - **Partially Matched:** Similarity $0.55 - 0.74$.
  - **Missing:** Similarity $< 0.55$.

---

### 14. Skill Evidence Validation (Novelty 2)
Cross-references every claimed skill with resume context across 4 tiers:
- **Level 0 (Missing - 0.0):** No evidence found.
- **Level 1 (Low - 0.3):** Listed in skills section only.
- **Level 2 (Moderate - 0.7):** Supported by project, course, or certification.
- **Level 3 (High - 1.0):** Supported by substantial work experience or production project achievements.

---

### 15. Bias Reduction & Fairness (Novelty 3)
- Masks: Candidate Name, Email, Phone, Physical Address, URLs, Dates of Birth.
- Feature Isolation: Scoring engine receives strictly merit-based qualification vectors.
- Audit: Evaluates Selection Rate Parity and Demographic Parity Difference across candidate cohorts.
- *Academic Disclosure:* PII masking is a best-effort bias-reduction technique; it does not claim to eliminate societal bias entirely.

---

### 16. Candidate Scoring Engine
Transparent, configurable 5-pillar mathematical formulation:

$$\text{Overall Score} = (0.35 \times S_{\text{skills}}) + (0.25 \times S_{\text{evidence}}) + (0.20 \times S_{\text{exp}}) + (0.10 \times S_{\text{edu}}) + (0.10 \times S_{\text{proj\_certs}})$$

All component scores normalized to $0 - 100$. Zero magic numbers.

---

### 17. Candidate Ranking Engine
- Sorts candidate results descending by `overall_score`.
- Deterministic tie-breaking using semantic skill score followed by evidence score.
- Assigns sequential ranks ($1, 2, 3, \dots$).

---

### 18. Explainable Recommendation (Novelty 4)
Synthesizes transparent, data-driven candidate rationales:
- Overall Match percentage and recommendation category (`HIGH RELEVANCE`, `RELEVANT`, `MODERATE RELEVANCE`, `LOW RELEVANCE`).
- Specific list of matched, partial, and missing skills.
- Direct citations of contextual evidence snippets.
- Experience and educational alignment.

---

### 19. Recruiter Dashboard
Streamlit UI features:
- Job description input (Text, PDF, DOCX).
- Multi-resume batch upload.
- 1-Click Academic Demo Mode.
- Anonymized leaderboard with candidate name toggle.
- Detailed candidate skill evidence breakdown tables.
- Interactive Plotly visualizations (score comparison & 5-pillar breakdown).
- Batch CSV export.

---

### 20. Database & Persistence
- SQLite database (`database/screening_system.db`).
- Normalized relational schema: `candidates`, `candidate_profiles`, `job_descriptions`, `candidate_skills`, `screening_results`, `fairness_audit`.
- Safe session management ensuring immediate connection closure on Windows.

---

### 21. Testing & Verification
- Test Suite: 26 unit and end-to-end integration tests in `tests/`.
- 100% Pass Rate across all modules (document processing, PII masking, taxonomy, semantic matching, evidence validation, scoring, ranking, explainability, database, DOCX/PDF E2E).

---

### 22. Performance Evaluation Results
Measured on real multi-candidate batch processing:
- **Job Description Parsing:** $\sim 136.8 \text{ ms}$
- **Job Requirement Extraction:** $\sim 16.8 \text{ ms}$
- **Document Processing & PII Masking:** $\sim 22.1 \text{ ms}$
- **NLP Information Extraction:** $\sim 10.8 \text{ ms}$
- **Semantic Skill Matching:** $\sim 3.5 \text{ ms}$
- **Skill Evidence Validation:** $\sim 4.4 \text{ ms}$
- **Scoring & Explainability:** $\sim 0.06 \text{ ms}$
- **Single Candidate Total Time:** $\sim 40.9 \text{ ms}$
- **Batch Processing Rate:** $\sim 69.0 \text{ ms}$ per candidate.

---

### 23. Model Evaluation Metrics (Controlled Dataset)
Evaluated against ground-truth labeled technical datasets:
- **Skill Extraction:** Precision: 1.0000 | Recall: 1.0000 | F1-Score: 1.0000 | Accuracy: 1.0000
- **Semantic Matching:** Precision: 1.0000 | Recall: 1.0000 | F1-Score: 1.0000 | Accuracy: 1.0000

---

### 24. Key Advantages
1. Eliminates keyword rigidity through neural semantic embeddings.
2. Prevents keyword stuffing via contextual evidence validation.
3. Mitigates demographic bias through automated PII anonymization.
4. Provides full transparency with explainable rationales and audit logs.
5. High throughput and lightweight deployment with zero external API dependencies.

---

### 25. Limitations
- Scanned/image-only PDFs require machine-readable text (OCR integration can be added).
- PII masking is best-effort and cannot guarantee removal of subtle proxy cues in free-form narrative.

---

### 26. Future Scope
- Integration of Tesseract OCR for scanned PDF ingestion.
- Interactive UI weight tuning sliders for real-time recruiter customization.
- Integration of active learning loops incorporating recruiter hiring feedback.

---

### 27. Conclusion
The AI-Driven Resume Screening and Skill Validation System successfully achieves all project aims. By combining semantic matching, evidence validation, PII anonymization, and explainable scoring into a cohesive, privacy-preserving architecture, the system establishes a fair, transparent, and highly effective AI recruitment tool.
