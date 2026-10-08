# AI-Driven Resume Screening and Skill Validation System with Bias Reduction Techniques

## Project Aim
Develop an AI-powered recruitment system that intelligently screens resumes, validates candidate skills against verifiable contextual evidence, minimizes hiring bias through PII anonymization and fairness metrics, and assists recruiters in selecting suitable candidates through fair, transparent, and explainable evaluation.

---

## 🌟 Key Academic Novelties

1. **Semantic Candidate-Job Matching (Sentence-BERT / Embeddings)**
   - Goes beyond rigid keyword matching by generating dense semantic embeddings for job requirements and candidate profiles.
   - Computes cosine similarity to capture conceptual alignment (e.g. recognizing that *"Built NLP applications using transformers"* strongly satisfies *"Natural Language Processing"*).

2. **Skill Evidence Validation (Contextual Depth Scoring)**
   - Candidates do not receive full credit merely because a skill appears in a skills listing.
   - Evaluates evidence strength across 4 levels:
     - **Level 0 (Missing):** No evidence found (Weight: 0.0)
     - **Level 1 (Low):** Mentioned in skills list only (Weight: 0.3)
     - **Level 2 (Moderate):** Supported by project, certification, or course (Weight: 0.7)
     - **Level 3 (High):** Supported by substantial work experience or deep project achievements (Weight: 1.0)

3. **Bias Reduction & Anonymized Evaluation**
   - Strips and masks Personally Identifiable Information (PII) including Candidate Name, Email, Phone Number, Physical Address, URLs, and Dates of Birth before evaluation.
   - Operates on recruiter-facing anonymized identifiers (e.g., `ANON-XXXX`).
   - Evaluates batch score distributions with Fairlearn demographic disparity metrics.
   - *Academic Note:* PII detection and masking is a best-effort bias-reduction technique designed to mitigate cognitive and demographic bias during initial screening; it does not claim to eliminate societal bias entirely.

4. **Transparent & Explainable Scoring (XAI)**
   - Transparent multi-factor weighted scoring model with no magic numbers:
     - **Semantic Skill Match:** 35%
     - **Skill Evidence Validation:** 25%
     - **Experience Relevance:** 20%
     - **Education Relevance:** 10%
     - **Projects & Certifications:** 10%
   - Generates detailed, data-driven rationales explaining *why* each candidate received their score, highlighting matched skills, evidence snippets, and skill gaps.

---

## 🔄 System Architecture & Logical Flow

```
START
  ↓
Resume + Job Description
  ↓
Text Preprocessing & Normalization
  ↓
Information Extraction
  ├── Candidate Profile (Skills, Education, Experience, Certifications, Projects)
  └── Job Requirements (Required Skills, Preferred Skills, Experience, Education)
  ↓
Semantic Skill Matching (Sentence-BERT & Cosine Similarity)
  ↓
Skill Evidence Validation (4-Level Contextual Evidence Scoring)
  ↓
Bias Reduction (PII Masking & Merit Attribute Isolation)
  ↓
Candidate Score Generation (Transparent Weighted Formula)
  ↓
Explainable Recommendation (Data-Driven XAI Breakdown)
  ↓
Candidate Ranking (Ordered by Score Descending)
  ↓
Recruiter Dashboard (Streamlit UI & Plotly Analytics)
  ↓
END
```

---

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| **Programming Language** | Python 3.10+ / 3.11+ / 3.13 |
| **User Interface** | Streamlit |
| **Database** | SQLite (`database/screening_system.db`) |
| **PDF Extraction** | PyMuPDF (`fitz`) / `pdfplumber` / `PyPDF2` |
| **DOCX Extraction** | `python-docx` |
| **NLP Engine** | `spaCy` & Custom Rule-Based Information Extractors |
| **Skill Taxonomy** | Domain Skill Taxonomy & Alias Normalization Dictionary |
| **Semantic AI & Embeddings** | Sentence-BERT (`sentence-transformers`) / Scikit-learn Cosine Embeddings |
| **Similarity Metrics** | Cosine Similarity (`scikit-learn`) |
| **Fairness & Bias Audit** | `Fairlearn` & Parity Disparity Metrics |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Plotly Express & Plotly Graph Objects |
| **Testing** | `pytest` & Python `unittest` (25 tests, 100% passing) |
| **Configuration** | `python-dotenv` / `.env` |

---

## 📂 Project Structure

```
S7 mini_project/
│
├── app.py                             # Streamlit Recruiter Dashboard
├── config.py                          # Central configuration and scoring weights
├── database.py                        # SQLite persistence & database operations
│
├── database/
│   └── schema.sql                     # Database schema definition
│
├── modules/
│   ├── document_processing/           # Document validation, extraction, cleaning, PII masking
│   │   ├── file_validator.py
│   │   ├── pdf_parser.py
│   │   ├── docx_parser.py
│   │   ├── text_cleaner.py
│   │   ├── pii_detector.py
│   │   ├── anonymizer.py
│   │   └── document_processor.py
│   │
│   ├── job_processing/                # Job description ingestion & requirement extraction
│   │   ├── job_parser.py
│   │   └── requirement_extractor.py
│   │
│   ├── nlp/                           # NLP information extraction & skill taxonomy
│   │   ├── text_preprocessor.py
│   │   ├── skill_taxonomy.py
│   │   ├── skill_extractor.py
│   │   └── information_extractor.py
│   │
│   ├── semantic_matching/             # SBERT embeddings & cosine similarity matcher
│   │   ├── embedding_model.py
│   │   ├── similarity.py
│   │   └── matcher.py
│   │
│   ├── skill_validation/              # Evidence extractor & contextual validation
│   │   ├── evidence_extractor.py
│   │   └── skill_validator.py
│   │
│   ├── fairness/                      # Bias reduction pipeline & Fairlearn audit
│   │   ├── bias_reduction.py
│   │   └── fairness_metrics.py
│   │
│   ├── scoring/                       # Multi-factor scoring & ranking engines
│   │   ├── scoring_engine.py
│   │   └── ranking_engine.py
│   │
│   ├── explainability/                # Natural language explanation generator
│   │   └── explanation_engine.py
│   │
│   ├── models/                        # Typed domain data models
│   │   ├── candidate.py
│   │   ├── job.py
│   │   └── result.py
│   │
│   ├── services/                      # Central screening orchestrator service
│   │   └── screening_service.py
│   │
│   └── utils/
│       └── logging_utils.py           # Privacy-safe logging utility (no PII in logs)
│
├── data/
│   ├── sample_job_descriptions/       # Sample jobs (Senior ML Engineer, Fullstack Dev)
│   │   └── senior_ml_engineer.txt
│   └── sample_resumes/                # 3 Sample resumes for academic demo
│       ├── candidate_a_alex_turner.txt
│       ├── candidate_b_priya_sharma.txt
│       └── candidate_c_john_doe.txt
│
├── tests/                             # Comprehensive test suite (25 Unit & Integration tests)
│   ├── test_file_validator.py
│   ├── test_document_processing.py
│   ├── test_pii_anonymizer.py
│   ├── test_job_processing.py
│   ├── test_nlp_extraction.py
│   ├── test_skill_taxonomy.py
│   ├── test_semantic_matching.py
│   ├── test_skill_validation.py
│   ├── test_fairness_bias.py
│   ├── test_scoring_ranking.py
│   ├── test_explainability.py
│   ├── test_database.py
│   └── test_screening_service_e2e.py
│
├── requirements.txt                   # Project dependencies
├── .env.example                       # Environment configuration template
├── .env                              # Active environment configuration
└── README.md                          # Project documentation
```

---

## 🚀 Installation & Setup

### 1. Clone or Open the Repository
```bash
cd "c:\Users\jo_te\Desktop\S7 mini_project"
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env` (or modify `.env` directly):
```bash
cp .env.example .env
```

---

## 🧪 Running the Test Suite

Execute the test suite using Python's unittest runner or pytest:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

All 25 unit and end-to-end integration tests execute and pass:
- File validation (valid, empty, corrupted, extension check)
- PII detection & masking (name, email, phone, address, dates)
- Skill taxonomy normalization & alias matching
- Information extraction (degrees, years of experience, certifications, projects)
- Job requirement extraction
- Semantic skill matching & cosine similarity
- Skill evidence extraction across 4 confidence tiers
- Bias reduction attribute isolation
- Multi-factor scoring & ranking engine
- Explainability generation
- SQLite database persistence
- End-to-end pipeline validation with 3 candidates

---

## 🖥️ Running the Recruiter Dashboard

Launch the Streamlit web application:

```bash
streamlit run app.py
```
or with a specific Python interpreter:
```bash
python -m streamlit run app.py
```

---

## 📖 How to Use the System

1. **Academic Demo Mode:**
   - In the sidebar, click **"🚀 Load Sample Job & 3 Resumes"**.
   - The system automatically loads a *Senior ML Engineer* position and screens 3 candidates:
     - **Candidate A (Alex Turner):** High match (~89.6%, Rank #1)
     - **Candidate B (Priya Sharma):** Moderate match (~54.8%, Rank #2)
     - **Candidate C (John Doe):** Low match (~37.2%, Rank #3)
2. **Custom Screening:**
   - Paste or upload a Job Description in **Tab 1**.
   - Upload candidate resumes (PDF, DOCX, or TXT).
   - Click **"🚀 Start Screening & Evaluation Pipeline"**.
3. **View Leaderboard & Profiles:**
   - Navigate to **Tab 2** to inspect ranked candidate scores, matched/missing skills, verified evidence snippets, and explainable rationales.
   - Toggle **"Reveal Candidate Names"** in the sidebar to switch between anonymized IDs and names.
4. **Visual Analytics:**
   - Navigate to **Tab 3** to explore Plotly interactive score comparisons and 5-pillar breakdown charts.
5. **Fairness Audit & Export:**
   - Navigate to **Tab 4** to review Fairlearn demographic parity metrics and download the screening leaderboard as CSV.

---

## 🔒 Privacy & Compliance
- **Zero PII in Logs:** Custom `PrivacyFilter` ensures candidate emails, phone numbers, and raw resume texts are sanitized before logging.
- **Anonymized Processing:** Evaluates candidates purely on merit-based qualifications.

---

## ⚖️ Limitations & Future Enhancements
- **Optical Character Recognition (OCR):** Scanned/image-only PDFs currently prompt for a text-based format; future versions can integrate Tesseract OCR.
- **Dynamic Weight Tuning:** Recruiter sliders in the UI for on-the-fly customized weight adjustment.
