"""
AI-Driven Resume Screening and Skill Validation System.
Interactive Recruiter Dashboard Application - Light & White Theme.
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from modules.services.screening_service import ScreeningService
from modules.models.result import BatchScreeningResult, CandidateScreeningResult
from modules.document_processing.file_validator import ValidationError
from config import SCORING_WEIGHTS, DATABASE_PATH
import database

# Page Configuration - Standard expanded sidebar
st.set_page_config(
    page_title="AI Resume Screening & Skill Validation",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling - High-contrast Light/White Theme with larger readable fonts
st.markdown(
    """
    <style>
    /* Force Crisp White App Background */
    .stApp {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        font-size: 1.12rem !important;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3.5rem !important;
        max-width: 95% !important;
    }

    /* Large Header & Subtitle */
    .hero-banner {
        background: linear-gradient(135deg, #EFF6FF 0%, #F8FAFC 100%);
        border: 2px solid #DBEAFE;
        border-radius: 14px;
        padding: 24px 30px;
        margin-bottom: 24px;
        box-shadow: 0 4px 16px rgba(37, 99, 235, 0.06);
    }
    .hero-title {
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        color: #1E3A8A !important;
        line-height: 1.25 !important;
        margin-bottom: 8px !important;
    }
    .hero-subtitle {
        font-size: 1.25rem !important;
        color: #475569 !important;
        font-weight: 500 !important;
        margin-bottom: 0px !important;
    }

    /* Section Headings */
    .section-header {
        font-size: 1.7rem !important;
        font-weight: 750 !important;
        color: #0F172A !important;
        margin-top: 1.8rem !important;
        margin-bottom: 1.0rem !important;
        padding-bottom: 6px !important;
        border-bottom: 2px solid #E2E8F0 !important;
    }

    /* KPI Stat Cards (White Background with Crisp Borders & Shadows) */
    .kpi-card {
        background-color: #FFFFFF !important;
        border: 2px solid #E2E8F0 !important;
        border-radius: 12px !important;
        padding: 20px 16px !important;
        text-align: center !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.05) !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 22px rgba(37, 99, 235, 0.12) !important;
        border-color: #93C5FD !important;
    }
    .kpi-value {
        font-size: 2.7rem !important;
        font-weight: 850 !important;
        color: #1D4ED8 !important;
        line-height: 1.1 !important;
    }
    .kpi-label {
        font-size: 1.0rem !important;
        font-weight: 650 !important;
        color: #64748B !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        margin-top: 8px !important;
    }

    /* Status Badges - Large and Vibrant */
    .badge-high {
        background-color: #DCFCE7 !important;
        color: #15803D !important;
        padding: 6px 16px !important;
        border-radius: 9999px !important;
        font-weight: 750 !important;
        font-size: 1.05rem !important;
        border: 1px solid #86EFAC !important;
        display: inline-block;
    }
    .badge-rel {
        background-color: #DBEAFE !important;
        color: #1D4ED8 !important;
        padding: 6px 16px !important;
        border-radius: 9999px !important;
        font-weight: 750 !important;
        font-size: 1.05rem !important;
        border: 1px solid #93C5FD !important;
        display: inline-block;
    }
    .badge-mod {
        background-color: #FEF3C7 !important;
        color: #B45309 !important;
        padding: 6px 16px !important;
        border-radius: 9999px !important;
        font-weight: 750 !important;
        font-size: 1.05rem !important;
        border: 1px solid #FDE68A !important;
        display: inline-block;
    }
    .badge-low {
        background-color: #FEE2E2 !important;
        color: #B91C1C !important;
        padding: 6px 16px !important;
        border-radius: 9999px !important;
        font-weight: 750 !important;
        font-size: 1.05rem !important;
        border: 1px solid #FCA5A5 !important;
        display: inline-block;
    }

    /* Skill Tags */
    .skill-tag {
        display: inline-block;
        background-color: #EFF6FF !important;
        color: #1E40AF !important;
        padding: 6px 14px !important;
        border-radius: 8px !important;
        margin: 4px !important;
        font-size: 1.02rem !important;
        font-weight: 600 !important;
        border: 1px solid #BFDBFE !important;
    }
    .missing-tag {
        display: inline-block;
        background-color: #FEF2F2 !important;
        color: #991B1B !important;
        padding: 6px 14px !important;
        border-radius: 8px !important;
        margin: 4px !important;
        font-size: 1.02rem !important;
        font-weight: 600 !important;
        border: 1px solid #FECACA !important;
    }

    /* Candidate Detail Card */
    .profile-card {
        background-color: #FFFFFF !important;
        border: 2px solid #E2E8F0 !important;
        border-radius: 14px !important;
        padding: 24px !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.04) !important;
        margin-bottom: 20px !important;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #F8FAFC !important;
        border-right: 2px solid #E2E8F0 !important;
    }
    .sidebar-title {
        font-size: 1.5rem !important;
        font-weight: 800 !important;
        color: #1E3A8A !important;
        margin-bottom: 12px !important;
    }

    /* Increase Streamlit General Font Sizes */
    p, span, label, .stMarkdown, .stSelectbox, .stCheckbox {
        font-size: 1.1rem !important;
    }
    button[kind="primary"] {
        font-size: 1.25rem !important;
        font-weight: 700 !important;
        padding: 12px 28px !important;
        border-radius: 10px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_screening_service():
    """Initializes and caches the screening service."""
    return ScreeningService()


def load_sample_data():
    """Loads pre-configured sample files (preferring DOCX) for instant demonstration."""
    sample_job_docx = PROJECT_ROOT / "data" / "sample_job_descriptions" / "senior_ml_engineer.docx"
    sample_job_txt = PROJECT_ROOT / "data" / "sample_job_descriptions" / "senior_ml_engineer.txt"
    sample_resumes_dir = PROJECT_ROOT / "data" / "sample_resumes"

    job_input = sample_job_docx if sample_job_docx.exists() else sample_job_txt
    job_fname = job_input.name

    resumes = []
    filenames = []

    docx_resumes = sorted(sample_resumes_dir.glob("*.docx")) if sample_resumes_dir.exists() else []
    if docx_resumes:
        for rpath in docx_resumes:
            resumes.append(rpath)
            filenames.append(rpath.name)
    elif sample_resumes_dir.exists():
        for rpath in sorted(sample_resumes_dir.glob("*.txt")):
            resumes.append(rpath)
            filenames.append(rpath.name)

    return job_input, resumes, filenames, job_fname


def render_sidebar():
    """Renders the recruiter sidebar with controls, weights, and shortcuts."""
    st.sidebar.markdown('<div class="sidebar-title">🏢 Recruiter Portal</div>', unsafe_allow_html=True)
    st.sidebar.caption("Intelligent Candidate Screening & Skill Validation")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚡ Quick Actions")
    load_demo = st.sidebar.button(
        "🚀 Load Demo Dataset",
        use_container_width=True,
        type="primary",
        help="Loads Senior ML Engineer position and 3 candidates (Alex Turner, Priya Sharma, John Doe)",
    )

    reset_btn = st.sidebar.button(
        "🔄 Clear / Reset Dashboard",
        use_container_width=True,
        help="Clears active job and screening results",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 5-Pillar Scoring Framework")
    st.sidebar.markdown(
        f"""
        - 🧠 **Semantic Skills:** `{SCORING_WEIGHTS['semantic_skill_match']*100:.0f}%`
        - 🔬 **Skill Evidence:** `{SCORING_WEIGHTS['skill_evidence']*100:.0f}%`
        - ⏳ **Experience Tenure:** `{SCORING_WEIGHTS['experience_relevance']*100:.0f}%`
        - 🎓 **Education Level:** `{SCORING_WEIGHTS['education_relevance']*100:.0f}%`
        - 🛠️ **Projects & Certs:** `{SCORING_WEIGHTS['projects_certs']*100:.0f}%`
        """
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🛡️ System Specifications")
    st.sidebar.caption("• **Semantic Engine:** Sentence-BERT / MiniLM")
    st.sidebar.caption("• **Document Parsers:** PyMuPDF & python-docx")
    st.sidebar.caption("• **Persistence:** SQLite Database Active")

    return load_demo, reset_btn


def main():
    # Session state initialization
    if "screening_results" not in st.session_state:
        st.session_state.screening_results = None
    if "job_text_input" not in st.session_state:
        st.session_state.job_text_input = ""
    if "show_names" not in st.session_state:
        st.session_state.show_names = False

    service = get_screening_service()
    load_demo, reset_btn = render_sidebar()

    if reset_btn:
        st.session_state.screening_results = None
        st.session_state.job_text_input = ""
        st.rerun()

    # Handle demo loading
    if load_demo:
        demo_job, demo_resumes, demo_filenames, demo_job_name = load_sample_data()
        with st.spinner("Processing demo candidates through the AI screening pipeline..."):
            results = service.process_screening(
                job_input=demo_job,
                resume_files=demo_resumes,
                job_filename=demo_job_name,
                resume_filenames=demo_filenames,
                job_title="Senior Machine Learning Engineer",
            )
            st.session_state.screening_results = results
            st.success("✅ Demo dataset processed! Scroll down to inspect the candidate results.")

    # Top Hero Banner
    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-title">🎯 AI-Driven Resume Screening & Skill Validation System</div>
            <div class="hero-subtitle">Fair, Explainable, Semantic Candidate Matching with Contextual Evidence Verification</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ==========================================
    # SECTION 1: JOB & RESUME INGESTION
    # ==========================================
    st.markdown('<div class="section-header">📥 1. Job Description & Candidate Resumes</div>', unsafe_allow_html=True)

    col_jd, col_resumes = st.columns([1, 1], gap="large")

    with col_jd:
        st.markdown("#### 📋 Step 1: Job Requirements")
        job_input_mode = st.radio("Choose Input Method:", ["Paste Job Text", "Upload Job Document (PDF/DOCX)"], horizontal=True)

        job_file = None
        job_text = ""

        if job_input_mode == "Paste Job Text":
            job_text = st.text_area(
                "Job Description Content:",
                value=st.session_state.job_text_input,
                height=220,
                placeholder="Paste the target job description, qualifications, required skills, and responsibilities...",
            )
        else:
            job_file = st.file_uploader(
                "Upload Job Description File:",
                type=["pdf", "docx", "txt"],
                key="jd_file_uploader_white",
            )
            if job_file:
                st.caption(f"📄 Selected file: **{job_file.name}**")

    with col_resumes:
        st.markdown("#### 📂 Step 2: Candidate Resumes")
        resume_uploads = st.file_uploader(
            "Upload Resumes (PDF, DOCX, TXT - Multiple Files Allowed):",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            key="resume_uploader_white",
            help="Select one or multiple resumes to screen against the job description",
        )

        if resume_uploads:
            st.info(f"📁 **{len(resume_uploads)} candidate resume(s)** uploaded and ready for evaluation.")
        else:
            st.caption("Upload candidate resumes here, or click **'🚀 Load Demo Dataset'** in the sidebar to test instantly.")

    st.markdown("<br>", unsafe_allow_html=True)
    process_btn = st.button("🚀 Run Screening & Evaluation Pipeline", type="primary", use_container_width=True)

    if process_btn:
        active_job_input = job_file if job_file is not None else job_text
        job_fname = job_file.name if job_file else "Job_Description.txt"

        if not active_job_input or (isinstance(active_job_input, str) and not active_job_input.strip()):
            st.error("⚠️ Please provide a Job Description (paste text or upload a document).")
        elif not resume_uploads:
            st.error("⚠️ Please upload at least one candidate resume to screen.")
        else:
            try:
                with st.spinner("Executing semantic matching, skill evidence validation, and candidate scoring..."):
                    resumes_data = [r.read() for r in resume_uploads]
                    resume_fnames = [r.name for r in resume_uploads]
                    batch_res = service.process_screening(
                        job_input=active_job_input,
                        resume_files=resumes_data,
                        job_filename=job_fname,
                        resume_filenames=resume_fnames,
                    )
                    st.session_state.screening_results = batch_res
                    st.success(f"✅ Screening complete for {batch_res.total_candidates} candidate(s)! Scroll down to view the full dashboard.")
            except ValidationError as ve:
                st.error(f"Validation Error: {str(ve)}")
            except Exception as e:
                st.error(f"Processing Error: {str(e)}")

    # Retrieve current results
    results: BatchScreeningResult = st.session_state.screening_results
    show_names = st.session_state.get("show_names", False)

    # ==========================================
    # SCROLLING DOWN FOR RESULT: REAL DASHBOARD
    # ==========================================
    if results and results.ranked_candidates:
        st.markdown("<br>", unsafe_allow_html=True)

        # ==========================================
        # SECTION 2: EXECUTIVE KPI SUMMARY (4 CARDS)
        # ==========================================
        st.markdown(f'<div class="section-header">📊 2. Executive Overview: {results.job_title}</div>', unsafe_allow_html=True)

        k1, k2, k3, k4 = st.columns(4, gap="medium")
        with k1:
            st.markdown(
                f'<div class="kpi-card"><div class="kpi-value">{results.total_candidates}</div><div class="kpi-label">Candidates Evaluated</div></div>',
                unsafe_allow_html=True,
            )
        with k2:
            top_candidate = results.ranked_candidates[0]
            top_display = top_candidate.candidate_name if show_names else top_candidate.anonymized_id
            st.markdown(
                f'<div class="kpi-card"><div class="kpi-value">{top_candidate.overall_score:.1f}%</div><div class="kpi-label">Top Score ({top_display})</div></div>',
                unsafe_allow_html=True,
            )
        with k3:
            high_count = sum(1 for c in results.ranked_candidates if c.recommendation == "HIGH RELEVANCE")
            st.markdown(
                f'<div class="kpi-card"><div class="kpi-value">{high_count}</div><div class="kpi-label">High Relevance Matches</div></div>',
                unsafe_allow_html=True,
            )
        with k4:
            avg_score = sum(c.overall_score for c in results.ranked_candidates) / len(results.ranked_candidates)
            st.markdown(
                f'<div class="kpi-card"><div class="kpi-value">{avg_score:.1f}%</div><div class="kpi-label">Average Match Score</div></div>',
                unsafe_allow_html=True,
            )

        # ==========================================
        # SECTION 3: CANDIDATE LEADERBOARD & RANKING
        # ==========================================
        st.markdown('<div class="section-header">🏆 3. Candidate Ranking & Leaderboard</div>', unsafe_allow_html=True)

        filter_col1, filter_col2 = st.columns([2, 2], gap="large")
        with filter_col1:
            rec_filters = st.multiselect(
                "Filter by Recommendation Tier:",
                options=["HIGH RELEVANCE", "RELEVANT", "MODERATE RELEVANCE", "LOW RELEVANCE"],
                default=["HIGH RELEVANCE", "RELEVANT", "MODERATE RELEVANCE", "LOW RELEVANCE"],
            )
        with filter_col2:
            score_filter = st.slider("Minimum Match Score (%)", min_value=0.0, max_value=100.0, value=0.0, step=5.0)

        filtered_list = [
            c for c in results.ranked_candidates
            if c.recommendation in rec_filters and c.overall_score >= score_filter
        ]

        leaderboard_rows = []
        for c in filtered_list:
            c_name_str = c.candidate_name if show_names else c.anonymized_id
            leaderboard_rows.append({
                "Rank": f"#{c.rank}",
                "Candidate ID": c.anonymized_id,
                "Candidate Name": c_name_str,
                "Overall Score": f"{c.overall_score:.1f}%",
                "Semantic Skills": f"{c.score_breakdown.semantic_skill_match_score:.1f}%" if c.score_breakdown else "N/A",
                "Skill Evidence": f"{c.score_breakdown.skill_evidence_score:.1f}%" if c.score_breakdown else "N/A",
                "Experience Score": f"{c.score_breakdown.experience_relevance_score:.1f}%" if c.score_breakdown else "N/A",
                "Recommendation": c.recommendation,
            })

        df_leaderboard = pd.DataFrame(leaderboard_rows)
        st.dataframe(df_leaderboard, use_container_width=True, hide_index=True)

        # ==========================================
        # SECTION 4: CANDIDATE PROFILE & EVIDENCE DEEP-DIVE
        # ==========================================
        st.markdown('<div class="section-header">🔍 4. Candidate Profile & Verified Evidence Deep-Dive</div>', unsafe_allow_html=True)

        candidate_labels = [
            f"Rank #{c.rank} - {c.candidate_name if show_names else c.anonymized_id} (Score: {c.overall_score:.1f}%)"
            for c in results.ranked_candidates
        ]
        chosen_idx = st.selectbox(
            "Select Candidate to Inspect:",
            range(len(candidate_labels)),
            format_func=lambda i: candidate_labels[i],
        )
        selected_cand = results.ranked_candidates[chosen_idx]
        selected_display_name = selected_cand.candidate_name if show_names else selected_cand.anonymized_id

        detail_left, detail_right = st.columns([1, 2], gap="large")

        with detail_left:
            st.markdown(f"### 👤 {selected_display_name}")
            st.caption(f"Candidate ID: `{selected_cand.anonymized_id}`")

            rec_badge_class = {
                "HIGH RELEVANCE": "badge-high",
                "RELEVANT": "badge-rel",
                "MODERATE RELEVANCE": "badge-mod",
                "LOW RELEVANCE": "badge-low",
            }.get(selected_cand.recommendation, "badge-mod")

            st.markdown(
                f'<div style="margin: 12px 0;"><span class="{rec_badge_class}">{selected_cand.recommendation}</span></div>',
                unsafe_allow_html=True,
            )
            st.progress(min(1.0, selected_cand.overall_score / 100.0), text=f"Overall Match: {selected_cand.overall_score:.1f}%")

            if selected_cand.score_breakdown:
                st.markdown("#### 📊 5-Pillar Score Breakdown")
                st.markdown(
                    f"""
                    - 🧠 **Semantic Skill Match:** `{selected_cand.score_breakdown.semantic_skill_match_score:.1f}%`
                    - 🔬 **Skill Evidence Validation:** `{selected_cand.score_breakdown.skill_evidence_score:.1f}%`
                    - ⏳ **Experience Relevance:** `{selected_cand.score_breakdown.experience_relevance_score:.1f}%`
                    - 🎓 **Education Relevance:** `{selected_cand.score_breakdown.education_relevance_score:.1f}%`
                    - 🛠️ **Projects & Certs:** `{selected_cand.score_breakdown.projects_certs_score:.1f}%`
                    """
                )

        with detail_right:
            st.markdown("#### 💡 Explainable AI Rationale")
            st.info(f"📢 {selected_cand.explanation}")

            st.markdown("#### 🎯 Skill Alignment Breakdown")
            col_match, col_miss = st.columns(2)

            with col_match:
                st.markdown("**✅ Matched & Partial Skills:**")
                all_matched_skills = selected_cand.matched_skills + selected_cand.partial_skills
                if all_matched_skills:
                    for m in all_matched_skills:
                        sim_pct = f"{m.similarity_score * 100:.0f}%" if m.similarity_score else "100%"
                        st.markdown(f'<span class="skill-tag">✓ {m.job_skill} ({sim_pct})</span>', unsafe_allow_html=True)
                else:
                    st.write("No matching skills identified.")

            with col_miss:
                st.markdown("**❌ Missing Skills:**")
                if selected_cand.missing_skills:
                    for m in selected_cand.missing_skills:
                        st.markdown(f'<span class="missing-tag">✗ {m.job_skill}</span>', unsafe_allow_html=True)
                else:
                    st.write("No major required skills missing.")

            st.markdown("#### 🔬 Verified Skill Evidence (Contextual Validation)")
            if selected_cand.evidence_items:
                ev_table_rows = []
                for ev in selected_cand.evidence_items:
                    ev_table_rows.append({
                        "Skill": ev.skill_name,
                        "Evidence Level": f"Level {ev.evidence_level}",
                        "Confidence": ev.confidence_label,
                        "Source Section": ev.source_section,
                        "Contextual Snippet": ev.snippet,
                    })
                st.dataframe(pd.DataFrame(ev_table_rows), use_container_width=True, hide_index=True)
            else:
                st.caption("No contextual evidence records found.")

            if selected_cand.relevant_projects:
                st.markdown("#### 📁 Highlighted Projects Found")
                for p in selected_cand.relevant_projects[:4]:
                    st.markdown(f"- **{p}**")

        # ==========================================
        # SECTION 5: VISUAL ANALYTICS (ONLY 4 BARS & NAME REVEAL)
        # ==========================================
        st.markdown('<div class="section-header">📊 5. Visual Analytics & Candidate Benchmark</div>', unsafe_allow_html=True)

        va_col1, va_col2 = st.columns([1.5, 1], gap="large")

        with va_col1:
            # NAME REVEAL CHECKBOX IN VISUAL ANALYTICS
            reveal_names_checkbox = st.checkbox(
                "👁️ Reveal Candidate Real Names across Visual Analytics & Leaderboard",
                value=st.session_state.show_names,
                key="name_reveal_visual_analytics",
                help="Check to display real candidate names instead of anonymized IDs.",
            )
            if reveal_names_checkbox != st.session_state.show_names:
                st.session_state.show_names = reveal_names_checkbox
                st.rerun()

        with va_col2:
            st.caption("Displaying top candidate competencies and comparative visual analytics.")

        # ONLY 4 BARS AVAILABLE AS REQUESTED
        top_4_candidates = results.ranked_candidates[:4]
        bar_names = [c.candidate_name if st.session_state.show_names else c.anonymized_id for c in top_4_candidates]
        bar_scores = [c.overall_score for c in top_4_candidates]
        bar_colors = ['#16A34A' if s >= 80 else '#2563EB' if s >= 65 else '#D97706' if s >= 50 else '#DC2626' for s in bar_scores]

        st.markdown(f"#### 🏅 Top {len(top_4_candidates)} Ranked Candidates Match Comparison (4-Bar Benchmark)")

        fig_4bars = go.Figure(
            data=[go.Bar(
                x=bar_names,
                y=bar_scores,
                marker=dict(color=bar_colors, line=dict(color='#1E293B', width=1.5)),
                text=[f"{s:.1f}%" for s in bar_scores],
                textposition='auto',
                textfont=dict(size=16, color='#FFFFFF', family="Arial Black"),
            )]
        )
        fig_4bars.update_layout(
            title=f"Overall Match Scores (Top {len(top_4_candidates)} Candidates)",
            xaxis_title="Candidate",
            yaxis_title="Overall Match Score (%)",
            yaxis=dict(range=[0, 100], gridcolor='#E2E8F0'),
            xaxis=dict(gridcolor='#E2E8F0'),
            template="plotly_white",
            height=420,
            font=dict(size=14, color="#0F172A"),
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
        )
        st.plotly_chart(fig_4bars, use_container_width=True)

        # 5-Pillar Competency Breakdown for the Top Candidates
        st.markdown(f"#### 🔍 5-Pillar Competency Breakdown (Top {len(top_4_candidates)} Candidates)")
        pillar_rows = []
        for c in top_4_candidates:
            c_label = c.candidate_name if st.session_state.show_names else c.anonymized_id
            if c.score_breakdown:
                pillar_rows.append({"Candidate": c_label, "Competency Pillar": "Semantic Skills", "Score": c.score_breakdown.semantic_skill_match_score})
                pillar_rows.append({"Candidate": c_label, "Competency Pillar": "Skill Evidence", "Score": c.score_breakdown.skill_evidence_score})
                pillar_rows.append({"Candidate": c_label, "Competency Pillar": "Experience", "Score": c.score_breakdown.experience_relevance_score})
                pillar_rows.append({"Candidate": c_label, "Competency Pillar": "Education", "Score": c.score_breakdown.education_relevance_score})
                pillar_rows.append({"Candidate": c_label, "Competency Pillar": "Projects & Certs", "Score": c.score_breakdown.projects_certs_score})

        if pillar_rows:
            df_pillars = pd.DataFrame(pillar_rows)
            fig_pillars = px.bar(
                df_pillars,
                x="Candidate",
                y="Score",
                color="Competency Pillar",
                barmode="group",
                title=f"5-Pillar Score Breakdown (Top {len(top_4_candidates)} Candidates)",
                template="plotly_white",
                color_discrete_sequence=["#2563EB", "#7C3AED", "#059669", "#D97706", "#DB2777"],
            )
            fig_pillars.update_layout(
                yaxis=dict(range=[0, 100], gridcolor='#E2E8F0'),
                xaxis=dict(gridcolor='#E2E8F0'),
                height=420,
                font=dict(size=14, color="#0F172A"),
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
            )
            st.plotly_chart(fig_pillars, use_container_width=True)

        # ==========================================
        # SECTION 6: FAIRNESS & EXPORT
        # ==========================================
        st.markdown('<div class="section-header">⚖️ 6. Fairness Audit & Leaderboard Export</div>', unsafe_allow_html=True)

        audit_col1, audit_col2 = st.columns([1.5, 1], gap="large")

        with audit_col1:
            st.info(
                "**Merit-Based Assessment Policy:** Candidate evaluation operates strictly on verified technical competencies, "
                "contextual experience, education, and project achievements."
            )
            f_metrics = results.fairness_metrics
            fc1, fc2, fc3, fc4 = st.columns(4)
            with fc1:
                st.metric("Total Evaluated", f_metrics.get("total_candidates_audited", 0))
            with fc2:
                st.metric("Selection Rate", f"{f_metrics.get('selection_rate_pct', 0.0):.1f}%")
            with fc3:
                st.metric("Mean Score", f"{f_metrics.get('score_mean', 0.0):.1f}")
            with fc4:
                st.metric("Parity Difference", f"{f_metrics.get('demographic_parity_difference', 0.0):.3f}")

        with audit_col2:
            st.markdown("#### 💾 Export Recruiter Report")
            st.caption("Download the complete evaluated candidate rankings as CSV for offline review or ATS integration.")
            csv_payload = df_leaderboard.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Screening Leaderboard as CSV",
                data=csv_payload,
                file_name="candidate_screening_leaderboard.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True,
            )

    else:
        st.info("👋 Upload a job description and resumes above, or click **'🚀 Load Demo Dataset'** in the sidebar to run the screening engine and view results.")


if __name__ == "__main__":
    main()
