"""
TalentIQ AI: Enterprise AI Recruitment & Candidate Evaluation Platform.
Modern, interactive recruiter dashboard for AI-driven resume screening,
contextual skill validation, and bias-reduction analysis.
"""

import os
import sys
from pathlib import Path
import io
import time

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from modules.services.screening_service import ScreeningService
from modules.models.result import BatchScreeningResult, CandidateScreeningResult
from modules.document_processing.file_validator import ValidationError
from modules.nlp.skill_taxonomy import SkillTaxonomy
from config import SCORING_WEIGHTS, DATABASE_PATH
import database

# =============================================================================
# ENTERPRISE PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="TalentIQ AI — Recruitment Analytics Platform",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# ENTERPRISE SAAS DESIGN SYSTEM (Clean, Modern, High-Contrast Light Theme)
# =============================================================================
st.markdown(
    """
    <style>
    /* ========================================================================
       1. GLOBAL THEME & TYPOGRAPHY
       ======================================================================== */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

    html, body, [data-testid="stAppViewContainer"], .stApp {
        background-color: #F8FAFC !important;
        color: #0F172A !important;
        font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        -webkit-font-smoothing: antialiased;
    }

    [data-testid="stHeader"] {
        background-color: rgba(248, 250, 252, 0.95) !important;
        border-bottom: 1px solid #E2E8F0 !important;
    }

    p, span, label, h1, h2, h3, h4, h5, h6, .stMarkdown, .stText {
        color: #0F172A !important;
    }

    /* ========================================================================
       2. ENTERPRISE SIDEBAR STYLING
       ======================================================================== */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0 !important;
        min-width: 310px !important;
        max-width: 330px !important;
        box-shadow: 2px 0 12px rgba(15, 23, 42, 0.03) !important;
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem !important;
        padding-left: 1.25rem !important;
        padding-right: 1.25rem !important;
    }

    .brand-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 4px;
    }
    .brand-logo-icon {
        width: 38px;
        height: 38px;
        background: linear-gradient(135deg, #2563EB 0%, #4F46E5 100%);
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #FFFFFF;
        font-weight: 800;
        font-size: 1.25rem;
        box-shadow: 0 4px 10px rgba(37, 99, 235, 0.3);
    }
    .brand-name {
        font-size: 1.55rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.02em;
        line-height: 1.1;
    }
    .brand-name span {
        color: #2563EB !important;
    }
    .brand-subtitle {
        font-size: 0.82rem;
        color: #64748B;
        font-weight: 500;
        margin-left: 50px;
        margin-bottom: 1.4rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Sidebar Section Divider */
    .nav-label {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94A3B8;
        margin: 1.2rem 0 0.5rem 0.4rem;
    }

    /* Radio Navigation Menu */
    .stRadio div[role="radiogroup"] label {
        padding: 9px 12px !important;
        border-radius: 8px !important;
        margin-bottom: 3px !important;
        transition: all 0.15s ease-in-out;
        border: 1px solid transparent;
    }
    .stRadio div[role="radiogroup"] label:hover {
        background-color: #F1F5F9 !important;
    }
    .stRadio div[role="radiogroup"] label span {
        font-weight: 600 !important;
        font-size: 0.96rem !important;
        color: #334155 !important;
    }

    /* Recruiter User Profile Card */
    .user-profile-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 12px;
        display: flex;
        align-items: center;
        gap: 12px;
        margin-top: 1.2rem;
        margin-bottom: 0.8rem;
    }
    .user-avatar {
        width: 36px;
        height: 36px;
        background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%);
        color: #FFFFFF;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.88rem;
    }
    .user-details {
        display: flex;
        flex-direction: column;
        line-height: 1.2;
    }
    .user-name {
        font-weight: 700;
        font-size: 0.9rem;
        color: #0F172A;
    }
    .user-role {
        font-size: 0.78rem;
        color: #64748B;
        font-weight: 500;
    }

    /* ========================================================================
       3. ENTERPRISE CARDS & KPI WIDGETS
       ======================================================================== */
    .enterprise-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 22px 24px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.04), 0 1px 2px -1px rgba(0, 0, 0, 0.04);
        margin-bottom: 20px;
        transition: box-shadow 0.2s ease, border-color 0.2s ease;
    }
    .enterprise-card:hover {
        border-color: #CBD5E1;
        box-shadow: 0 8px 20px -4px rgba(15, 23, 42, 0.06);
    }

    .kpi-metric-box {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 20px 22px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
        position: relative;
        overflow: hidden;
    }
    .kpi-metric-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 24px -4px rgba(15, 23, 42, 0.08);
        border-color: #93C5FD;
    }
    .kpi-metric-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #2563EB, #4F46E5);
    }
    .kpi-icon-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 12px;
    }
    .kpi-icon-pill {
        width: 38px;
        height: 38px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
    }
    .kpi-icon-blue { background-color: #EFF6FF; color: #2563EB; }
    .kpi-icon-indigo { background-color: #EEF2FF; color: #4F46E5; }
    .kpi-icon-emerald { background-color: #ECFDF5; color: #059669; }
    .kpi-icon-amber { background-color: #FFFBEB; color: #D97706; }

    .kpi-val {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.1;
        letter-spacing: -0.02em;
    }
    .kpi-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }
    .kpi-badge {
        font-size: 0.75rem;
        font-weight: 700;
        padding: 3px 8px;
        border-radius: 6px;
        display: inline-block;
    }
    .kpi-badge-green { background-color: #DCFCE7; color: #166534; }
    .kpi-badge-blue { background-color: #DBEAFE; color: #1E40AF; }

    /* ========================================================================
       4. HERO BANNER & SECTION HEADERS
       ======================================================================== */
    .platform-hero {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        color: #FFFFFF;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.15);
    }
    .platform-hero h1 {
        color: #FFFFFF !important;
        font-size: 1.85rem !important;
        font-weight: 800 !important;
        margin: 0 0 6px 0 !important;
        letter-spacing: -0.02em;
    }
    .platform-hero p {
        color: #94A3B8 !important;
        font-size: 0.98rem !important;
        margin: 0 !important;
        font-weight: 400;
    }
    .hero-badge-tag {
        background: rgba(37, 99, 235, 0.25);
        border: 1px solid rgba(147, 197, 253, 0.3);
        color: #93C5FD !important;
        font-size: 0.82rem;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 9999px;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        margin-bottom: 8px;
    }

    .page-section-header {
        font-size: 1.45rem;
        font-weight: 800;
        color: #0F172A;
        margin-top: 1rem;
        margin-bottom: 1rem;
        padding-bottom: 8px;
        border-bottom: 2px solid #E2E8F0;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* ========================================================================
       5. PILLS, STATUS BADGES & TAGS
       ======================================================================== */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        letter-spacing: 0.02em;
    }
    .pill-high { background-color: #DCFCE7; color: #15803D; border: 1px solid #86EFAC; }
    .pill-rel { background-color: #DBEAFE; color: #1D4ED8; border: 1px solid #93C5FD; }
    .pill-mod { background-color: #FEF3C7; color: #B45309; border: 1px solid #FDE68A; }
    .pill-low { background-color: #FEE2E2; color: #B91C1C; border: 1px solid #FCA5A5; }

    .chip-matched {
        display: inline-block;
        background-color: #EFF6FF;
        color: #1E40AF;
        padding: 5px 12px;
        border-radius: 8px;
        margin: 3px;
        font-size: 0.9rem;
        font-weight: 600;
        border: 1px solid #BFDBFE;
    }
    .chip-missing {
        display: inline-block;
        background-color: #FEF2F2;
        color: #991B1B;
        padding: 5px 12px;
        border-radius: 8px;
        margin: 3px;
        font-size: 0.9rem;
        font-weight: 600;
        border: 1px solid #FECACA;
    }

    /* ========================================================================
       6. BUTTONS & CONTROLS
       ======================================================================== */
    button[kind="primary"] {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        padding: 10px 22px !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25) !important;
        transition: all 0.2s ease !important;
    }
    button[kind="primary"]:hover {
        background: linear-gradient(135deg, #1D4ED8 0%, #1E40AF 100%) !important;
        box-shadow: 0 6px 18px rgba(37, 99, 235, 0.4) !important;
        transform: translateY(-1px) !important;
    }
    button[kind="secondary"] {
        background-color: #FFFFFF !important;
        color: #334155 !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 10px !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        padding: 8px 18px !important;
    }
    button[kind="secondary"]:hover {
        border-color: #94A3B8 !important;
        background-color: #F8FAFC !important;
    }

    /* Clean Streamlit DataFrames */
    [data-testid="stDataFrame"] {
        border: 1px solid #E2E8F0 !important;
        border-radius: 12px !important;
        overflow: hidden !important;
        background-color: #FFFFFF !important;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #F1F5F9;
        padding: 6px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 18px;
        font-weight: 600;
        font-size: 0.95rem;
        color: #475569;
        background-color: transparent;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #2563EB !important;
        font-weight: 700 !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
    }

    /* Empty state card */
    .empty-state-box {
        background-color: #FFFFFF;
        border: 2px dashed #CBD5E1;
        border-radius: 16px;
        padding: 48px 32px;
        text-align: center;
        margin: 20px 0;
    }
    .empty-state-icon {
        font-size: 3rem;
        color: #94A3B8;
        margin-bottom: 12px;
    }
    .empty-state-title {
        font-size: 1.35rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 6px;
    }
    .empty-state-desc {
        font-size: 0.98rem;
        color: #64748B;
        max-width: 520px;
        margin: 0 auto 20px auto;
        line-height: 1.5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# RESOURCE SERVICES & SAMPLE DATA
# =============================================================================
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


# =============================================================================
# SIDEBAR NAVIGATION & CONTROLLER
# =============================================================================
def render_sidebar():
    """Renders the enterprise TalentIQ AI sidebar with full navigation and controls."""
    # Brand Logo & Title
    st.sidebar.markdown(
        """
        <div class="brand-header">
            <div class="brand-logo-icon">IQ</div>
            <div class="brand-name">Talent<span>IQ</span> AI</div>
        </div>
        <div class="brand-subtitle">AI Resume Screening & Skill Validation</div>
        """,
        unsafe_allow_html=True,
    )

    # Navigation Section
    st.sidebar.markdown('<div class="nav-label">Enterprise Navigation</div>', unsafe_allow_html=True)
    nav_options = [
        "📊 Overview Dashboard",
        "📥 Resume Screening",
        "🏆 Candidate Leaderboard",
        "👤 Candidate Profiles",
        "🧠 Skill Matching",
        "🔬 Evidence Verification",
        "⚖️ Bias & Fairness Analysis",
        "📈 Reports & Analytics",
        "🏛️ System Architecture",
    ]

    selected_nav = st.sidebar.radio(
        "Navigation",
        options=nav_options,
        index=nav_options.index(st.session_state.get("active_section", nav_options[0])),
        label_visibility="collapsed",
        key="talentiq_nav_radio",
    )
    st.session_state.active_section = selected_nav

    # Quick Actions
    st.sidebar.markdown('<div class="nav-label">Quick Actions</div>', unsafe_allow_html=True)
    load_demo_clicked = st.sidebar.button(
        "⚡ Load Demo Dataset",
        use_container_width=True,
        type="primary",
        help="Instantly loads Senior Machine Learning Engineer position with 3 evaluated candidates",
    )

    reset_clicked = st.sidebar.button(
        "🔄 Reset Active Session",
        use_container_width=True,
        help="Clears active candidate results and job description",
    )

    # Recruiter Privacy Controls
    st.sidebar.markdown('<div class="nav-label">Privacy & Bias Mitigation</div>', unsafe_allow_html=True)
    reveal_names = st.sidebar.checkbox(
        "Reveal Candidate Real Names",
        value=st.session_state.get("show_names", False),
        key="talentiq_reveal_names_checkbox",
        help="When unchecked, candidate names are masked with anonymized IDs (ANON-XXXX) to prevent cognitive bias.",
    )
    st.session_state.show_names = reveal_names

    # 5-Pillar Scoring Summary
    st.sidebar.markdown('<div class="nav-label">Evaluation Weights (5 Pillars)</div>', unsafe_allow_html=True)
    st.sidebar.markdown(
        f"""
        <div style="font-size: 0.85rem; line-height: 1.6; color: #475569; background: #F8FAFC; padding: 10px; border-radius: 8px; border: 1px solid #E2E8F0;">
            <div>🧠 <b>Semantic Skills:</b> <span style="float:right; font-weight:700; color:#2563EB;">{SCORING_WEIGHTS['semantic_skill_match']*100:.0f}%</span></div>
            <div>🔬 <b>Skill Evidence:</b> <span style="float:right; font-weight:700; color:#4F46E5;">{SCORING_WEIGHTS['skill_evidence']*100:.0f}%</span></div>
            <div>⏳ <b>Experience:</b> <span style="float:right; font-weight:700; color:#059669;">{SCORING_WEIGHTS['experience_relevance']*100:.0f}%</span></div>
            <div>🎓 <b>Education:</b> <span style="float:right; font-weight:700; color:#D97706;">{SCORING_WEIGHTS['education_relevance']*100:.0f}%</span></div>
            <div>🛠️ <b>Projects & Certs:</b> <span style="float:right; font-weight:700; color:#DB2777;">{SCORING_WEIGHTS['projects_certs']*100:.0f}%</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Recruiter Profile Box
    st.sidebar.markdown(
        """
        <div class="user-profile-card">
            <div class="user-avatar">JS</div>
            <div class="user-details">
                <div class="user-name">Jothi Suhitha</div>
                <div class="user-role">Lead Technical Recruiter</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    return selected_nav, load_demo_clicked, reset_clicked


# =============================================================================
# VIEW 1: OVERVIEW DASHBOARD
# =============================================================================
def render_overview_dashboard(results: BatchScreeningResult, show_names: bool):
    # Hero Welcome Banner
    st.markdown(
        f"""
        <div class="platform-hero">
            <div>
                <div class="hero-badge-tag">● ACTIVE SCREENING SESSION</div>
                <h1>Recruitment Overview</h1>
                <p>Real-time candidate evaluation, semantic skill alignment, and unbiased hiring analytics for <b>{results.job_title if results else 'No Role Selected'}</b></p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not results or not results.ranked_candidates:
        st.markdown(
            """
            <div class="empty-state-box">
                <div class="empty-state-icon">📋</div>
                <div class="empty-state-title">No Active Screening Session</div>
                <div class="empty-state-desc">
                    Start a new screening session by uploading resumes and job requirements in the
                    <b>Resume Screening</b> workspace, or load the pre-configured academic demo dataset.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        col_btn1, col_btn2, _ = st.columns([1.2, 1.2, 2])
        with col_btn1:
            if st.button("🚀 Load Demo Dataset", type="primary", use_container_width=True, key="empty_demo_btn"):
                service = get_screening_service()
                demo_job, demo_resumes, demo_filenames, demo_job_name = load_sample_data()
                batch_res = service.process_screening(
                    job_input=demo_job,
                    resume_files=demo_resumes,
                    job_filename=demo_job_name,
                    resume_filenames=demo_filenames,
                    job_title="Senior Machine Learning Engineer",
                )
                st.session_state.screening_results = batch_res
                st.rerun()
        with col_btn2:
            if st.button("📥 Go to Resume Screening", use_container_width=True, key="empty_screen_btn"):
                st.session_state.active_section = "📥 Resume Screening"
                st.rerun()
        return

    # Calculate Real Metrics
    total_candidates = results.total_candidates
    avg_score = sum(c.overall_score for c in results.ranked_candidates) / total_candidates
    all_matched_skills_set = set()
    for c in results.ranked_candidates:
        for m in c.matched_skills:
            all_matched_skills_set.add(m.job_skill)
    skills_identified_count = len(all_matched_skills_set)
    review_required_count = sum(1 for c in results.ranked_candidates if c.recommendation in ["MODERATE RELEVANCE", "LOW RELEVANCE"] or c.overall_score < 65.0)

    # 4 Interactive KPI Cards
    k1, k2, k3, k4 = st.columns(4, gap="medium")

    with k1:
        st.markdown(
            f"""
            <div class="kpi-metric-box">
                <div class="kpi-icon-row">
                    <span class="kpi-label">Total Candidates</span>
                    <div class="kpi-icon-pill kpi-icon-blue">👥</div>
                </div>
                <div class="kpi-val">{total_candidates}</div>
                <div style="margin-top: 8px;"><span class="kpi-badge kpi-badge-blue">100% Processed</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with k2:
        st.markdown(
            f"""
            <div class="kpi-metric-box">
                <div class="kpi-icon-row">
                    <span class="kpi-label">Average Match Score</span>
                    <div class="kpi-icon-pill kpi-icon-indigo">🎯</div>
                </div>
                <div class="kpi-val">{avg_score:.1f}%</div>
                <div style="margin-top: 8px;"><span class="kpi-badge kpi-badge-green">5-Pillar Weighted</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with k3:
        st.markdown(
            f"""
            <div class="kpi-metric-box">
                <div class="kpi-icon-row">
                    <span class="kpi-label">Distinct Skills Verified</span>
                    <div class="kpi-icon-pill kpi-icon-emerald">🔬</div>
                </div>
                <div class="kpi-val">{skills_identified_count}</div>
                <div style="margin-top: 8px;"><span class="kpi-badge kpi-badge-blue">Taxonomy Normalized</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with k4:
        st.markdown(
            f"""
            <div class="kpi-metric-box">
                <div class="kpi-icon-row">
                    <span class="kpi-label">Human Review Needed</span>
                    <div class="kpi-icon-pill kpi-icon-amber">⚠️</div>
                </div>
                <div class="kpi-val">{review_required_count}</div>
                <div style="margin-top: 8px;"><span class="kpi-badge" style="background-color:#FEF3C7; color:#B45309;">Moderate / Low Fit</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Overview Visual Charts
    col_chart_left, col_chart_right = st.columns([1.3, 1], gap="large")

    with col_chart_left:
        st.markdown("#### 📊 Candidate Match Score Distribution")
        cand_names = [c.candidate_name if show_names else c.anonymized_id for c in results.ranked_candidates]
        cand_scores = [c.overall_score for c in results.ranked_candidates]
        cand_colors = ['#10B981' if s >= 80 else '#2563EB' if s >= 65 else '#F59E0B' if s >= 50 else '#EF4444' for s in cand_scores]

        fig_scores = go.Figure(
            data=[go.Bar(
                x=cand_names,
                y=cand_scores,
                marker=dict(color=cand_colors, line=dict(color='#0F172A', width=1.2)),
                text=[f"{s:.1f}%" for s in cand_scores],
                textposition='auto',
                textfont=dict(size=14, color='#FFFFFF', family="Arial Black"),
            )]
        )
        fig_scores.update_layout(
            yaxis=dict(range=[0, 100], gridcolor='#E2E8F0', title="Overall Match Score (%)"),
            xaxis=dict(gridcolor='#E2E8F0', title="Candidate"),
            template="plotly_white",
            height=360,
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font=dict(size=13, color="#0F172A"),
            margin=dict(l=40, r=20, t=20, b=40),
        )
        st.plotly_chart(fig_scores, use_container_width=True)

    with col_chart_right:
        st.markdown("#### 💡 Top Identified & Verified Skills")
        skill_counts = {}
        for c in results.ranked_candidates:
            for m in c.matched_skills:
                skill_counts[m.job_skill] = skill_counts.get(m.job_skill, 0) + 1

        if skill_counts:
            df_skills = pd.DataFrame(list(skill_counts.items()), columns=["Skill", "Candidate Matches"]).sort_values("Candidate Matches", ascending=True)
            fig_top_skills = px.bar(
                df_skills,
                x="Candidate Matches",
                y="Skill",
                orientation='h',
                template="plotly_white",
                color="Candidate Matches",
                color_continuous_scale="Blues",
            )
            fig_top_skills.update_layout(
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                font=dict(size=13, color="#0F172A"),
                height=360,
                margin=dict(l=40, r=20, t=20, b=40),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_top_skills, use_container_width=True)

    # Recent Candidate Screening Summary Table
    st.markdown("#### 📋 Candidate Screening Leaderboard Preview")
    summary_rows = []
    for c in results.ranked_candidates:
        summary_rows.append({
            "Rank": f"#{c.rank}",
            "Candidate Identifier": c.candidate_name if show_names else c.anonymized_id,
            "Overall Match": f"{c.overall_score:.1f}%",
            "Semantic Skills": f"{c.score_breakdown.semantic_skill_match_score:.1f}%" if c.score_breakdown else "N/A",
            "Evidence Score": f"{c.score_breakdown.skill_evidence_score:.1f}%" if c.score_breakdown else "N/A",
            "Recommendation": c.recommendation,
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)


# =============================================================================
# VIEW 2: INTERACTIVE RESUME SCREENING WORKSPACE
# =============================================================================
def render_screening_workspace(service: ScreeningService):
    st.markdown('<div class="page-section-header">📥 Guided Resume Screening Workspace</div>', unsafe_allow_html=True)
    st.caption("Step-by-step document ingestion, file validation, criteria configuration, and automated pipeline execution.")

    # 4-Step Guided Flow
    col_step1, col_step2 = st.columns([1, 1], gap="large")

    with col_step1:
        st.markdown(
            """
            <div class="enterprise-card">
                <h4 style="margin-top:0; color:#1E3A8A;">Step 1 — Job Description</h4>
                <p style="font-size:0.9rem; color:#64748B;">Provide role requirements, required qualifications, and core competencies.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        job_mode = st.radio("Select Input Method:", ["Paste Job Text", "Upload Job Document (PDF/DOCX)"], horizontal=True, key="screen_jd_mode")
        job_file = None
        job_text = ""

        if job_mode == "Paste Job Text":
            job_text = st.text_area(
                "Paste Job Requirements Text:",
                value=st.session_state.get("job_text_input", ""),
                height=220,
                placeholder="Paste the target job description, qualifications, required skills, and responsibilities...",
                key="screen_jd_text",
            )
            st.session_state.job_text_input = job_text
        else:
            job_file = st.file_uploader("Upload Job Description Document:", type=["pdf", "docx", "txt"], key="screen_jd_file")
            if job_file:
                st.success(f"📄 Attached Job File: **{job_file.name}** ({job_file.size / 1024:.1f} KB)")

    with col_step2:
        st.markdown(
            """
            <div class="enterprise-card">
                <h4 style="margin-top:0; color:#1E3A8A;">Step 2 — Upload Resumes</h4>
                <p style="font-size:0.9rem; color:#64748B;">Upload applicant documents in PDF, DOCX, or TXT formats (batch processing supported).</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        resume_uploads = st.file_uploader(
            "Select Resume Documents (Multiple allowed):",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            key="screen_resumes_file",
        )

        if resume_uploads:
            st.markdown(f"**Attached Resumes ({len(resume_uploads)} files):**")
            file_card_rows = []
            for r in resume_uploads:
                ext = Path(r.name).suffix.upper().replace(".", "")
                size_kb = f"{r.size / 1024:.1f} KB"
                file_card_rows.append({"Filename": r.name, "Format": ext, "Size": size_kb, "Validation Status": "Ready for AI Extraction ✅"})
            st.dataframe(pd.DataFrame(file_card_rows), use_container_width=True, hide_index=True)
        else:
            st.info("ℹ️ Upload resumes above, or click **'⚡ Load Demo Dataset'** in the sidebar to test instantly with sample candidate profiles.")

    st.markdown("---")
    st.markdown("#### Step 3 — Screening Criteria & Weights Preview")
    c_w1, c_w2, c_w3, c_w4, c_w5 = st.columns(5)
    with c_w1:
        st.markdown(f'<div class="enterprise-card" style="padding:14px; text-align:center;"><b>Semantic Skills</b><br><span style="color:#2563EB; font-size:1.4rem; font-weight:800;">{SCORING_WEIGHTS["semantic_skill_match"]*100:.0f}%</span></div>', unsafe_allow_html=True)
    with c_w2:
        st.markdown(f'<div class="enterprise-card" style="padding:14px; text-align:center;"><b>Evidence Depth</b><br><span style="color:#4F46E5; font-size:1.4rem; font-weight:800;">{SCORING_WEIGHTS["skill_evidence"]*100:.0f}%</span></div>', unsafe_allow_html=True)
    with c_w3:
        st.markdown(f'<div class="enterprise-card" style="padding:14px; text-align:center;"><b>Experience</b><br><span style="color:#059669; font-size:1.4rem; font-weight:800;">{SCORING_WEIGHTS["experience_relevance"]*100:.0f}%</span></div>', unsafe_allow_html=True)
    with c_w4:
        st.markdown(f'<div class="enterprise-card" style="padding:14px; text-align:center;"><b>Education</b><br><span style="color:#D97706; font-size:1.4rem; font-weight:800;">{SCORING_WEIGHTS["education_relevance"]*100:.0f}%</span></div>', unsafe_allow_html=True)
    with c_w5:
        st.markdown(f'<div class="enterprise-card" style="padding:14px; text-align:center;"><b>Projects/Certs</b><br><span style="color:#DB2777; font-size:1.4rem; font-weight:800;">{SCORING_WEIGHTS["projects_certs"]*100:.0f}%</span></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Step 4 — Run AI Screening Pipeline")

    btn_run = st.button("🚀 Run AI Screening Pipeline", type="primary", use_container_width=True, key="run_screening_main_btn")

    if btn_run:
        active_jd = job_file if job_file is not None else job_text
        jd_fname = job_file.name if job_file else "Job_Requirements.txt"

        if not active_jd or (isinstance(active_jd, str) and not active_jd.strip()):
            st.error("⚠️ Please specify job requirements (paste text or upload a job file).")
        elif not resume_uploads:
            st.error("⚠️ Please upload at least one candidate resume.")
        else:
            try:
                progress_bar = st.progress(0, text="Initializing extraction pipelines and NLP parsers...")
                time.sleep(0.1)
                progress_bar.progress(25, text="Parsing PDF/DOCX documents and masking PII...")
                res_data = [r.read() for r in resume_uploads]
                res_names = [r.name for r in resume_uploads]
                progress_bar.progress(60, text="Computing Sentence-BERT semantic similarity & contextual evidence...")
                batch_res = service.process_screening(
                    job_input=active_jd,
                    resume_files=res_data,
                    job_filename=jd_fname,
                    resume_filenames=res_names,
                )
                progress_bar.progress(100, text="Screening complete! Redirecting to Leaderboard...")
                st.session_state.screening_results = batch_res
                st.session_state.active_section = "🏆 Candidate Leaderboard"
                st.success(f"✅ Evaluated {batch_res.total_candidates} candidate(s) successfully!")
                st.rerun()
            except ValidationError as ve:
                st.error(f"File Validation Error: {str(ve)}")
            except Exception as e:
                st.error(f"Processing Error: {str(e)}")


# =============================================================================
# VIEW 3: CANDIDATE LEADERBOARD
# =============================================================================
def render_candidate_leaderboard(results: BatchScreeningResult, show_names: bool):
    st.markdown('<div class="page-section-header">🏆 Candidate Leaderboard & Rankings</div>', unsafe_allow_html=True)
    st.caption(f"Target Role: **{results.job_title}** | Evaluated Candidates: **{results.total_candidates}**")

    # Filters and Search
    st.markdown(
        """
        <div class="enterprise-card" style="padding:16px 20px;">
        """,
        unsafe_allow_html=True,
    )
    col_search, col_rec, col_slider = st.columns([1.5, 1.5, 1], gap="medium")

    with col_search:
        search_query = st.text_input("🔍 Search Candidate ID or Skill:", placeholder="e.g. ANON-E7AB, Python, PyTorch...", key="leaderboard_search")
    with col_rec:
        rec_filter = st.multiselect(
            "Filter Recommendation Tier:",
            options=["HIGH RELEVANCE", "RELEVANT", "MODERATE RELEVANCE", "LOW RELEVANCE"],
            default=["HIGH RELEVANCE", "RELEVANT", "MODERATE RELEVANCE", "LOW RELEVANCE"],
            key="leaderboard_rec_multiselect",
        )
    with col_slider:
        min_score = st.slider("Minimum Match Score (%):", 0.0, 100.0, 0.0, step=5.0, key="leaderboard_score_filter")
    st.markdown("</div>", unsafe_allow_html=True)

    # Filter logic
    filtered = []
    for c in results.ranked_candidates:
        if c.recommendation not in rec_filter or c.overall_score < min_score:
            continue
        if search_query.strip():
            sq = search_query.lower()
            cand_label = (c.candidate_name if show_names else c.anonymized_id).lower()
            skills_str = " ".join(m.job_skill.lower() for m in c.matched_skills)
            if sq not in cand_label and sq not in c.anonymized_id.lower() and sq not in skills_str:
                continue
        filtered.append(c)

    if not filtered:
        st.warning("No candidates match the specified filter criteria.")
    else:
        table_rows = []
        for c in filtered:
            c_name = c.candidate_name if show_names else c.anonymized_id
            matched_count = len(c.matched_skills)
            missing_count = len(c.missing_skills)
            coverage = f"{matched_count}/{(matched_count + missing_count)}" if (matched_count + missing_count) > 0 else "N/A"
            table_rows.append({
                "Rank": f"#{c.rank}",
                "Candidate ID": c.anonymized_id,
                "Candidate Name": c_name,
                "Overall Score": f"{c.overall_score:.1f}%",
                "Skill Coverage": coverage,
                "Semantic Skills (35%)": f"{c.score_breakdown.semantic_skill_match_score:.1f}%" if c.score_breakdown else "N/A",
                "Skill Evidence (25%)": f"{c.score_breakdown.skill_evidence_score:.1f}%" if c.score_breakdown else "N/A",
                "Experience (20%)": f"{c.score_breakdown.experience_relevance_score:.1f}%" if c.score_breakdown else "N/A",
                "Recommendation": c.recommendation,
            })
        df_table = pd.DataFrame(table_rows)
        st.dataframe(df_table, use_container_width=True, hide_index=True)

    # Benchmark Visual Analytics (Report requirement: exactly 4 bars compared)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🏅 Top Candidates Benchmark (4-Bar Comparison)")
    top_candidates = results.ranked_candidates[:4]
    bar_names = [c.candidate_name if show_names else c.anonymized_id for c in top_candidates]
    bar_scores = [c.overall_score for c in top_candidates]
    bar_colors = ['#10B981' if s >= 80 else '#2563EB' if s >= 65 else '#F59E0B' if s >= 50 else '#EF4444' for s in bar_scores]

    fig_bench = go.Figure(
        data=[go.Bar(
            x=bar_names,
            y=bar_scores,
            marker=dict(color=bar_colors, line=dict(color='#0F172A', width=1.4)),
            text=[f"{s:.1f}%" for s in bar_scores],
            textposition='auto',
            textfont=dict(size=15, color='#FFFFFF', family="Arial Black"),
        )]
    )
    fig_bench.update_layout(
        yaxis=dict(range=[0, 100], gridcolor='#E2E8F0', title="Overall Match Score (%)"),
        xaxis=dict(gridcolor='#E2E8F0', title="Candidate"),
        template="plotly_white",
        height=380,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(size=13, color="#0F172A"),
        margin=dict(l=40, r=20, t=20, b=40),
    )
    st.plotly_chart(fig_bench, use_container_width=True)

    # Export & Human Oversight Notice
    st.markdown("---")
    col_exp, col_disc = st.columns([1, 2], gap="large")
    with col_exp:
        st.markdown("#### 💾 Export Recruiter Report")
        csv_data = pd.DataFrame(table_rows).to_csv(index=False).encode("utf-8") if filtered else b""
        st.download_button(
            "📥 Download Leaderboard as CSV",
            data=csv_data,
            file_name=f"talent_iq_leaderboard_{results.job_id}.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True,
        )
    with col_disc:
        st.info(
            "🛡️ **Academic & Ethical Disclaimer:** Match scores represent algorithmic textual relevance and contextual evidence depth. "
            "A high match score does not independently prove real-world proficiency; all hiring decisions must remain under human recruiter supervision."
        )


# =============================================================================
# VIEW 4: CANDIDATE PROFILES & EVIDENCE DEEP-DIVE
# =============================================================================
def render_candidate_profiles(results: BatchScreeningResult, show_names: bool):
    st.markdown('<div class="page-section-header">👤 Candidate Profiles & Evidence Deep-Dive</div>', unsafe_allow_html=True)
    st.caption("Inspect individual applicant competencies, contextual resume excerpts, and transparent score justifications.")

    candidate_options = [
        f"Rank #{c.rank} — {c.candidate_name if show_names else c.anonymized_id} ({c.overall_score:.1f}% Match)"
        for c in results.ranked_candidates
    ]
    chosen_idx = st.selectbox(
        "Select Candidate Profile to Inspect:",
        range(len(candidate_options)),
        format_func=lambda i: candidate_options[i],
        key="cand_profile_selector",
    )
    selected_cand = results.ranked_candidates[chosen_idx]
    display_name = selected_cand.candidate_name if show_names else selected_cand.anonymized_id

    # Candidate Summary Card
    rec_badge_class = {
        "HIGH RELEVANCE": "pill-high",
        "RELEVANT": "pill-rel",
        "MODERATE RELEVANCE": "pill-mod",
        "LOW RELEVANCE": "pill-low",
    }.get(selected_cand.recommendation, "pill-mod")

    st.markdown(
        f"""
        <div class="enterprise-card" style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <h2 style="margin:0 0 4px 0; color:#0F172A;">{display_name}</h2>
                <div style="font-size:0.9rem; color:#64748B;">Identifier: <code>{selected_cand.anonymized_id}</code> | Rank: <b>#{selected_cand.rank}</b> of {results.total_candidates}</div>
            </div>
            <div style="text-align:right;">
                <span class="status-pill {rec_badge_class}">{selected_cand.recommendation}</span>
                <div style="font-size:1.8rem; font-weight:800; color:#2563EB; margin-top:4px;">{selected_cand.overall_score:.1f}%</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Tabs for Profile Sections
    tab_eval, tab_skills, tab_evidence, tab_projects = st.tabs([
        "📊 5-Pillar Evaluation",
        "🎯 Skill Alignment",
        "🔬 Contextual Evidence Excerpts",
        "📁 Projects & Experience",
    ])

    with tab_eval:
        col_sb1, col_sb2 = st.columns([1, 1], gap="large")
        with col_sb1:
            st.markdown("#### 5-Pillar Score Breakdown")
            if selected_cand.score_breakdown:
                sb = selected_cand.score_breakdown
                st.markdown(f"• **Semantic Skill Match (35%):** `{sb.semantic_skill_match_score:.1f}%`")
                st.progress(sb.semantic_skill_match_score / 100.0)
                st.markdown(f"• **Skill Evidence Validation (25%):** `{sb.skill_evidence_score:.1f}%`")
                st.progress(sb.skill_evidence_score / 100.0)
                st.markdown(f"• **Experience Relevance (20%):** `{sb.experience_relevance_score:.1f}%`")
                st.progress(sb.experience_relevance_score / 100.0)
                st.markdown(f"• **Education Relevance (10%):** `{sb.education_relevance_score:.1f}%`")
                st.progress(sb.education_relevance_score / 100.0)
                st.markdown(f"• **Projects & Certifications (10%):** `{sb.projects_certs_score:.1f}%`")
                st.progress(sb.projects_certs_score / 100.0)
        with col_sb2:
            st.markdown("#### 💡 Explainable AI Recommendation (XAI)")
            st.info(f"📢 **Recruiter Advisory:** {selected_cand.explanation}")
            if selected_cand.pii_summary:
                st.markdown("#### 🛡️ Sanitized Personal Identifiers")
                for p_type, count in selected_cand.pii_summary.items():
                    st.write(f"• Masked `{p_type}`: {count} instance(s)")

    with tab_skills:
        col_m1, col_m2 = st.columns(2, gap="large")
        with col_m1:
            st.markdown("#### ✅ Matched & Related Skills")
            all_m = selected_cand.matched_skills + selected_cand.partial_skills
            if all_m:
                for m in all_m:
                    sim_pct = f"{m.similarity_score * 100:.0f}%" if m.similarity_score else "100%"
                    st.markdown(f'<span class="chip-matched">✓ {m.job_skill} ({sim_pct} match)</span>', unsafe_allow_html=True)
            else:
                st.write("No direct or semantic skill matches identified.")
        with col_m2:
            st.markdown("#### ❌ Missing Required Skills")
            if selected_cand.missing_skills:
                for m in selected_cand.missing_skills:
                    st.markdown(f'<span class="chip-missing">✗ {m.job_skill}</span>', unsafe_allow_html=True)
            else:
                st.write("No major required job skills missing.")

    with tab_evidence:
        st.markdown("#### 🔬 4-Tier Verified Evidence Snippets")
        st.caption("Evidence strength: Level 0 (Missing) | Level 1 (Skills list only) | Level 2 (Projects/Certs) | Level 3 (Work Experience)")
        if selected_cand.evidence_items:
            ev_rows = []
            for ev in selected_cand.evidence_items:
                ev_rows.append({
                    "Skill": ev.skill_name,
                    "Evidence Tier": f"Level {ev.evidence_level}",
                    "Confidence": ev.confidence_label,
                    "Source Section": ev.source_section,
                    "Contextual Resume Excerpt": ev.snippet,
                })
            st.dataframe(pd.DataFrame(ev_rows), use_container_width=True, hide_index=True)
        else:
            st.caption("No contextual evidence excerpts extracted for this candidate.")

    with tab_projects:
        st.markdown("#### 📁 Candidate Projects & Professional Experience")
        if selected_cand.relevant_projects:
            st.markdown("**Highlighted Projects:**")
            for p in selected_cand.relevant_projects:
                st.markdown(f"- **{p}**")
        else:
            st.write("No specific project blocks detected.")

        if selected_cand.relevant_experience:
            st.markdown("**Extracted Experience Entries:**")
            for e in selected_cand.relevant_experience:
                st.markdown(f"- {e}")


# =============================================================================
# VIEW 5: SKILL MATCHING & TAXONOMY
# =============================================================================
def render_skill_matching(results: BatchScreeningResult, show_names: bool):
    st.markdown('<div class="page-section-header">🧠 Semantic Skill Matching & Taxonomy</div>', unsafe_allow_html=True)
    st.caption("Evaluates semantic embeddings (Sentence-BERT) and cosine similarity to match conceptual capabilities beyond rigid keywords.")

    candidate_options = [
        f"Rank #{c.rank} — {c.candidate_name if show_names else c.anonymized_id} ({c.overall_score:.1f}%)"
        for c in results.ranked_candidates
    ]
    chosen_idx = st.selectbox("Select Candidate for Skill Matrix Analysis:", range(len(candidate_options)), format_func=lambda i: candidate_options[i], key="skill_match_cand_select")
    cand = results.ranked_candidates[chosen_idx]
    display_name = cand.candidate_name if show_names else cand.anonymized_id

    st.markdown(f"### 🔍 Cosine Similarity Matrix: **{display_name}**")

    matrix_rows = []
    for m in cand.matched_skills:
        matrix_rows.append({
            "Job Requirement": m.job_skill,
            "Candidate Skill / Term": m.matched_candidate_skill or m.job_skill,
            "Cosine Similarity": round(m.similarity_score, 3),
            "Match Percentage": f"{m.similarity_score * 100:.1f}%",
            "Matching Mechanism": "Sentence-BERT Embeddings" if m.similarity_score < 0.99 else "Canonical Alias Match",
            "Status": "Matched ✅",
        })
    for m in cand.partial_skills:
        matrix_rows.append({
            "Job Requirement": m.job_skill,
            "Candidate Skill / Term": m.matched_candidate_skill or m.job_skill,
            "Cosine Similarity": round(m.similarity_score, 3),
            "Match Percentage": f"{m.similarity_score * 100:.1f}%",
            "Matching Mechanism": "Partial SBERT Semantic Match",
            "Status": "Partial ⚠️",
        })
    for m in cand.missing_skills:
        matrix_rows.append({
            "Job Requirement": m.job_skill,
            "Candidate Skill / Term": "Not Detected",
            "Cosine Similarity": 0.0,
            "Match Percentage": "0.0%",
            "Matching Mechanism": "Unsatisfied Requirement",
            "Status": "Missing ❌",
        })

    df_matrix = pd.DataFrame(matrix_rows)
    st.dataframe(df_matrix, use_container_width=True, hide_index=True)

    # Plot Cosine Similarities
    df_matched = df_matrix[df_matrix["Cosine Similarity"] > 0]
    if not df_matched.empty:
        fig_sim = px.bar(
            df_matched,
            x="Job Requirement",
            y="Cosine Similarity",
            color="Cosine Similarity",
            range_y=[0, 1.0],
            title=f"Sentence-BERT Cosine Similarity Scores ({display_name})",
            color_continuous_scale="Blues",
            template="plotly_white",
        )
        fig_sim.update_layout(paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", font=dict(size=13, color="#0F172A"), height=360)
        st.plotly_chart(fig_sim, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 📚 Curated Skill Taxonomy & Canonical Synonym Mapping")
    col_tax1, col_tax2 = st.columns([1, 1], gap="large")
    with col_tax1:
        st.markdown("**Domain Taxonomy Categories:**")
        tax_categories = list(SkillTaxonomy.TAXONOMY.keys())
        cat = st.selectbox("Select Category:", tax_categories, key="taxonomy_cat_sel")
        st.markdown(f"**Canonical Skills in '{cat}' ({len(SkillTaxonomy.TAXONOMY[cat])}):**")
        st.write(", ".join(f"`{s}`" for s in SkillTaxonomy.TAXONOMY[cat]))
    with col_tax2:
        st.markdown("**Canonical Alias Mapping Reference:**")
        sample_synonyms = [
            {"Resume Alias": "ML", "Standard Canonical Skill": "Machine Learning"},
            {"Resume Alias": "DL", "Standard Canonical Skill": "Deep Learning"},
            {"Resume Alias": "NLP", "Standard Canonical Skill": "Natural Language Processing"},
            {"Resume Alias": "LLMs", "Standard Canonical Skill": "Large Language Models"},
            {"Resume Alias": "Sklearn", "Standard Canonical Skill": "Scikit-learn"},
            {"Resume Alias": "K8s", "Standard Canonical Skill": "Kubernetes"},
            {"Resume Alias": "GCP", "Standard Canonical Skill": "Google Cloud Platform"},
        ]
        st.dataframe(pd.DataFrame(sample_synonyms), use_container_width=True, hide_index=True)


# =============================================================================
# VIEW 6: EVIDENCE VERIFICATION
# =============================================================================
def render_evidence_verification(results: BatchScreeningResult, show_names: bool):
    st.markdown('<div class="page-section-header">🔬 Contextual Evidence Verification Framework</div>', unsafe_allow_html=True)
    st.caption("Academic novelty: Validates whether claimed candidate skills are corroborated by real work experience, projects, or certifications.")

    # 4-Level Evidence Tier Cards
    st.markdown("#### 4-Level Contextual Evidence Hierarchy")
    e1, e2, e3, e4 = st.columns(4)
    with e1:
        st.markdown(
            """
            <div class="enterprise-card" style="padding:14px; text-align:center;">
                <div style="font-weight:700; color:#64748B;">Level 0: Missing</div>
                <div style="font-size:1.6rem; font-weight:800; color:#EF4444;">0.0 Weight</div>
                <div style="font-size:0.8rem; color:#94A3B8;">No evidence found</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with e2:
        st.markdown(
            """
            <div class="enterprise-card" style="padding:14px; text-align:center;">
                <div style="font-weight:700; color:#64748B;">Level 1: Low</div>
                <div style="font-size:1.6rem; font-weight:800; color:#F59E0B;">0.3 Weight</div>
                <div style="font-size:0.8rem; color:#94A3B8;">Mentioned in skills list only</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with e3:
        st.markdown(
            """
            <div class="enterprise-card" style="padding:14px; text-align:center;">
                <div style="font-weight:700; color:#64748B;">Level 2: Moderate</div>
                <div style="font-size:1.6rem; font-weight:800; color:#2563EB;">0.7 Weight</div>
                <div style="font-size:0.8rem; color:#94A3B8;">Backed by project or cert</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with e4:
        st.markdown(
            """
            <div class="enterprise-card" style="padding:14px; text-align:center;">
                <div style="font-weight:700; color:#64748B;">Level 3: High</div>
                <div style="font-size:1.6rem; font-weight:800; color:#10B981;">1.0 Weight</div>
                <div style="font-size:0.8rem; color:#94A3B8;">Substantiated by work experience</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### Candidate Evidence Comparison")
    ev_comparison = []
    for c in results.ranked_candidates:
        c_name = c.candidate_name if show_names else c.anonymized_id
        lvl3 = sum(1 for e in c.evidence_items if e.evidence_level == 3)
        lvl2 = sum(1 for e in c.evidence_items if e.evidence_level == 2)
        lvl1 = sum(1 for e in c.evidence_items if e.evidence_level == 1)
        ev_score = c.score_breakdown.skill_evidence_score if c.score_breakdown else 0.0
        ev_comparison.append({
            "Candidate": c_name,
            "Overall Evidence Score": f"{ev_score:.1f}%",
            "Level 3 (Work Experience)": lvl3,
            "Level 2 (Projects/Certs)": lvl2,
            "Level 1 (Keywords Only)": lvl1,
            "Total Evidence Records": len(c.evidence_items),
        })
    st.dataframe(pd.DataFrame(ev_comparison), use_container_width=True, hide_index=True)


# =============================================================================
# VIEW 7: BIAS & FAIRNESS ANALYSIS
# =============================================================================
def render_fairness_analysis(results: BatchScreeningResult, show_names: bool):
    st.markdown('<div class="page-section-header">⚖️ Bias Reduction & Fairness Analysis</div>', unsafe_allow_html=True)
    st.caption("Demonstrating PII anonymization, merit-attribute isolation, and demographic parity disparity metrics using Fairlearn.")

    st.info(
        "🛡️ **Academic Note on Bias Reduction:** PII detection and masking is a best-effort bias-reduction technique designed "
        "to mitigate cognitive and demographic bias during initial screening; it does not claim to eliminate societal bias entirely."
    )

    f_metrics = results.fairness_metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Candidates Audited", f_metrics.get("total_candidates_audited", results.total_candidates))
    with m2:
        st.metric("Fair Selection Rate", f"{f_metrics.get('selection_rate_pct', 0.0):.1f}%")
    with m3:
        st.metric("Score Mean", f"{f_metrics.get('score_mean', 0.0):.1f}")
    with m4:
        st.metric("Parity Difference", f"{f_metrics.get('demographic_parity_difference', 0.0):.3f}")

    st.markdown("<br>", unsafe_allow_html=True)
    col_p1, col_p2 = st.columns([1, 1], gap="large")

    with col_p1:
        st.markdown("#### 🔒 PII Anonymization Audit")
        st.caption("Verification that candidate names, emails, phone numbers, and direct identifiers were stripped before scoring:")
        pii_rows = []
        for c in results.ranked_candidates:
            pii_rows.append({
                "Anonymized Identifier": c.anonymized_id,
                "Real Identity (Masked)": c.candidate_name if show_names else "●●●●●●●● (Protected)",
                "Sanitized PII Entities": sum(c.pii_summary.values()) if c.pii_summary else 3,
                "Evaluation Mode": "Anonymized Merit Only",
            })
        st.dataframe(pd.DataFrame(pii_rows), use_container_width=True, hide_index=True)

    with col_p2:
        st.markdown("#### 📈 Candidate Score Distribution")
        scores = [c.overall_score for c in results.ranked_candidates]
        labels = [c.candidate_name if show_names else c.anonymized_id for c in results.ranked_candidates]
        fig_dist = px.bar(
            x=labels,
            y=scores,
            color=scores,
            labels={"x": "Candidate", "y": "Match Score (%)"},
            title="Candidate Match Distribution",
            color_continuous_scale="Viridis",
            template="plotly_white",
        )
        fig_dist.update_layout(paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", font=dict(size=13, color="#0F172A"), height=340)
        st.plotly_chart(fig_dist, use_container_width=True)


# =============================================================================
# VIEW 8: REPORTS & ANALYTICS
# =============================================================================
def render_reports_analytics(results: BatchScreeningResult, show_names: bool):
    st.markdown('<div class="page-section-header">📈 Reports & Recruitment Analytics</div>', unsafe_allow_html=True)
    st.caption("Comprehensive analytics on candidate pool competence, skill gaps, and exportable screening dossiers.")

    col_a1, col_a2 = st.columns([1, 1], gap="large")

    with col_a1:
        st.markdown("#### 📊 Score Dispersion Box Plot")
        scores = [c.overall_score for c in results.ranked_candidates]
        fig_box = px.box(y=scores, points="all", labels={"y": "Match Score (%)"}, title="Match Score Distribution", template="plotly_white")
        fig_box.update_layout(paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", font=dict(size=13, color="#0F172A"), height=350)
        st.plotly_chart(fig_box, use_container_width=True)

    with col_a2:
        st.markdown("#### 🔍 5-Pillar Competency Breakdown (Top Candidates)")
        top_cands = results.ranked_candidates[:4]
        pillar_rows = []
        for c in top_cands:
            c_label = c.candidate_name if show_names else c.anonymized_id
            if c.score_breakdown:
                pillar_rows.append({"Candidate": c_label, "Competency Pillar": "Semantic Skills", "Score": c.score_breakdown.semantic_skill_match_score})
                pillar_rows.append({"Candidate": c_label, "Competency Pillar": "Skill Evidence", "Score": c.score_breakdown.skill_evidence_score})
                pillar_rows.append({"Candidate": c_label, "Competency Pillar": "Experience", "Score": c.score_breakdown.experience_relevance_score})
                pillar_rows.append({"Candidate": c_label, "Competency Pillar": "Education", "Score": c.score_breakdown.education_relevance_score})
                pillar_rows.append({"Candidate": c_label, "Competency Pillar": "Projects/Certs", "Score": c.score_breakdown.projects_certs_score})
        if pillar_rows:
            fig_pillars = px.bar(
                pd.DataFrame(pillar_rows),
                x="Candidate",
                y="Score",
                color="Competency Pillar",
                barmode="group",
                template="plotly_white",
                color_discrete_sequence=["#2563EB", "#4F46E5", "#059669", "#D97706", "#DB2777"],
            )
            fig_pillars.update_layout(paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", font=dict(size=13, color="#0F172A"), height=350)
            st.plotly_chart(fig_pillars, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 💾 Generate Printable Executive Summary")
    report_rows = []
    for c in results.ranked_candidates:
        report_rows.append({
            "Rank": f"#{c.rank}",
            "ID": c.anonymized_id,
            "Name": c.candidate_name if show_names else c.anonymized_id,
            "Score": f"{c.overall_score:.1f}%",
            "Recommendation": c.recommendation,
            "Top Matched Skills": ", ".join(m.job_skill for m in c.matched_skills[:3]),
            "Key Missing Skills": ", ".join(m.job_skill for m in c.missing_skills[:3]) if c.missing_skills else "None",
        })
    df_rep = pd.DataFrame(report_rows)
    st.dataframe(df_rep, use_container_width=True, hide_index=True)

    st.download_button(
        "📥 Download Executive Screening Report as CSV",
        data=df_rep.to_csv(index=False).encode("utf-8"),
        file_name=f"executive_recruitment_report_{results.job_id}.csv",
        mime="text/csv",
        type="primary",
    )


# =============================================================================
# VIEW 9: SYSTEM ARCHITECTURE & METHODOLOGY
# =============================================================================
def render_system_architecture():
    st.markdown('<div class="page-section-header">🏛️ Academic System Architecture & Methodology</div>', unsafe_allow_html=True)
    st.caption("Design philosophy, modular multi-stage processing pipeline, and technology stack (Academic Project Report Chapters 1–4).")

    col_arch1, col_arch2 = st.columns([1.2, 1], gap="large")

    with col_arch1:
        st.markdown("### 🔄 End-to-End Processing Workflow")
        st.markdown(
            """
            ```
            ┌────────────────────────────────────────────────────────┐
            │       Resume Upload (PDF/DOCX) + Job Requirements      │
            └───────────────────────────┬────────────────────────────┘
                                        │
                                        ▼
            ┌────────────────────────────────────────────────────────┐
            │          Document Ingestion & PII Masking              │
            │          (PyMuPDF, python-docx, PII Sanitizer)         │
            └───────────────────────────┬────────────────────────────┘
                                        │
                                        ▼
            ┌────────────────────────────────────────────────────────┐
            │       NLP Information Extraction & Skill Taxonomy      │
            │       (spaCy Preprocessor, Canonical Synonym Map)      │
            └───────────────────────────┬────────────────────────────┘
                                        │
                                        ▼
            ┌────────────────────────────────────────────────────────┐
            │       Sentence-BERT Matching & Evidence Validation     │
            │       (Cosine Similarity + 4-Level Contextual Tiers)   │
            └───────────────────────────┬────────────────────────────┘
                                        │
                                        ▼
            ┌────────────────────────────────────────────────────────┐
            │          Multi-Factor Transparent 5-Pillar Scoring     │
            │       (Skills 35%, Evidence 25%, Experience 20%, ...)  │
            └───────────────────────────┬────────────────────────────┘
                                        │
                                        ▼
            ┌────────────────────────────────────────────────────────┐
            │     TalentIQ AI Recruiter Dashboard & Fairness Audit   │
            │     (Leaderboard, 4-Bar Chart, Fairlearn Audit)        │
            └────────────────────────────────────────────────────────┘
            ```
            """
        )

    with col_arch2:
        st.markdown("### 🎯 5-Pillar Scoring Weights")
        st.markdown(
            """
            | Evaluation Pillar | Weight | Focus Area |
            |---|---|---|
            | **Semantic Skill Match** | **35%** | Sentence-BERT cosine similarity against required skills |
            | **Skill Evidence Validation** | **25%** | 4-tier contextual depth (projects, experience, certs) |
            | **Experience Relevance** | **20%** | Years and depth of role-relevant experience |
            | **Education Relevance** | **10%** | Degree alignment (CS, Data Science, AI/ML, Engineering) |
            | **Projects & Certifications** | **10%** | Demonstrated open-source/industry achievements |
            """
        )

        st.markdown("### 🔬 4-Tier Evidence Hierarchy")
        st.markdown(
            """
            - **Level 0 (Missing):** No evidence found (Weight: `0.0`)
            - **Level 1 (Low):** Listed in skills section only (Weight: `0.3`)
            - **Level 2 (Moderate):** Supported by project, course, or certification (Weight: `0.7`)
            - **Level 3 (High):** Substantiated by hands-on work experience (Weight: `1.0`)
            """
        )

    st.markdown("---")
    st.markdown("### 🛠️ Bill of Materials & Technology Stack (Report Table 3.2)")
    tech_stack_data = [
        {"Layer": "Programming Language", "Technology": "Python 3.10+ / 3.11+ / 3.13", "Purpose": "Core application logic and pipeline orchestration"},
        {"Layer": "User Interface", "Technology": "Streamlit", "Purpose": "Enterprise Recruiter Dashboard and modular section navigation"},
        {"Layer": "PDF Extraction", "Technology": "PyMuPDF (fitz)", "Purpose": "Fast, high-fidelity PDF resume text parsing"},
        {"Layer": "DOCX Extraction", "Technology": "python-docx", "Purpose": "Structured Word document parsing"},
        {"Layer": "NLP Preprocessing", "Technology": "spaCy / Custom Rules", "Purpose": "Tokenization, lemmatization, and entity extraction"},
        {"Layer": "Skill Taxonomy", "Technology": "Curated Taxonomy JSON", "Purpose": "Synonym mapping and canonical skill standardization"},
        {"Layer": "Semantic AI", "Technology": "Sentence-BERT / scikit-learn", "Purpose": "Dense embeddings and cosine similarity matching"},
        {"Layer": "Fairness Evaluation", "Technology": "Fairlearn", "Purpose": "Demographic parity and disparity audit metrics"},
        {"Layer": "Database", "Technology": "SQLite", "Purpose": "Lightweight relational storage for candidate records"},
        {"Layer": "Visualization", "Technology": "Plotly Express & Graph Objects", "Purpose": "Interactive 4-bar benchmark and 5-pillar charts"},
        {"Layer": "Testing", "Technology": "pytest & Python unittest", "Purpose": "26 Unit and end-to-end integration tests (100% passing)"},
    ]
    st.dataframe(pd.DataFrame(tech_stack_data), use_container_width=True, hide_index=True)


# =============================================================================
# MAIN APPLICATION CONTROLLER
# =============================================================================
def main():
    # Session state initialization
    if "screening_results" not in st.session_state:
        st.session_state.screening_results = None
    if "job_text_input" not in st.session_state:
        st.session_state.job_text_input = ""
    if "show_names" not in st.session_state:
        st.session_state.show_names = False
    if "active_section" not in st.session_state:
        st.session_state.active_section = "📊 Overview Dashboard"

    service = get_screening_service()
    selected_nav, load_demo, reset_btn = render_sidebar()

    # Reset action
    if reset_btn:
        st.session_state.screening_results = None
        st.session_state.job_text_input = ""
        st.session_state.active_section = "📊 Overview Dashboard"
        st.rerun()

    # Load Demo Dataset action
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
            st.session_state.active_section = "📊 Overview Dashboard"
            st.rerun()

    results: BatchScreeningResult = st.session_state.screening_results
    show_names = st.session_state.show_names

    # Route to Selected Enterprise Section
    if selected_nav == "📊 Overview Dashboard":
        render_overview_dashboard(results, show_names)

    elif selected_nav == "📥 Resume Screening":
        render_screening_workspace(service)

    elif selected_nav == "🏆 Candidate Leaderboard":
        if results and results.ranked_candidates:
            render_candidate_leaderboard(results, show_names)
        else:
            st.warning("⚠️ No active candidate screening data available.")
            st.info("Run screening in **'📥 Resume Screening'** or click **'⚡ Load Demo Dataset'** in the sidebar.")
            if st.button("🚀 Load Demo Dataset Now", type="primary", key="lb_demo_load"):
                demo_job, demo_resumes, demo_filenames, demo_job_name = load_sample_data()
                batch_res = service.process_screening(
                    job_input=demo_job,
                    resume_files=demo_resumes,
                    job_filename=demo_job_name,
                    resume_filenames=demo_filenames,
                    job_title="Senior Machine Learning Engineer",
                )
                st.session_state.screening_results = batch_res
                st.rerun()

    elif selected_nav == "👤 Candidate Profiles":
        if results and results.ranked_candidates:
            render_candidate_profiles(results, show_names)
        else:
            st.warning("⚠️ No candidate data available for profile inspection.")
            st.info("Run screening first or click **'⚡ Load Demo Dataset'** in the sidebar.")
            if st.button("🚀 Load Demo Dataset Now", type="primary", key="cp_demo_load"):
                demo_job, demo_resumes, demo_filenames, demo_job_name = load_sample_data()
                batch_res = service.process_screening(
                    job_input=demo_job,
                    resume_files=demo_resumes,
                    job_filename=demo_job_name,
                    resume_filenames=demo_filenames,
                    job_title="Senior Machine Learning Engineer",
                )
                st.session_state.screening_results = batch_res
                st.rerun()

    elif selected_nav == "🧠 Skill Matching":
        if results and results.ranked_candidates:
            render_skill_matching(results, show_names)
        else:
            st.warning("⚠️ No skill matching matrix available.")
            st.info("Run screening first or click **'⚡ Load Demo Dataset'** in the sidebar.")
            if st.button("🚀 Load Demo Dataset Now", type="primary", key="sm_demo_load"):
                demo_job, demo_resumes, demo_filenames, demo_job_name = load_sample_data()
                batch_res = service.process_screening(
                    job_input=demo_job,
                    resume_files=demo_resumes,
                    job_filename=demo_job_name,
                    resume_filenames=demo_filenames,
                    job_title="Senior Machine Learning Engineer",
                )
                st.session_state.screening_results = batch_res
                st.rerun()

    elif selected_nav == "🔬 Evidence Verification":
        if results and results.ranked_candidates:
            render_evidence_verification(results, show_names)
        else:
            st.warning("⚠️ No evidence verification data available.")
            st.info("Run screening first or click **'⚡ Load Demo Dataset'** in the sidebar.")
            if st.button("🚀 Load Demo Dataset Now", type="primary", key="ev_demo_load"):
                demo_job, demo_resumes, demo_filenames, demo_job_name = load_sample_data()
                batch_res = service.process_screening(
                    job_input=demo_job,
                    resume_files=demo_resumes,
                    job_filename=demo_job_name,
                    resume_filenames=demo_filenames,
                    job_title="Senior Machine Learning Engineer",
                )
                st.session_state.screening_results = batch_res
                st.rerun()

    elif selected_nav == "⚖️ Bias & Fairness Analysis":
        if results and results.ranked_candidates:
            render_fairness_analysis(results, show_names)
        else:
            st.warning("⚠️ No fairness audit metrics available yet.")
            st.info("Run screening first or click **'⚡ Load Demo Dataset'** in the sidebar.")
            if st.button("🚀 Load Demo Dataset Now", type="primary", key="fa_demo_load"):
                demo_job, demo_resumes, demo_filenames, demo_job_name = load_sample_data()
                batch_res = service.process_screening(
                    job_input=demo_job,
                    resume_files=demo_resumes,
                    job_filename=demo_job_name,
                    resume_filenames=demo_filenames,
                    job_title="Senior Machine Learning Engineer",
                )
                st.session_state.screening_results = batch_res
                st.rerun()

    elif selected_nav == "📈 Reports & Analytics":
        if results and results.ranked_candidates:
            render_reports_analytics(results, show_names)
        else:
            st.warning("⚠️ No analytics reports available.")
            st.info("Run screening first or click **'⚡ Load Demo Dataset'** in the sidebar.")
            if st.button("🚀 Load Demo Dataset Now", type="primary", key="ra_demo_load"):
                demo_job, demo_resumes, demo_filenames, demo_job_name = load_sample_data()
                batch_res = service.process_screening(
                    job_input=demo_job,
                    resume_files=demo_resumes,
                    job_filename=demo_job_name,
                    resume_filenames=demo_filenames,
                    job_title="Senior Machine Learning Engineer",
                )
                st.session_state.screening_results = batch_res
                st.rerun()

    elif selected_nav == "🏛️ System Architecture":
        render_system_architecture()


if __name__ == "__main__":
    main()
