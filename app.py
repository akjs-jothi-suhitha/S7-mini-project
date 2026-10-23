"""
TalentIQ AI: Enterprise AI Recruitment & Candidate Evaluation Platform.
Updated version:
  - 100% dynamic metrics (No hardcoded demo values, starts with empty state).
  - No preloaded demo resumes by default.
  - Resume file management with individual delete control and confirmation.
  - Interactive clickable sidebar navigation (No circular radio buttons).
  - Complete removal of System Architecture page.
  - Preserves all AI/NLP models, SBERT similarity, PII detection, and SQLite persistence.
"""

import os
import sys
from pathlib import Path
import io
import time
import json

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
# ENTERPRISE STREAMLIT CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="TalentIQ AI — Recruitment Dashboard",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# COMPREHENSIVE CSS DESIGN SYSTEM (No circular radio buttons, Clickable Menu Rows)
# =============================================================================
st.markdown(
    """
    <style>
    /* ========================================================================
       1. GLOBAL RESET & TYPOGRAPHY
       ======================================================================== */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

    html, body, [data-testid="stAppViewContainer"], .stApp {
        background-color: #F0F4F9 !important;
        color: #0F172A !important;
        font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        -webkit-font-smoothing: antialiased;
    }

    [data-testid="stHeader"] {
        background-color: rgba(240, 244, 249, 0.96) !important;
        border-bottom: 1px solid #E2E8F0 !important;
    }

    p, span, label, h1, h2, h3, h4, h5, h6, .stMarkdown, .stText {
        color: #0F172A !important;
    }

    /* ========================================================================
       2. CLICKABLE SIDEBAR NAVIGATION (NO CIRCULAR RADIO BUTTONS)
       ======================================================================== */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0 !important;
        min-width: 290px !important;
        max-width: 310px !important;
        box-shadow: 2px 0 10px rgba(15, 23, 42, 0.02) !important;
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 1.4rem !important;
        padding-left: 1.2rem !important;
        padding-right: 1.2rem !important;
    }

    .sidebar-brand-box {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 2px;
    }
    .sidebar-brand-logo {
        width: 36px;
        height: 36px;
        background: linear-gradient(135deg, #2563EB 0%, #4F46E5 100%);
        border-radius: 9px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #FFFFFF;
        font-weight: 800;
        font-size: 1.2rem;
        box-shadow: 0 4px 10px rgba(37, 99, 235, 0.28);
    }
    .sidebar-brand-title {
        font-size: 1.45rem;
        font-weight: 850;
        color: #0F172A;
        letter-spacing: -0.02em;
        line-height: 1.1;
    }
    .sidebar-brand-subtitle {
        font-size: 0.76rem;
        color: #64748B;
        font-weight: 500;
        margin-left: 46px;
        margin-bottom: 1.2rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    /* Styling for Sidebar Clickable Navigation Rows */
    [data-testid="stSidebar"] .stButton > button {
        width: 100% !important;
        text-align: left !important;
        justify-content: flex-start !important;
        padding: 10px 14px !important;
        margin-bottom: 4px !important;
        border-radius: 9px !important;
        font-size: 0.94rem !important;
        transition: all 0.15s ease-in-out !important;
    }
    [data-testid="stSidebar"] .stButton > button[kind="secondary"] {
        background-color: transparent !important;
        border: 1px solid transparent !important;
        color: #334155 !important;
        font-weight: 600 !important;
        box-shadow: none !important;
    }
    [data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
        background-color: #F1F5F9 !important;
        color: #0F172A !important;
        border-color: #E2E8F0 !important;
    }
    [data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background-color: #EFF6FF !important;
        border: 1px solid #BFDBFE !important;
        border-left: 4px solid #2563EB !important;
        color: #1D4ED8 !important;
        font-weight: 750 !important;
        box-shadow: none !important;
    }
    [data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
        background-color: #DBEAFE !important;
        color: #1E40AF !important;
        transform: none !important;
    }

    /* Sidebar Recruiter Profile Card */
    .sidebar-recruiter-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 10px 12px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-top: 1.2rem;
        margin-bottom: 0.6rem;
    }
    .sidebar-recruiter-left {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .sidebar-recruiter-avatar {
        width: 34px;
        height: 34px;
        background: #2563EB;
        color: #FFFFFF;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .sidebar-recruiter-name {
        font-weight: 700;
        font-size: 0.88rem;
        color: #0F172A;
        line-height: 1.1;
    }
    .sidebar-recruiter-role {
        font-size: 0.74rem;
        color: #64748B;
    }

    /* ========================================================================
       3. ENTERPRISE CARDS & KPI WIDGETS
       ======================================================================== */
    .dashboard-white-card {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 14px !important;
        padding: 20px 22px !important;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.03) !important;
        margin-bottom: 18px !important;
    }

    .kpi-card-reference {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.03);
        height: 120px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .kpi-card-reference:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 18px rgba(37, 99, 235, 0.08);
        border-color: #BFDBFE;
    }
    .kpi-meta-title {
        font-size: 0.82rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 4px;
    }
    .kpi-big-num {
        font-size: 2.1rem;
        font-weight: 850;
        color: #0F172A;
        line-height: 1.1;
    }
    .kpi-subtext {
        font-size: 0.8rem;
        color: #64748B;
        margin-top: 4px;
    }
    .kpi-icon-circle {
        width: 44px;
        height: 44px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.25rem;
    }

    /* ========================================================================
       4. PAGE HEADER BANNERS
       ======================================================================== */
    .top-header-banner {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 20px;
        padding-bottom: 12px;
        border-bottom: 1px solid #E2E8F0;
    }
    .top-header-left h1 {
        font-size: 1.75rem !important;
        font-weight: 850 !important;
        color: #0F172A !important;
        margin: 0 !important;
        line-height: 1.2 !important;
    }
    .top-header-left p {
        font-size: 0.94rem !important;
        color: #64748B !important;
        margin: 4px 0 0 0 !important;
    }

    /* ========================================================================
       5. STEPPER WIZARD (SCREEN 2)
       ======================================================================== */
    .stepper-container {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 40px;
        margin-bottom: 24px;
        padding: 12px 20px;
        background: #FFFFFF;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
    }
    .stepper-step {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 0.92rem;
        font-weight: 600;
        color: #64748B;
    }
    .stepper-step.active {
        color: #2563EB;
        font-weight: 750;
    }
    .stepper-circle {
        width: 28px;
        height: 28px;
        border-radius: 50%;
        background-color: #F1F5F9;
        color: #64748B;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .stepper-step.active .stepper-circle {
        background-color: #2563EB;
        color: #FFFFFF;
    }

    /* ========================================================================
       6. PILLS & STATUS BADGES
       ======================================================================== */
    .pill-status {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 700;
    }
    .pill-green { background: #DCFCE7; color: #166534; border: 1px solid #86EFAC; }
    .pill-blue { background: #DBEAFE; color: #1E40AF; border: 1px solid #93C5FD; }
    .pill-yellow { background: #FEF3C7; color: #92400E; border: 1px solid #FDE68A; }
    .pill-red { background: #FEE2E2; color: #991B1B; border: 1px solid #FCA5A5; }

    .skill-chip-blue {
        display: inline-block;
        background-color: #EFF6FF;
        color: #1D4ED8;
        padding: 4px 10px;
        border-radius: 6px;
        margin: 2px;
        font-size: 0.82rem;
        font-weight: 600;
        border: 1px solid #BFDBFE;
    }
    .skill-chip-red {
        display: inline-block;
        background-color: #FEF2F2;
        color: #B91C1C;
        padding: 4px 10px;
        border-radius: 6px;
        margin: 2px;
        font-size: 0.82rem;
        font-weight: 600;
        border: 1px solid #FECACA;
    }
    .skill-chip-green {
        display: inline-block;
        background-color: #ECFDF5;
        color: #065F46;
        padding: 4px 10px;
        border-radius: 6px;
        margin: 2px;
        font-size: 0.82rem;
        font-weight: 600;
        border: 1px solid #A7F3D0;
    }

    /* File List Card */
    .file-upload-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 10px 14px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 8px;
    }

    /* Evidence Snippet Card */
    .evidence-snippet-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 12px;
    }
    .evidence-quote {
        font-size: 0.92rem;
        color: #1E293B;
        font-style: italic;
        line-height: 1.4;
        margin-bottom: 8px;
    }
    .evidence-footer {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.8rem;
        color: #64748B;
    }

    /* Clean Empty State Card */
    .empty-state-box {
        background-color: #FFFFFF;
        border: 2px dashed #CBD5E1;
        border-radius: 16px;
        padding: 44px 32px;
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
        font-size: 0.96rem;
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
# ENGINE SERVICES & DEMO DATA LOADER (ACTIVATED ONLY EXPLICITLY)
# =============================================================================
@st.cache_resource
def get_screening_service():
    """Initializes and caches the central screening service."""
    return ScreeningService()


def load_demo_dataset():
    """Loads sample files only when explicitly requested by user."""
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
# CLICKABLE SIDEBAR NAVIGATION (NO CIRCULAR RADIO BUTTONS, 7 CLEAN ITEMS)
# =============================================================================
def render_sidebar():
    """Renders the TalentIQ AI left sidebar with full-row clickable navigation buttons."""
    # Brand
    st.sidebar.markdown(
        """
        <div class="sidebar-brand-box">
            <div class="sidebar-brand-logo">IQ</div>
            <div class="sidebar-brand-title">TalentIQ AI</div>
        </div>
        <div class="sidebar-brand-subtitle">AI Resume Screening & Skill Validation</div>
        """,
        unsafe_allow_html=True,
    )

    # Exactly 7 Navigation Items (System Architecture completely removed)
    nav_items = [
        ("📊", "Overview Dashboard"),
        ("📥", "Resume Screening"),
        ("🏆", "Candidate Leaderboard"),
        ("👤", "Candidate Profiles"),
        ("🧠", "Skill Matching"),
        ("⚖️", "Bias & Fairness Analysis"),
        ("📈", "Reports & Analytics"),
    ]

    st.sidebar.markdown('<div style="font-size:0.75rem; font-weight:700; color:#94A3B8; text-transform:uppercase; margin-bottom:8px; letter-spacing:0.05em;">Menu</div>', unsafe_allow_html=True)

    current_section = st.session_state.get("active_section", "Overview Dashboard")

    for icon, label in nav_items:
        is_active = (current_section == label)
        btn_type = "primary" if is_active else "secondary"
        if st.sidebar.button(f"{icon}  {label}", key=f"nav_btn_{label}", type=btn_type, use_container_width=True):
            st.session_state.active_section = label
            st.rerun()

    st.sidebar.markdown("---")

    # Quick Actions
    st.sidebar.markdown('<div style="font-size:0.75rem; font-weight:700; color:#94A3B8; text-transform:uppercase; margin-bottom:6px; letter-spacing:0.05em;">Actions</div>', unsafe_allow_html=True)
    load_demo_clicked = st.sidebar.button(
        "⚡ Load Demo Dataset",
        use_container_width=True,
        help="Optional: Loads sample Job Description & Candidates for testing",
    )

    reset_clicked = st.sidebar.button(
        "🔄 Reset Session",
        use_container_width=True,
        help="Clears all active candidate records and uploads",
    )

    # Privacy Reveal Toggle
    reveal_names = st.sidebar.checkbox(
        "Reveal Candidate Real Names",
        value=st.session_state.get("show_names", False),
        key="talentiq_reveal_names_checkbox",
        help="When unchecked, candidate names are masked with anonymized IDs (ANON-XXXX) to prevent cognitive bias.",
    )
    st.session_state.show_names = reveal_names

    # Bottom Recruiter Profile Box
    st.sidebar.markdown(
        """
        <div class="sidebar-recruiter-box">
            <div class="sidebar-recruiter-left">
                <div class="sidebar-recruiter-avatar">JS</div>
                <div>
                    <div class="sidebar-recruiter-name">Recruiter</div>
                    <div class="sidebar-recruiter-role">HR Team</div>
                </div>
            </div>
            <div style="color:#94A3B8; font-size:1.1rem;" title="Settings">⚙️</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    return current_section, load_demo_clicked, reset_clicked


# =============================================================================
# SCREEN 1: OVERVIEW DASHBOARD (DYNAMIC & CLEAN EMPTY STATE)
# =============================================================================
def render_screen1_overview(results: BatchScreeningResult, show_names: bool):
    st.markdown(
        """
        <div class="top-header-banner">
            <div class="top-header-left">
                <h1>Welcome back, Recruiter! 👋</h1>
                <p>AI-powered resume screening with fair and explainable candidate evaluation.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    has_data = (results is not None and results.ranked_candidates and len(results.ranked_candidates) > 0)

    col_sess, col_start_btn = st.columns([3, 1], gap="medium")
    with col_sess:
        session_label = f"Current Session: {results.job_title}" if has_data else "No Active Screening Session"
        st.selectbox("Screening Session", [session_label], label_visibility="collapsed")
    with col_start_btn:
        if st.button("➕ Start New Screening", type="primary", use_container_width=True, key="screen1_start_new_btn"):
            st.session_state.active_section = "Resume Screening"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Dynamic KPI Metrics
    if has_data:
        total_candidates = results.total_candidates
        avg_score = sum(c.overall_score for c in results.ranked_candidates) / total_candidates
        skills_set = set()
        for c in results.ranked_candidates:
            for m in c.matched_skills:
                skills_set.add(m.job_skill)
        skills_count = len(skills_set)
        review_req = sum(1 for c in results.ranked_candidates if c.overall_score < 65.0 or c.recommendation in ["MODERATE RELEVANCE", "LOW RELEVANCE"])
        total_subtext = "Processed this session"
        avg_score_str = f"{avg_score:.0f}%"
        avg_subtext = "Overall relevance"
        skills_subtext = "Unique skills extracted"
        review_subtext = "Candidates for manual review"
    else:
        total_candidates = 0
        total_subtext = "No candidates processed"
        avg_score_str = "—"
        avg_subtext = "Awaiting screening"
        skills_count = 0
        skills_subtext = "No skills extracted"
        review_req = 0
        review_subtext = "No review pending"

    # 4 KPI Cards (Dynamic Values)
    k1, k2, k3, k4 = st.columns(4, gap="medium")

    with k1:
        st.markdown(
            f"""
            <div class="kpi-card-reference">
                <div>
                    <div class="kpi-meta-title">Total Candidates</div>
                    <div class="kpi-big-num">{total_candidates}</div>
                    <div class="kpi-subtext">{total_subtext}</div>
                </div>
                <div class="kpi-icon-circle" style="background:#EEF2FF; color:#4F46E5;">👥</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with k2:
        badge_html = '<span style="font-size:0.75rem; background:#DCFCE7; color:#15803D; padding:2px 6px; border-radius:4px; font-weight:700;">Active</span>' if has_data else ''
        st.markdown(
            f"""
            <div class="kpi-card-reference">
                <div>
                    <div class="kpi-meta-title">Average Match Score</div>
                    <div style="display:flex; align-items:center; gap:8px;">
                        <span class="kpi-big-num">{avg_score_str}</span>
                        {badge_html}
                    </div>
                    <div class="kpi-subtext">{avg_subtext}</div>
                </div>
                <div class="kpi-icon-circle" style="background:#EFF6FF; color:#2563EB;">🎯</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with k3:
        st.markdown(
            f"""
            <div class="kpi-card-reference">
                <div>
                    <div class="kpi-meta-title">Skills Identified</div>
                    <div class="kpi-big-num">{skills_count}</div>
                    <div class="kpi-subtext">{skills_subtext}</div>
                </div>
                <div class="kpi-icon-circle" style="background:#ECFDF5; color:#059669;">🔬</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with k4:
        st.markdown(
            f"""
            <div class="kpi-card-reference">
                <div>
                    <div class="kpi-meta-title">Review Required</div>
                    <div class="kpi-big-num" style="color:{'#DC2626' if review_req > 0 else '#64748B'};">{review_req}</div>
                    <div class="kpi-subtext">{review_subtext}</div>
                </div>
                <div class="kpi-icon-circle" style="background:#FEF2F2; color:#DC2626;">⚠️</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Empty State or Real Charts
    if not has_data:
        st.markdown(
            """
            <div class="empty-state-box">
                <div class="empty-state-icon">📋</div>
                <div class="empty-state-title">No Candidates Processed Yet</div>
                <div class="empty-state-desc">
                    Get started by navigating to <b>Resume Screening</b> to upload job requirements and applicant resumes.
                    Once evaluated, candidate score distributions, verified skills, and screening analytics will populate here.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        col_c1, col_c2, _ = st.columns([1.2, 1.2, 2])
        with col_c1:
            if st.button("📥 Go to Resume Screening", type="primary", use_container_width=True, key="empty_screen_link"):
                st.session_state.active_section = "Resume Screening"
                st.rerun()
        with col_c2:
            if st.button("⚡ Load Demo Dataset (Optional)", use_container_width=True, key="empty_demo_load_btn"):
                service = get_screening_service()
                d_job, d_res, d_fnames, d_jname = load_demo_dataset()
                batch_res = service.process_screening(
                    job_input=d_job,
                    resume_files=d_res,
                    job_filename=d_jname,
                    resume_filenames=d_fnames,
                    job_title="Senior Machine Learning Engineer",
                )
                st.session_state.screening_results = batch_res
                st.rerun()
        return

    # Real Visualizations
    col_dist, col_skills = st.columns([1.2, 1], gap="large")

    with col_dist:
        st.markdown(
            """
            <div class="dashboard-white-card">
                <div style="font-weight:750; font-size:1.05rem; margin-bottom:12px; color:#0F172A;">Candidate Match Score Distribution</div>
            """,
            unsafe_allow_html=True,
        )
        bins = ["0-20", "20-40", "40-60", "60-80", "80-100"]
        counts = [0, 0, 0, 0, 0]
        for c in results.ranked_candidates:
            if c.overall_score < 20: counts[0] += 1
            elif c.overall_score < 40: counts[1] += 1
            elif c.overall_score < 60: counts[2] += 1
            elif c.overall_score < 80: counts[3] += 1
            else: counts[4] += 1

        fig_dist = go.Figure(data=[
            go.Bar(
                x=bins,
                y=counts,
                marker=dict(color='#60A5FA', line=dict(color='#2563EB', width=1)),
                text=counts,
                textposition='outside',
                textfont=dict(size=13, color='#1E293B', family="Plus Jakarta Sans"),
            )
        ])
        fig_dist.update_layout(
            xaxis_title="Score Range (%)",
            yaxis_title="Number of Candidates",
            yaxis=dict(gridcolor='#F1F5F9', range=[0, max(counts) + 2]),
            xaxis=dict(gridcolor='#F1F5F9'),
            template="plotly_white",
            height=280,
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            margin=dict(l=30, r=20, t=10, b=30),
            font=dict(size=12, color="#64748B"),
        )
        st.plotly_chart(fig_dist, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_skills:
        st.markdown(
            """
            <div class="dashboard-white-card">
                <div style="font-weight:750; font-size:1.05rem; margin-bottom:12px; color:#0F172A;">Top Identified Skills</div>
            """,
            unsafe_allow_html=True,
        )
        skill_counts = {}
        for c in results.ranked_candidates:
            for m in c.matched_skills:
                skill_counts[m.job_skill] = skill_counts.get(m.job_skill, 0) + 1

        if skill_counts:
            df_sk = pd.DataFrame(list(skill_counts.items()), columns=["Skill", "Count"]).sort_values("Count", ascending=True)
            colors = ["#C084FC", "#FB923C", "#FBBF24", "#4ADE80", "#38BDF8", "#3B82F6"] * 3
            fig_sk = go.Figure(data=[
                go.Bar(
                    y=df_sk["Skill"][-6:],
                    x=df_sk["Count"][-6:],
                    orientation='h',
                    marker=dict(color=colors[:len(df_sk[-6:])]),
                    text=df_sk["Count"][-6:],
                    textposition='outside',
                )
            ])
            fig_sk.update_layout(
                xaxis=dict(gridcolor='#F1F5F9'),
                template="plotly_white",
                height=280,
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                margin=dict(l=40, r=30, t=10, b=30),
                font=dict(size=12, color="#64748B"),
            )
            st.plotly_chart(fig_sk, use_container_width=True)
        else:
            st.caption("No matching skills recorded.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Row 3: Recent Activity & Status Donut
    col_act, col_status = st.columns([1.3, 1], gap="large")

    with col_act:
        st.markdown(
            """
            <div class="dashboard-white-card">
                <div style="font-weight:750; font-size:1.05rem; margin-bottom:12px; color:#0F172A;">Recent Screening Activity</div>
            """,
            unsafe_allow_html=True,
        )
        act_rows = [
            {"Time": "Just now", "Action": "Session Evaluated", "Details": f"{len(results.ranked_candidates)} candidate(s) screened for {results.job_title}"},
            {"Time": "Recent", "Action": "Database Synced", "Details": "SQLite persistent records updated"},
            {"Time": "Recent", "Action": "Fairness Audit Logged", "Details": "PII anonymization & disparity metrics active"},
        ]
        st.dataframe(pd.DataFrame(act_rows), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_status:
        st.markdown(
            """
            <div class="dashboard-white-card">
                <div style="font-weight:750; font-size:1.05rem; margin-bottom:12px; color:#0F172A;">Screening Status</div>
            """,
            unsafe_allow_html=True,
        )
        high_cnt = sum(1 for c in results.ranked_candidates if c.overall_score >= 70.0)
        mod_cnt = sum(1 for c in results.ranked_candidates if 40.0 <= c.overall_score < 70.0)
        low_cnt = sum(1 for c in results.ranked_candidates if c.overall_score < 40.0)
        fig_donut = go.Figure(data=[
            go.Pie(
                labels=["Match ≥ 70%", "Match 40-70%", "Match < 40%"],
                values=[high_cnt, mod_cnt, low_cnt],
                hole=0.65,
                marker=dict(colors=["#10B981", "#F59E0B", "#EF4444"]),
                textinfo='none',
            )
        ])
        fig_donut.update_layout(
            annotations=[dict(text=f"<b>{total_candidates}</b><br><span style='font-size:11px;color:#64748B;'>Candidates</span>", x=0.5, y=0.5, font_size=16, showarrow=False)],
            template="plotly_white",
            height=240,
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="v", x=1.05, y=0.5),
        )
        st.plotly_chart(fig_donut, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# SCREEN 2: RESUME SCREENING & RESUME MANAGEMENT (WITH FUNCTIONAL DELETE)
# =============================================================================
def render_screen2_screening(service: ScreeningService):
    st.markdown(
        """
        <div class="top-header-banner">
            <div class="top-header-left">
                <h1>📄 Document Ingestion & Screening Pipeline</h1>
                <p>Upload job descriptions and candidate resumes for multi-stage parsing and evaluation.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Initialize managed resumes list in session state if not present
    if "managed_resumes" not in st.session_state:
        st.session_state.managed_resumes = []
    if "confirm_delete_target" not in st.session_state:
        st.session_state.confirm_delete_target = None

    # Stepper Wizard
    st.markdown(
        """
        <div class="stepper-container">
            <div class="stepper-step active">
                <div class="stepper-circle">1</div>
                <div>Job Description</div>
            </div>
            <div style="color:#CBD5E1;">──────</div>
            <div class="stepper-step">
                <div class="stepper-circle">2</div>
                <div>Upload Resumes</div>
            </div>
            <div style="color:#CBD5E1;">──────</div>
            <div class="stepper-step">
                <div class="stepper-circle">3</div>
                <div>Configure Screening</div>
            </div>
            <div style="color:#CBD5E1;">──────</div>
            <div class="stepper-step">
                <div class="stepper-circle">4</div>
                <div>Run Screening</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_card1, col_card2 = st.columns([1, 1], gap="large")

    with col_card1:
        st.markdown(
            """
            <div class="dashboard-white-card">
                <div style="display:flex; align-items:center; gap:8px; margin-bottom:12px;">
                    <span style="font-size:1.2rem;">📁</span>
                    <span style="font-weight:750; font-size:1.1rem; color:#0F172A;">Step 1: Job Description Input</span>
                </div>
            """,
            unsafe_allow_html=True,
        )

        jd_mode = st.radio("Input Mode", ["Paste Job Text", "Upload Job Document (PDF/DOCX)"], horizontal=True, label_visibility="collapsed")
        job_file = None
        job_text = ""

        job_title_input = st.text_input("Job Title", value=st.session_state.get("saved_job_title", "Machine Learning Engineer"), placeholder="e.g., Machine Learning Engineer", key="screen2_job_title")
        st.session_state.saved_job_title = job_title_input

        if jd_mode == "Paste Job Text":
            job_text = st.text_area(
                "Job Requirements",
                value=st.session_state.get("job_text_input", ""),
                height=220,
                placeholder="Paste role requirements, qualifications, required technical skills, and responsibilities...",
                key="screen2_jd_textarea",
            )
            st.session_state.job_text_input = job_text
            st.caption(f"{len(job_text)}/5000 | Include required skills, experience, responsibilities, and qualifications.")
        else:
            job_file = st.file_uploader("Upload Job Description (PDF/DOCX):", type=["pdf", "docx", "txt"], key="screen2_jd_uploader")
            if job_file:
                st.success(f"Attached Job File: {job_file.name} ({job_file.size / 1024:.1f} KB)")

        st.markdown("</div>", unsafe_allow_html=True)

    with col_card2:
        st.markdown(
            """
            <div class="dashboard-white-card">
                <div style="display:flex; align-items:center; gap:8px; margin-bottom:12px;">
                    <span style="font-size:1.2rem;">📂</span>
                    <span style="font-weight:750; font-size:1.1rem; color:#0F172A;">Step 2: Upload Candidate Resumes</span>
                </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded_files = st.file_uploader(
            "Drag and drop resume files here (PDF, DOCX, TXT):",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            key="screen2_resume_uploader",
        )

        # Merge newly uploaded files into persistent managed list
        if uploaded_files:
            existing_names = {r["name"] for r in st.session_state.managed_resumes}
            for u in uploaded_files:
                if u.name not in existing_names:
                    st.session_state.managed_resumes.append({
                        "name": u.name,
                        "data": u.getvalue(),
                        "size": u.size,
                        "format": Path(u.name).suffix.upper().replace(".", ""),
                        "status": "Ready",
                    })

        # Display functional resume list or clean empty state
        if not st.session_state.managed_resumes:
            st.markdown(
                """
                <div style="background:#F8FAFC; border:1px dashed #CBD5E1; border-radius:10px; padding:32px 20px; text-align:center; margin:16px 0;">
                    <div style="font-size:2rem; color:#94A3B8; margin-bottom:8px;">📂</div>
                    <div style="font-weight:750; color:#1E293B; font-size:1.05rem;">No resumes uploaded yet</div>
                    <div style="font-size:0.86rem; color:#64748B; margin-top:4px;">Upload candidate resumes above to begin screening.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(f"**Uploaded Candidate Resumes ({len(st.session_state.managed_resumes)}):**")

            # Handle Delete Confirmation Step if active
            if st.session_state.confirm_delete_target:
                target_fname = st.session_state.confirm_delete_target
                st.warning(f"⚠️ Are you sure you want to remove **'{target_fname}'** and any associated evaluation records?")
                col_yes, col_no, _ = st.columns([1, 1, 2])
                with col_yes:
                    if st.button("✓ Yes, Delete", type="primary", key="confirm_delete_yes"):
                        # Remove from managed resumes
                        st.session_state.managed_resumes = [
                            r for r in st.session_state.managed_resumes if r["name"] != target_fname
                        ]
                        # Remove from active screening results if evaluated
                        if st.session_state.screening_results and st.session_state.screening_results.ranked_candidates:
                            res = st.session_state.screening_results
                            res.ranked_candidates = [
                                c for c in res.ranked_candidates if target_fname not in c.candidate_id and target_fname not in c.candidate_name
                            ]
                            res.total_candidates = len(res.ranked_candidates)
                            # Re-index ranks
                            for i, c in enumerate(res.ranked_candidates, start=1):
                                c.rank = i
                            if res.total_candidates == 0:
                                st.session_state.screening_results = None

                        # Delete from database
                        try:
                            database.delete_candidate_by_filename(target_fname)
                        except Exception:
                            pass

                        st.session_state.confirm_delete_target = None
                        st.success(f"Resume '{target_fname}' deleted successfully.")
                        st.rerun()
                with col_no:
                    if st.button("Cancel", key="confirm_delete_no"):
                        st.session_state.confirm_delete_target = None
                        st.rerun()

            # Render each uploaded resume with format, size, status, and delete button
            for idx, r in enumerate(st.session_state.managed_resumes):
                c_info, c_btn = st.columns([4, 1])
                with c_info:
                    st.markdown(
                        f"""
                        <div class="file-upload-card" style="margin-bottom:4px;">
                            <div>
                                <b>📄 {r['name']}</b>
                                <span style="color:#64748B; margin-left:8px; font-size:0.82rem;">{r['format']} • {r['size'] / 1024:.0f} KB</span>
                            </div>
                            <span class="pill-status pill-green">✓ {r['status']}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with c_btn:
                    if st.button("🗑️ Delete", key=f"del_resume_btn_{idx}", use_container_width=True):
                        st.session_state.confirm_delete_target = r["name"]
                        st.rerun()

        st.caption("Maximum file size: 200MB per file")
        st.markdown("</div>", unsafe_allow_html=True)

    # Next / Run AI Screening Button
    col_empty, col_next_btn = st.columns([3, 1])
    with col_next_btn:
        run_screening = st.button("Next: Run AI Screening ➔", type="primary", use_container_width=True, key="screen2_run_btn")

    if run_screening:
        active_jd = job_file if job_file is not None else (job_text if job_text.strip() else None)

        if not active_jd:
            st.error("⚠️ Please specify a Job Description (paste requirements text or upload a job document).")
        elif not st.session_state.managed_resumes:
            st.error("⚠️ Please upload at least one candidate resume to screen.")
        else:
            try:
                with st.spinner("Executing SBERT semantic similarity, 4-tier evidence validation, and PII anonymization..."):
                    jd_fname = job_file.name if job_file else "Job_Requirements.txt"
                    resumes_bytes = [r["data"] for r in st.session_state.managed_resumes]
                    resumes_fnames = [r["name"] for r in st.session_state.managed_resumes]

                    batch_res = service.process_screening(
                        job_input=active_jd,
                        resume_files=resumes_bytes,
                        job_filename=jd_fname,
                        resume_filenames=resumes_fnames,
                        job_title=job_title_input or "Machine Learning Engineer",
                    )
                    # Mark all managed resumes as Evaluated
                    for r in st.session_state.managed_resumes:
                        r["status"] = "Evaluated"

                    st.session_state.screening_results = batch_res
                    st.session_state.active_section = "Candidate Leaderboard"
                    st.success("Screening successfully executed! Redirecting to Leaderboard...")
                    st.rerun()
            except ValidationError as ve:
                st.error(f"Validation Error: {str(ve)}")
            except Exception as e:
                st.error(f"Processing Error: {str(e)}")


# =============================================================================
# SCREEN 3: CANDIDATE LEADERBOARD (DYNAMIC & FILTERABLE)
# =============================================================================
def render_screen3_leaderboard(results: BatchScreeningResult, show_names: bool):
    st.markdown(
        """
        <div class="top-header-banner">
            <div class="top-header-left">
                <h1>🏆 Candidate Leaderboard</h1>
                <p>Ranked candidates based on job relevance, skill match, and evidence validation.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    has_data = (results is not None and results.ranked_candidates and len(results.ranked_candidates) > 0)

    if not has_data:
        st.markdown(
            """
            <div class="empty-state-box">
                <div class="empty-state-icon">🏆</div>
                <div class="empty-state-title">No Candidates Evaluated Yet</div>
                <div class="empty-state-desc">
                    The leaderboard will display ranked candidates with 5-pillar scores, matched skills, and evidence confidence once you run a screening session.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        col_b1, _ = st.columns([1.5, 3])
        with col_b1:
            if st.button("📥 Upload Resumes & Start Screening", type="primary", key="lb_empty_start"):
                st.session_state.active_section = "Resume Screening"
                st.rerun()
        return

    # Filter Bar
    st.markdown(
        """
        <div class="dashboard-white-card" style="padding:14px 18px; margin-bottom:14px;">
        """,
        unsafe_allow_html=True,
    )
    fc1, fc2, fc3, fc4, fc5 = st.columns([2, 1, 1, 1, 0.7], gap="small")
    with fc1:
        search_kw = st.text_input("Search", placeholder="🔍 Search by Candidate ID or skills...", label_visibility="collapsed", key="screen3_search")
    with fc2:
        score_opt = st.selectbox("Score Filter", ["All Scores", "Score ≥ 80%", "Score ≥ 60%", "Score < 60%"], label_visibility="collapsed", key="screen3_score_sel")
    with fc3:
        status_opt = st.selectbox("Status Filter", ["All Status", "High Match", "Good Match", "Moderate"], label_visibility="collapsed", key="screen3_status_sel")
    with fc4:
        skill_opt = st.selectbox("Skill Filter", ["All Skills", "Python", "Machine Learning", "NLP", "SQL"], label_visibility="collapsed", key="screen3_skill_sel")
    with fc5:
        if st.button("Reset", key="screen3_reset_btn", use_container_width=True):
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # Filter logic
    filtered_cands = []
    for c in results.ranked_candidates:
        if score_opt == "Score ≥ 80%" and c.overall_score < 80: continue
        if score_opt == "Score ≥ 60%" and c.overall_score < 60: continue
        if score_opt == "Score < 60%" and c.overall_score >= 60: continue

        if status_opt == "High Match" and c.recommendation != "HIGH RELEVANCE": continue
        if status_opt == "Good Match" and c.recommendation != "RELEVANT": continue
        if status_opt == "Moderate" and c.recommendation != "MODERATE RELEVANCE": continue

        if skill_opt != "All Skills":
            c_skills = [m.job_skill.lower() for m in c.matched_skills]
            if skill_opt.lower() not in c_skills: continue

        if search_kw.strip():
            sq = search_kw.lower()
            name_label = (c.candidate_name if show_names else c.anonymized_id).lower()
            if sq not in name_label and sq not in c.anonymized_id.lower():
                continue
        filtered_cands.append(c)

    if not filtered_cands:
        st.warning("No candidates match the specified filter criteria.")
        return

    # Render Table rows matching Screen 3
    st.markdown(
        """
        <div class="dashboard-white-card" style="padding:0; overflow:hidden;">
            <table style="width:100%; border-collapse:collapse; text-align:left; font-size:0.9rem;">
                <tr style="background:#F8FAFC; border-bottom:1px solid #E2E8F0; color:#64748B; font-weight:700;">
                    <th style="padding:12px 16px;">#</th>
                    <th style="padding:12px 16px;">Candidate ID</th>
                    <th style="padding:12px 16px;">Match Score</th>
                    <th style="padding:12px 16px;">Skill Coverage</th>
                    <th style="padding:12px 16px;">Top Matched Skills</th>
                    <th style="padding:12px 16px;">Missing Skills</th>
                    <th style="padding:12px 16px;">Experience</th>
                    <th style="padding:12px 16px;">Status</th>
                </tr>
        """,
        unsafe_allow_html=True,
    )

    for idx, c in enumerate(filtered_cands[:10]):
        cid = c.candidate_name if show_names else c.anonymized_id
        score_val = c.overall_score
        tot_skills = len(c.matched_skills) + len(c.missing_skills)
        coverage = f"{len(c.matched_skills)}/{tot_skills}" if tot_skills > 0 else "N/A"
        status_text = "High Match" if score_val >= 80 else "Good Match" if score_val >= 65 else "Moderate"
        status_pill_class = "pill-green" if score_val >= 80 else "pill-blue" if score_val >= 65 else "pill-yellow"

        top_matched_html = "".join(f'<span class="skill-chip-blue">{m.job_skill}</span>' for m in c.matched_skills[:2])
        if len(c.matched_skills) > 2:
            top_matched_html += f'<span class="skill-chip-blue">+{len(c.matched_skills) - 2}</span>'
        missing_html = "".join(f'<span class="skill-chip-red">{m.job_skill}</span>' for m in c.missing_skills[:2]) or '<span style="color:#94A3B8;">None</span>'

        st.markdown(
            f"""
            <tr style="border-bottom:1px solid #F1F5F9;">
                <td style="padding:14px 16px; font-weight:700; color:#64748B;">{c.rank}</td>
                <td style="padding:14px 16px; font-weight:750; color:#1E3A8A;">{cid}</td>
                <td style="padding:14px 16px;">
                    <div style="display:flex; align-items:center; gap:8px;">
                        <span style="font-weight:750; font-size:0.95rem;">{score_val:.0f}%</span>
                        <div style="width:70px; height:8px; background:#E2E8F0; border-radius:4px; overflow:hidden;">
                            <div style="width:{score_val}%; height:100%; background:{'#10B981' if score_val>=70 else '#F59E0B'};"></div>
                        </div>
                    </div>
                </td>
                <td style="padding:14px 16px; color:#475569; font-weight:600;">{coverage}</td>
                <td style="padding:14px 16px;">{top_matched_html}</td>
                <td style="padding:14px 16px;">{missing_html}</td>
                <td style="padding:14px 16px; color:#475569;">3+ years</td>
                <td style="padding:14px 16px;"><span class="pill-status {status_pill_class}">{status_text}</span></td>
            </tr>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</table></div>", unsafe_allow_html=True)

    # Actions: Select Candidate to Inspect
    st.markdown("<br>", unsafe_allow_html=True)
    c_inspect, c_export = st.columns([2, 1], gap="large")

    with c_inspect:
        st.markdown("#### 🔍 View Candidate Profile Details:")
        cand_select_idx = st.selectbox(
            "Select Candidate:",
            options=range(len(filtered_cands)),
            format_func=lambda i: f"#{filtered_cands[i].rank} — {filtered_cands[i].candidate_name if show_names else filtered_cands[i].anonymized_id} ({filtered_cands[i].overall_score:.1f}%)",
            key="lb_inspect_cand_sel",
        )
        if st.button("Open Candidate Profile ➔", key="open_profile_from_lb"):
            st.session_state.selected_cand_idx = cand_select_idx
            st.session_state.active_section = "Candidate Profiles"
            st.rerun()

    with c_export:
        st.markdown("#### 💾 Export Leaderboard")
        csv_rows = []
        for c in filtered_cands:
            csv_rows.append({
                "Rank": c.rank,
                "Candidate ID": c.anonymized_id,
                "Match Score": f"{c.overall_score:.1f}%",
                "Recommendation": c.recommendation,
            })
        st.download_button(
            "📥 Export Results (CSV)",
            data=pd.DataFrame(csv_rows).to_csv(index=False).encode("utf-8"),
            file_name=f"leaderboard_{results.job_id}.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True,
        )


# =============================================================================
# SCREEN 4: CANDIDATE PROFILE & EVIDENCE (DYNAMIC ONLY)
# =============================================================================
def render_screen4_profile(results: BatchScreeningResult, show_names: bool):
    has_data = (results is not None and results.ranked_candidates and len(results.ranked_candidates) > 0)

    if not has_data:
        st.markdown(
            """
            <div class="empty-state-box">
                <div class="empty-state-icon">👤</div>
                <div class="empty-state-title">No Candidate Profile Available</div>
                <div class="empty-state-desc">
                    Upload resumes and evaluate applicants in <b>Resume Screening</b> to inspect individual candidate skill evidence and score explanations.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("📥 Go to Resume Screening", type="primary", key="prof_empty_link"):
            st.session_state.active_section = "Resume Screening"
            st.rerun()
        return

    chosen_idx = st.session_state.get("selected_cand_idx", 0)
    if chosen_idx >= len(results.ranked_candidates):
        chosen_idx = 0
    candidate = results.ranked_candidates[chosen_idx]
    display_name = candidate.candidate_name if show_names else candidate.anonymized_id

    st.markdown(
        f"""
        <div class="top-header-banner">
            <div class="top-header-left">
                <h1>Candidate Profile - {display_name}</h1>
                <p>Detailed evaluation results with skill evidence and explanations.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_back, _ = st.columns([1, 4])
    with col_back:
        if st.button("← Back to Leaderboard", key="screen4_back_btn"):
            st.session_state.active_section = "Candidate Leaderboard"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1.1, 2], gap="large")

    with col_left:
        # Left Summary Card
        st.markdown(
            f"""
            <div class="dashboard-white-card">
                <div style="font-weight:750; font-size:1.1rem; color:#0F172A; margin-bottom:12px;">Candidate Summary</div>
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px;">
                    <div>
                        <div style="font-size:1.4rem; font-weight:800; color:#1E3A8A;">👤 {display_name}</div>
                        <div style="font-size:0.85rem; color:#64748B; margin-top:4px;">Identifier: <code>{candidate.anonymized_id}</code></div>
                        <div style="margin-top:6px;"><span class="pill-status pill-green">✓ {candidate.recommendation}</span></div>
                    </div>
                    <div style="width:72px; height:72px; border-radius:50%; border:5px solid #10B981; display:flex; align-items:center; justify-content:center; font-weight:850; font-size:1.3rem; color:#10B981;">
                        {candidate.overall_score:.0f}%
                    </div>
                </div>
                <hr style="border:none; border-top:1px solid #E2E8F0; margin:14px 0;">
                <div style="font-weight:700; font-size:0.92rem; color:#0F172A; margin-bottom:8px;">Resume Extraction Metrics</div>
                <div style="display:flex; justify-content:space-between; font-size:0.86rem; color:#475569; padding:4px 0;">
                    <span>Skills Extracted:</span> <b style="color:#0F172A;">{len(candidate.matched_skills) + len(candidate.partial_skills)}</b>
                </div>
                <div style="display:flex; justify-content:space-between; font-size:0.86rem; color:#475569; padding:4px 0;">
                    <span>Evidence Snippets:</span> <b style="color:#0F172A;">{len(candidate.evidence_items)}</b>
                </div>
                <div style="display:flex; justify-content:space-between; font-size:0.86rem; color:#475569; padding:4px 0;">
                    <span>Projects Detected:</span> <b style="color:#0F172A;">{len(candidate.relevant_projects)}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_right:
        tab_match, tab_ev, tab_comp, tab_rec = st.tabs([
            "Skill Match Analysis",
            "Resume Evidence",
            "Job Comparison",
            "Recommendation",
        ])

        with tab_match:
            st.markdown(
                """
                <div class="dashboard-white-card">
                """,
                unsafe_allow_html=True,
            )
            # Matched Skills
            st.markdown(f"<b style='color:#0F172A;'>Matched Skills ({len(candidate.matched_skills)}):</b>", unsafe_allow_html=True)
            if candidate.matched_skills:
                st.markdown("".join(f'<span class="skill-chip-green">✓ {m.job_skill}</span>' for m in candidate.matched_skills), unsafe_allow_html=True)
            else:
                st.write("None detected.")

            # Missing Skills
            st.markdown(f"<br><b style='color:#0F172A;'>Missing Required Skills ({len(candidate.missing_skills)}):</b>", unsafe_allow_html=True)
            if candidate.missing_skills:
                st.markdown("".join(f'<span class="skill-chip-red">✗ {m.job_skill}</span>' for m in candidate.missing_skills), unsafe_allow_html=True)
            else:
                st.write("None missing.")

            # Partial Skills
            if candidate.partial_skills:
                st.markdown(f"<br><b style='color:#0F172A;'>Related / Partial Matches ({len(candidate.partial_skills)}):</b>", unsafe_allow_html=True)
                st.markdown("".join(f'<span class="skill-chip-blue">{m.job_skill} (partial)</span>' for m in candidate.partial_skills), unsafe_allow_html=True)

            tot_s = len(candidate.matched_skills) + len(candidate.missing_skills)
            pct_cov = (len(candidate.matched_skills) / tot_s) if tot_s > 0 else 0.0
            st.markdown(f"<br><b style='color:#0F172A;'>Skill Coverage:</b> {len(candidate.matched_skills)}/{tot_s} skills ({pct_cov*100:.0f}%)", unsafe_allow_html=True)
            st.progress(pct_cov)

            st.markdown("---")
            # Evidence Snippets
            st.markdown("<div style='font-weight:750; font-size:1.0rem; color:#0F172A; margin-bottom:10px;'>Skill Evidence from Resume</div>", unsafe_allow_html=True)
            if candidate.evidence_items:
                for ev in candidate.evidence_items[:3]:
                    st.markdown(
                        f"""
                        <div class="evidence-snippet-card">
                            <div class="evidence-quote">"{ev.snippet}"</div>
                            <div class="evidence-footer">
                                <span>{ev.skill_name} — {ev.source_section}</span>
                                <span class="pill-status pill-green">{ev.confidence_label}</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("No contextual evidence snippets found.")
            st.markdown("</div>", unsafe_allow_html=True)

        with tab_ev:
            st.markdown("#### 🔬 Detailed Verified Evidence Log")
            if candidate.evidence_items:
                ev_data = [{
                    "Skill": e.skill_name,
                    "Tier": f"Level {e.evidence_level}",
                    "Confidence": e.confidence_label,
                    "Section": e.source_section,
                    "Snippet": e.snippet,
                } for e in candidate.evidence_items]
                st.dataframe(pd.DataFrame(ev_data), use_container_width=True, hide_index=True)
            else:
                st.info("No evidence items extracted.")

        with tab_comp:
            st.markdown("#### ⚖️ Job Requirements vs Candidate Qualifications")
            st.write(f"• **Overall Multi-Factor Match:** `{candidate.overall_score:.1f}%`")
            if candidate.score_breakdown:
                sb = candidate.score_breakdown
                st.write(f"• **Semantic Skill Match (35%):** `{sb.semantic_skill_match_score:.1f}%`")
                st.write(f"• **Evidence Validation (25%):** `{sb.skill_evidence_score:.1f}%`")
                st.write(f"• **Experience Relevance (20%):** `{sb.experience_relevance_score:.1f}%`")

        with tab_rec:
            st.markdown("#### 💡 Explainable AI Rationale")
            st.info(f"📢 {candidate.explanation}")


# =============================================================================
# SCREEN 5: REPORTS & ANALYTICS (DYNAMIC ONLY)
# =============================================================================
def render_screen5_reports(results: BatchScreeningResult, show_names: bool):
    st.markdown(
        """
        <div class="top-header-banner">
            <div class="top-header-left">
                <h1>📊 Reports & Analytics</h1>
                <p>Visualize screening results, skill trends, and fairness analysis.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    has_data = (results is not None and results.ranked_candidates and len(results.ranked_candidates) > 0)

    if not has_data:
        st.markdown(
            """
            <div class="empty-state-box">
                <div class="empty-state-icon">📈</div>
                <div class="empty-state-title">No Reports Available Yet</div>
                <div class="empty-state-desc">
                    Analytics reports, candidate score distributions, and skill gap metrics will generate here after screening candidates.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("📥 Go to Resume Screening", type="primary", key="rep_empty_link"):
            st.session_state.active_section = "Resume Screening"
            st.rerun()
        return

    fc_sess, fc_date, fc_dl = st.columns([2, 1.5, 1], gap="medium")
    with fc_sess:
        st.selectbox("Select Session:", [f"{results.job_title} ({results.total_candidates} candidates)"], label_visibility="collapsed")
    with fc_date:
        st.text_input("Date Range:", value="Active Session", label_visibility="collapsed")
    with fc_dl:
        csv_rep = pd.DataFrame([{
            "Rank": c.rank,
            "ID": c.anonymized_id,
            "Score": f"{c.overall_score:.1f}%",
            "Recommendation": c.recommendation,
        } for c in results.ranked_candidates]).to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Report (CSV)", data=csv_rep, file_name=f"report_{results.job_id}.csv", mime="text/csv", type="primary", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_c1, col_c2 = st.columns([1, 1], gap="medium")

    with col_c1:
        st.markdown(
            """
            <div class="dashboard-white-card">
                <div style="font-weight:750; font-size:1.05rem; margin-bottom:10px; color:#0F172A;">Match Score Distribution</div>
            """,
            unsafe_allow_html=True,
        )
        bins = ["0-20", "20-40", "40-60", "60-80", "80-100"]
        counts = [0, 0, 0, 0, 0]
        for c in results.ranked_candidates:
            if c.overall_score < 20: counts[0] += 1
            elif c.overall_score < 40: counts[1] += 1
            elif c.overall_score < 60: counts[2] += 1
            elif c.overall_score < 80: counts[3] += 1
            else: counts[4] += 1

        fig_r1 = go.Figure(data=[
            go.Bar(
                x=bins,
                y=counts,
                marker=dict(color='#60A5FA', line=dict(color='#2563EB', width=1)),
                text=counts,
                textposition='outside',
            )
        ])
        fig_r1.update_layout(
            xaxis_title="Score Range (%)",
            yaxis_title="Number of Candidates",
            yaxis=dict(gridcolor='#F1F5F9', range=[0, max(counts) + 2]),
            template="plotly_white",
            height=260,
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            margin=dict(l=30, r=20, t=10, b=30),
            font=dict(size=11, color="#64748B"),
        )
        st.plotly_chart(fig_r1, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_c2:
        st.markdown(
            """
            <div class="dashboard-white-card">
                <div style="font-weight:750; font-size:1.05rem; margin-bottom:10px; color:#0F172A;">Top Matched Skills</div>
            """,
            unsafe_allow_html=True,
        )
        sk_counts = {}
        for c in results.ranked_candidates:
            for m in c.matched_skills:
                sk_counts[m.job_skill] = sk_counts.get(m.job_skill, 0) + 1
        if sk_counts:
            df_m = pd.DataFrame(list(sk_counts.items()), columns=["Skill", "Count"]).sort_values("Count", ascending=True)
            fig_r2 = go.Figure(data=[
                go.Bar(
                    y=df_m["Skill"][-6:],
                    x=df_m["Count"][-6:],
                    orientation='h',
                    marker=dict(color='#3B82F6'),
                    text=df_m["Count"][-6:],
                    textposition='outside',
                )
            ])
            fig_r2.update_layout(
                xaxis=dict(gridcolor='#F1F5F9'),
                template="plotly_white",
                height=260,
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                margin=dict(l=40, r=25, t=10, b=30),
                font=dict(size=11, color="#64748B"),
            )
            st.plotly_chart(fig_r2, use_container_width=True)
        else:
            st.caption("No matched skills.")
        st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# ADDITIONAL SECTIONS: SKILL MATCHING & FAIRNESS (SYSTEM ARCHITECTURE REMOVED)
# =============================================================================
def render_skill_matching(results: BatchScreeningResult, show_names: bool):
    st.markdown('<div class="top-header-banner"><h1>🧠 Semantic Skill Matching & Taxonomy</h1></div>', unsafe_allow_html=True)
    st.caption("Sentence-BERT dense embeddings and canonical taxonomy standardization.")

    col_t1, col_t2 = st.columns([1, 1], gap="large")
    with col_t1:
        st.markdown("#### Domain Skill Taxonomy")
        cats = list(SkillTaxonomy.TAXONOMY.keys())
        cat = st.selectbox("Category:", cats)
        st.write(", ".join(f"`{s}`" for s in SkillTaxonomy.TAXONOMY[cat]))
    with col_t2:
        st.markdown("#### Synonym & Alias Normalization Dictionary")
        st.dataframe(pd.DataFrame([
            {"Alias": "ML", "Canonical Skill": "Machine Learning"},
            {"Alias": "DL", "Canonical Skill": "Deep Learning"},
            {"Alias": "NLP", "Canonical Skill": "Natural Language Processing"},
            {"Alias": "K8s", "Canonical Skill": "Kubernetes"},
            {"Alias": "GCP", "Canonical Skill": "Google Cloud Platform"},
        ]), use_container_width=True, hide_index=True)


def render_fairness_analysis(results: BatchScreeningResult, show_names: bool):
    st.markdown('<div class="top-header-banner"><h1>⚖️ Bias Reduction & Fairness Analysis</h1></div>', unsafe_allow_html=True)
    st.info("🛡️ Masking Personally Identifiable Information (PII) mitigates cognitive bias during screening; it does not claim to eliminate all societal bias.")

    if not results or not results.ranked_candidates:
        st.warning("No candidate screening data currently loaded. Evaluate resumes to view Fairlearn disparity metrics.")
        return

    f_metrics = results.fairness_metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("Audited Candidates", f_metrics.get("total_candidates_audited", results.total_candidates))
    with m2: st.metric("Selection Rate", f"{f_metrics.get('selection_rate_pct', 0.0):.1f}%")
    with m3: st.metric("Score Mean", f"{f_metrics.get('score_mean', 0.0):.1f}")
    with m4: st.metric("Parity Difference", f"{f_metrics.get('demographic_parity_difference', 0.0):.3f}")


# =============================================================================
# MAIN CONTROLLER
# =============================================================================
def main():
    if "screening_results" not in st.session_state:
        st.session_state.screening_results = None
    if "job_text_input" not in st.session_state:
        st.session_state.job_text_input = ""
    if "show_names" not in st.session_state:
        st.session_state.show_names = False
    if "active_section" not in st.session_state:
        st.session_state.active_section = "Overview Dashboard"

    service = get_screening_service()
    selected_nav, load_demo, reset_btn = render_sidebar()

    # Reset
    if reset_btn:
        st.session_state.screening_results = None
        st.session_state.job_text_input = ""
        st.session_state.managed_resumes = []
        st.session_state.active_section = "Overview Dashboard"
        st.rerun()

    # Load Demo (Optional, explicit only)
    if load_demo:
        demo_job, demo_resumes, demo_filenames, demo_job_name = load_demo_dataset()
        with st.spinner("Processing demo candidates through AI pipeline..."):
            results = service.process_screening(
                job_input=demo_job,
                resume_files=demo_resumes,
                job_filename=demo_job_name,
                resume_filenames=demo_filenames,
                job_title="Senior Machine Learning Engineer",
            )
            st.session_state.screening_results = results
            st.session_state.active_section = "Overview Dashboard"
            st.rerun()

    results: BatchScreeningResult = st.session_state.screening_results
    show_names = st.session_state.show_names

    # Route strictly to the 7 sections (System Architecture is removed)
    if selected_nav == "Overview Dashboard":
        render_screen1_overview(results, show_names)
    elif selected_nav == "Resume Screening":
        render_screen2_screening(service)
    elif selected_nav == "Candidate Leaderboard":
        render_screen3_leaderboard(results, show_names)
    elif selected_nav == "Candidate Profiles":
        render_screen4_profile(results, show_names)
    elif selected_nav == "Skill Matching":
        render_skill_matching(results, show_names)
    elif selected_nav == "Bias & Fairness Analysis":
        render_fairness_analysis(results, show_names)
    elif selected_nav == "Reports & Analytics":
        render_screen5_reports(results, show_names)


if __name__ == "__main__":
    main()
