"""
AI-Driven Resume Screening and Skill Validation System.
Interactive Recruiter Dashboard Application.
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

# Page Configuration - Completely collapse/hide sidebar
st.set_page_config(
    page_title="AI Resume Screening & Skill Validation",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom Styling (Dark & Light mode adaptive, glassmorphism, no jarring white boxes)
st.markdown(
    """
    <style>
    /* Completely hide Streamlit sidebar */
    [data-testid="stSidebar"], section[data-testid="stSidebar"] {
        display: none !important;
    }
    
    /* Hero Title & Subtitles */
    .hero-container {
        padding: 1.2rem 1.5rem;
        background: linear-gradient(135deg, rgba(30, 58, 138, 0.15) 0%, rgba(59, 130, 246, 0.08) 100%);
        border: 1px solid rgba(59, 130, 246, 0.25);
        border-radius: 12px;
        margin-bottom: 1.2rem;
    }
    .main-header {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38BDF8, #818CF8, #C084FC);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #94A3B8;
        margin-bottom: 0.5rem;
        font-weight: 400;
    }

    /* Metric Cards - Sleek translucent styling that fits dark & light themes */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border-radius: 10px;
        padding: 16px 20px;
        border: 1px solid rgba(148, 163, 184, 0.18);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.5);
    }
    .metric-val {
        font-size: 2.0rem;
        font-weight: 800;
        color: #38BDF8;
        line-height: 1.2;
    }
    .metric-label {
        font-size: 0.82rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
        font-weight: 600;
    }

    /* Badges */
    .badge-high {
        background-color: rgba(34, 197, 94, 0.2);
        color: #4ADE80;
        padding: 5px 12px;
        border-radius: 9999px;
        font-weight: 700;
        border: 1px solid rgba(34, 197, 94, 0.4);
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-rel {
        background-color: rgba(59, 130, 246, 0.2);
        color: #60A5FA;
        padding: 5px 12px;
        border-radius: 9999px;
        font-weight: 700;
        border: 1px solid rgba(59, 130, 246, 0.4);
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-mod {
        background-color: rgba(245, 158, 11, 0.2);
        color: #FBBF24;
        padding: 5px 12px;
        border-radius: 9999px;
        font-weight: 700;
        border: 1px solid rgba(245, 158, 11, 0.4);
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-low {
        background-color: rgba(239, 68, 68, 0.2);
        color: #F87171;
        padding: 5px 12px;
        border-radius: 9999px;
        font-weight: 700;
        border: 1px solid rgba(239, 68, 68, 0.4);
        font-size: 0.85rem;
        display: inline-block;
    }

    /* Skill tags */
    .skill-tag {
        display: inline-block;
        background: rgba(59, 130, 246, 0.15);
        color: #93C5FD;
        padding: 4px 10px;
        border-radius: 6px;
        margin: 3px;
        font-size: 0.84rem;
        font-weight: 500;
        border: 1px solid rgba(96, 165, 250, 0.3);
    }
    .missing-tag {
        display: inline-block;
        background: rgba(239, 68, 68, 0.15);
        color: #FCA5A5;
        padding: 4px 10px;
        border-radius: 6px;
        margin: 3px;
        font-size: 0.84rem;
        font-weight: 500;
        border: 1px solid rgba(248, 113, 113, 0.3);
    }

    /* Info card */
    .info-panel {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 1rem;
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


def render_top_bar():
    """Renders the top banner and action bar without any sidebar."""
    st.markdown(
        """
        <div class="hero-container">
            <div class="main-header">🎯 AI-Driven Resume Screening & Skill Validation System</div>
            <div class="sub-header">Fair, Explainable, Semantic Candidate Matching with Contextual Evidence Validation</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_btn1, col_btn2, col_info = st.columns([1.3, 1.3, 3.4])

    with col_btn1:
        load_demo = st.button("🚀 Load Demo Dataset", type="primary", use_container_width=True, help="Loads Senior ML Engineer job and 3 pre-configured resumes (Alex Turner, Priya Sharma, John Doe)")
    
    with col_btn2:
        reset_btn = st.button("🔄 Reset / Clear All", use_container_width=True, help="Clears current screening session results")

    with col_info:
        with st.expander("⚙️ System Architecture & 5-Pillar Scoring Weights", expanded=False):
            st.markdown(
                f"- **Semantic Skill Match:** `{SCORING_WEIGHTS['semantic_skill_match']*100:.0f}%` (Sentence-BERT & Cosine Similarity)\n"
                f"- **Skill Evidence Validation:** `{SCORING_WEIGHTS['skill_evidence']*100:.0f}%` (Contextual project & work verification)\n"
                f"- **Experience Relevance:** `{SCORING_WEIGHTS['experience_relevance']*100:.0f}%` (Tenure & domain alignment)\n"
                f"- **Education Relevance:** `{SCORING_WEIGHTS['education_relevance']*100:.0f}%` (Degree level & domain relevance)\n"
                f"- **Projects & Certifications:** `{SCORING_WEIGHTS['projects_certs']*100:.0f}%` (Portfolio depth & verified credentials)"
            )

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
    load_demo, reset_btn = render_top_bar()

    if reset_btn:
        st.session_state.screening_results = None
        st.session_state.job_text_input = ""
        st.rerun()

    # Handle demo loading
    if load_demo:
        demo_job, demo_resumes, demo_filenames, demo_job_name = load_sample_data()
        with st.spinner("Processing demo candidates through full screening pipeline..."):
            results = service.process_screening(
                job_input=demo_job,
                resume_files=demo_resumes,
                job_filename=demo_job_name,
                resume_filenames=demo_filenames,
                job_title="Senior Machine Learning Engineer",
            )
            st.session_state.screening_results = results
            st.success("Demo dataset processed successfully!")

    # Tab Layout
    tab_input, tab_results, tab_analytics, tab_fairness = st.tabs([
        "📥 1. Input & Screening",
        "🏆 2. Candidate Ranking & Profiles",
        "📊 3. Visual Analytics",
        "⚖️ 4. Fairness & Evaluation Audit",
    ])

    # TAB 1: INPUT & SCREENING
    with tab_input:
        st.subheader("Step 1: Provide Job Description")
        col_jd_text, col_jd_file = st.columns([3, 2])

        with col_jd_text:
            job_text = st.text_area(
                "Paste Job Description Text",
                value=st.session_state.job_text_input,
                height=210,
                placeholder="Paste the full job requirements, skills, experience, and responsibilities...",
            )

        with col_jd_file:
            job_file = st.file_uploader(
                "Or Upload Job Description Document",
                type=["pdf", "docx", "txt"],
                key="jd_file_uploader",
                help="Upload job description in PDF, DOCX, or TXT format",
            )
            if job_file:
                st.caption(f"📄 Selected Job File: `{job_file.name}`")

        st.markdown("---")
        st.subheader("Step 2: Upload Candidate Resumes")
        resume_uploads = st.file_uploader(
            "Upload Resumes (PDF, DOCX, TXT - Multiple Files Allowed)",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            key="resume_uploader",
            help="Select one or multiple resumes to screen against the job description",
        )

        if resume_uploads:
            st.info(f"📂 **{len(resume_uploads)} candidate resume(s)** selected for evaluation.")

        st.markdown("---")
        process_btn = st.button("🚀 Start Screening & Evaluation Pipeline", type="primary", use_container_width=True)

        if process_btn:
            active_job_input = job_file if job_file is not None else job_text
            job_fname = job_file.name if job_file else "Job_Description.txt"

            if not active_job_input or (isinstance(active_job_input, str) and not active_job_input.strip()):
                st.error("⚠️ Please provide a Job Description (either paste text or upload a document).")
            elif not resume_uploads:
                st.error("⚠️ Please upload at least one candidate resume to screen.")
            else:
                try:
                    with st.spinner("Executing semantic matching, skill evidence validation, and scoring..."):
                        resumes_data = [r.read() for r in resume_uploads]
                        resume_fnames = [r.name for r in resume_uploads]
                        batch_res = service.process_screening(
                            job_input=active_job_input,
                            resume_files=resumes_data,
                            job_filename=job_fname,
                            resume_filenames=resume_fnames,
                        )
                        st.session_state.screening_results = batch_res
                        st.success(f"Screening complete! Processed and ranked {batch_res.total_candidates} candidate(s).")
                except ValidationError as ve:
                    st.error(f"Validation Error: {str(ve)}")
                except Exception as e:
                    st.error(f"An error occurred during processing: {str(e)}")

    results: BatchScreeningResult = st.session_state.screening_results
    show_names = st.session_state.get("show_names", False)

    # TAB 2: CANDIDATE RANKING & PROFILES
    if results and results.ranked_candidates:
        with tab_results:
            st.subheader(f"🏆 Candidate Ranking for: {results.job_title}")

            # Top Summary Metrics (Translucent dark/light adaptive cards)
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            with mcol1:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-val">{results.total_candidates}</div><div class="metric-label">Candidates Evaluated</div></div>',
                    unsafe_allow_html=True,
                )
            with mcol2:
                top_cand = results.ranked_candidates[0]
                disp_top = top_cand.candidate_name if show_names else top_cand.anonymized_id
                st.markdown(
                    f'<div class="metric-card"><div class="metric-val">{top_cand.overall_score:.1f}%</div><div class="metric-label">Top Score ({disp_top})</div></div>',
                    unsafe_allow_html=True,
                )
            with mcol3:
                high_count = sum(1 for c in results.ranked_candidates if c.recommendation == "HIGH RELEVANCE")
                st.markdown(
                    f'<div class="metric-card"><div class="metric-val">{high_count}</div><div class="metric-label">High Relevance Matches</div></div>',
                    unsafe_allow_html=True,
                )
            with mcol4:
                avg_score = sum(c.overall_score for c in results.ranked_candidates) / len(results.ranked_candidates)
                st.markdown(
                    f'<div class="metric-card"><div class="metric-val">{avg_score:.1f}%</div><div class="metric-label">Average Batch Score</div></div>',
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # Interactive Filter Toolbar
            fcol1, fcol2 = st.columns([2, 2])
            with fcol1:
                filter_rec = st.multiselect(
                    "Filter by Recommendation Tier:",
                    options=["HIGH RELEVANCE", "RELEVANT", "MODERATE RELEVANCE", "LOW RELEVANCE"],
                    default=["HIGH RELEVANCE", "RELEVANT", "MODERATE RELEVANCE", "LOW RELEVANCE"],
                )
            with fcol2:
                min_score = st.slider("Minimum Overall Score (%)", min_value=0.0, max_value=100.0, value=0.0, step=5.0)

            filtered_candidates = [
                c for c in results.ranked_candidates
                if c.recommendation in filter_rec and c.overall_score >= min_score
            ]

            # Build leaderboard DataFrame
            leaderboard_data = []
            for c in filtered_candidates:
                cand_label = c.candidate_name if show_names else c.anonymized_id
                leaderboard_data.append({
                    "Rank": f"#{c.rank}",
                    "Candidate ID": c.anonymized_id,
                    "Candidate": cand_label,
                    "Overall Match": f"{c.overall_score:.1f}%",
                    "Semantic Skills": f"{c.score_breakdown.semantic_skill_match_score:.1f}%" if c.score_breakdown else "N/A",
                    "Evidence Score": f"{c.score_breakdown.skill_evidence_score:.1f}%" if c.score_breakdown else "N/A",
                    "Experience Score": f"{c.score_breakdown.experience_relevance_score:.1f}%" if c.score_breakdown else "N/A",
                    "Recommendation": c.recommendation,
                })
            df_leaderboard = pd.DataFrame(leaderboard_data)
            st.dataframe(df_leaderboard, use_container_width=True, hide_index=True)

            st.markdown("---")
            st.subheader("🔍 Deep-Dive Candidate Profile & Skill Evidence")

            cand_options = [
                f"Rank #{c.rank} - {c.candidate_name if show_names else c.anonymized_id} (Score: {c.overall_score:.1f}%)"
                for c in results.ranked_candidates
            ]
            selected_idx = st.selectbox(
                "Select Candidate to Inspect:",
                range(len(cand_options)),
                format_func=lambda i: cand_options[i],
            )
            cand = results.ranked_candidates[selected_idx]
            cand_display_label = cand.candidate_name if show_names else cand.anonymized_id

            col_p1, col_p2 = st.columns([1, 2])

            with col_p1:
                st.markdown(f"#### 👤 {cand_display_label}")
                st.caption(f"Candidate Identifier: `{cand.anonymized_id}`")

                rec_badge_class = {
                    "HIGH RELEVANCE": "badge-high",
                    "RELEVANT": "badge-rel",
                    "MODERATE RELEVANCE": "badge-mod",
                    "LOW RELEVANCE": "badge-low",
                }.get(cand.recommendation, "badge-mod")

                st.markdown(
                    f'<p>Status: <span class="{rec_badge_class}">{cand.recommendation}</span></p>',
                    unsafe_allow_html=True,
                )
                st.progress(min(1.0, cand.overall_score / 100.0), text=f"Overall Score: {cand.overall_score:.1f}%")

                if cand.score_breakdown:
                    st.markdown("##### 5-Pillar Score Breakdown")
                    st.write(f"- 🧠 **Semantic Skill Match:** {cand.score_breakdown.semantic_skill_match_score:.1f}%")
                    st.write(f"- 🔬 **Skill Evidence Validation:** {cand.score_breakdown.skill_evidence_score:.1f}%")
                    st.write(f"- ⏳ **Experience Relevance:** {cand.score_breakdown.experience_relevance_score:.1f}%")
                    st.write(f"- 🎓 **Education Relevance:** {cand.score_breakdown.education_relevance_score:.1f}%")
                    st.write(f"- 🛠️ **Projects & Certs:** {cand.score_breakdown.projects_certs_score:.1f}%")

            with col_p2:
                st.markdown("#### 💡 Explainable AI Rationale")
                st.info(cand.explanation)

                st.markdown("#### 🎯 Skill Alignment Breakdown")
                c_mat, c_miss = st.columns(2)

                with c_mat:
                    st.markdown("**✅ Matched & Partial Skills**")
                    all_matched = cand.matched_skills + cand.partial_skills
                    if all_matched:
                        for m in all_matched:
                            sim_pct = f"{m.similarity_score * 100:.0f}%" if m.similarity_score else "100%"
                            st.markdown(f'<span class="skill-tag">✓ {m.job_skill} ({sim_pct})</span>', unsafe_allow_html=True)
                    else:
                        st.write("No matching skills identified.")

                with c_miss:
                    st.markdown("**❌ Missing Skills**")
                    if cand.missing_skills:
                        for m in cand.missing_skills:
                            st.markdown(f'<span class="missing-tag">✗ {m.job_skill}</span>', unsafe_allow_html=True)
                    else:
                        st.write("No major required skills missing.")

                st.markdown("#### 🔬 Verified Skill Evidence (Contextual Validation)")
                if cand.evidence_items:
                    ev_data = []
                    for ev in cand.evidence_items:
                        ev_data.append({
                            "Skill": ev.skill_name,
                            "Evidence Level": f"Level {ev.evidence_level}",
                            "Confidence": ev.confidence_label,
                            "Source Section": ev.source_section,
                            "Contextual Snippet": ev.snippet,
                        })
                    st.dataframe(pd.DataFrame(ev_data), use_container_width=True, hide_index=True)
                else:
                    st.caption("No contextual evidence records found.")

                if cand.relevant_projects:
                    st.markdown("#### 📁 Key Projects Found")
                    for p in cand.relevant_projects[:4]:
                        st.markdown(f"- {p}")

        # TAB 3: VISUAL ANALYTICS & CANDIDATE COMPARISON
        with tab_analytics:
            st.subheader("📊 Visual Analytics & Candidate Comparison")

            # NAME REVEAL CHECKBOX IN VISUAL ANALYTICS (As requested)
            an_col1, an_col2 = st.columns([2, 2])
            with an_col1:
                reveal_names = st.checkbox(
                    "👁️ Reveal Candidate Real Names (Toggle from Anonymized IDs to Real Names)",
                    value=st.session_state.show_names,
                    key="reveal_names_checkbox_analytics",
                    help="When checked, real candidate names appear on charts and leaderboards.",
                )
                if reveal_names != st.session_state.show_names:
                    st.session_state.show_names = reveal_names
                    st.rerun()

            with an_col2:
                chart_type = st.radio(
                    "Select Comparison View:",
                    options=["Overall Match Ranking", "5-Pillar Radar Comparison", "Multi-Factor Stacked Breakdown"],
                    horizontal=True,
                )

            current_show_names = st.session_state.show_names
            c_names = [c.candidate_name if current_show_names else c.anonymized_id for c in results.ranked_candidates]
            c_scores = [c.overall_score for c in results.ranked_candidates]

            # Chart View 1: Overall Match Ranking
            if chart_type == "Overall Match Ranking":
                c_colors = ['#22C55E' if s >= 80 else '#3B82F6' if s >= 65 else '#F59E0B' if s >= 50 else '#EF4444' for s in c_scores]
                fig_scores = go.Figure(
                    data=[go.Bar(
                        x=c_names,
                        y=c_scores,
                        marker_color=c_colors,
                        text=[f"{s:.1f}%" for s in c_scores],
                        textposition='auto',
                    )]
                )
                fig_scores.update_layout(
                    title="Overall Candidate Match Scores",
                    xaxis_title="Candidate",
                    yaxis_title="Match Score (%)",
                    yaxis=dict(range=[0, 100]),
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_scores, use_container_width=True)

            # Chart View 2: Radar Polar Chart for Top Candidates
            elif chart_type == "5-Pillar Radar Comparison":
                pillars = ["Semantic Skills", "Skill Evidence", "Experience", "Education", "Projects & Certs"]
                fig_radar = go.Figure()

                top_n = min(3, len(results.ranked_candidates))
                colors = ["#38BDF8", "#A855F7", "#F59E0B"]

                for i in range(top_n):
                    cand_item = results.ranked_candidates[i]
                    lbl = cand_item.candidate_name if current_show_names else cand_item.anonymized_id
                    bd = cand_item.score_breakdown
                    if bd:
                        vals = [
                            bd.semantic_skill_match_score,
                            bd.skill_evidence_score,
                            bd.experience_relevance_score,
                            bd.education_relevance_score,
                            bd.projects_certs_score,
                        ]
                        vals.append(vals[0])  # close radar polygon
                        fig_radar.add_trace(go.Scatterpolar(
                            r=vals,
                            theta=pillars + [pillars[0]],
                            fill='toself',
                            name=f"#{cand_item.rank} {lbl}",
                            line_color=colors[i % len(colors)],
                        ))

                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 100]),
                        bgcolor="rgba(0,0,0,0)",
                    ),
                    title=f"5-Pillar Competency Radar (Top {top_n} Candidates)",
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_radar, use_container_width=True)

            # Chart View 3: Multi-Factor Grouped Breakdown
            else:
                breakdown_records = []
                for c in results.ranked_candidates:
                    c_lbl = c.candidate_name if current_show_names else c.anonymized_id
                    if c.score_breakdown:
                        breakdown_records.append({"Candidate": c_lbl, "Pillar": "Semantic Skills", "Score": c.score_breakdown.semantic_skill_match_score})
                        breakdown_records.append({"Candidate": c_lbl, "Pillar": "Skill Evidence", "Score": c.score_breakdown.skill_evidence_score})
                        breakdown_records.append({"Candidate": c_lbl, "Pillar": "Experience", "Score": c.score_breakdown.experience_relevance_score})
                        breakdown_records.append({"Candidate": c_lbl, "Pillar": "Education", "Score": c.score_breakdown.education_relevance_score})
                        breakdown_records.append({"Candidate": c_lbl, "Pillar": "Projects & Certs", "Score": c.score_breakdown.projects_certs_score})

                if breakdown_records:
                    df_bd = pd.DataFrame(breakdown_records)
                    fig_bd = px.bar(
                        df_bd,
                        x="Candidate",
                        y="Score",
                        color="Pillar",
                        barmode="group",
                        title="5-Pillar Competency Comparison across Candidates",
                        template="plotly_dark",
                        color_discrete_sequence=["#38BDF8", "#818CF8", "#34D399", "#FBBF24", "#F472B6"],
                    )
                    fig_bd.update_layout(
                        yaxis=dict(range=[0, 100]),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig_bd, use_container_width=True)

        # TAB 4: FAIRNESS & EVALUATION AUDIT
        with tab_fairness:
            st.subheader("⚖️ Fairness & Evaluation Audit")
            st.info(
                "**Merit-Based Evaluation Policy:** Candidate scoring and ranking are computed strictly on verified "
                "technical competencies, experience, education, and contextual project evidence."
            )

            st.markdown("### 📈 Batch Selection & Statistical Parity Metrics")

            f_metrics = results.fairness_metrics
            fc1, fc2, fc3, fc4 = st.columns(4)
            with fc1:
                st.metric("Total Candidates Audited", f_metrics.get("total_candidates_audited", 0))
            with fc2:
                st.metric("Batch Selection Rate", f"{f_metrics.get('selection_rate_pct', 0.0):.1f}%")
            with fc3:
                st.metric("Mean Score", f"{f_metrics.get('score_mean', 0.0):.1f}")
            with fc4:
                st.metric("Demographic Parity Diff", f"{f_metrics.get('demographic_parity_difference', 0.0):.3f}")

            st.markdown("<br>", unsafe_allow_html=True)
            st.write(f"**Disparity Status:** `{f_metrics.get('disparity_status', 'Evaluated')}`")

            # Export options
            st.markdown("---")
            st.subheader("💾 Export Screening Results")
            export_df = pd.DataFrame(leaderboard_data)
            csv_data = export_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Screening Leaderboard as CSV",
                data=csv_data,
                file_name="screening_results.csv",
                mime="text/csv",
                type="primary",
            )
    else:
        with tab_results:
            st.info("👋 Upload a job description and resumes in Tab 1, or click '🚀 Load Demo Dataset' in the top action bar to inspect results.")


if __name__ == "__main__":
    main()
