"""
AI-Driven Resume Screening and Skill Validation System with Bias Reduction Techniques.
Streamlit Recruiter Dashboard Application.
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
from modules.fairness.bias_reduction import BiasReductionPipeline
from modules.document_processing.file_validator import ValidationError
from config import SCORING_WEIGHTS, DATABASE_PATH
import database

# Page Configuration
st.set_page_config(
    page_title="AI Resume Screening & Skill Validation",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #E2E8F0;
        text-align: center;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1E40AF;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
    }
    .badge-high {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
    }
    .badge-rel {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
    }
    .badge-mod {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
    }
    .badge-low {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
    }
    .skill-tag {
        display: inline-block;
        background-color: #EFF6FF;
        color: #1D4ED8;
        padding: 3px 8px;
        border-radius: 4px;
        margin: 2px;
        font-size: 0.85rem;
        border: 1px solid #BFDBFE;
    }
    .missing-tag {
        display: inline-block;
        background-color: #FEF2F2;
        color: #B91C1C;
        padding: 3px 8px;
        border-radius: 4px;
        margin: 2px;
        font-size: 0.85rem;
        border: 1px solid #FECACA;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_screening_service():
    """Initializes and caches the screening service."""
    return ScreeningService()


def render_header():
    """Renders application header and novelty banner."""
    st.markdown('<div class="main-header">🎯 AI-Driven Resume Screening & Skill Validation System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Fair, Explainable, Semantic Candidate Matching with Evidence Validation & Bias Reduction</div>', unsafe_allow_html=True)

    with st.expander("ℹ️ System Architecture & Academic Novelty Highlights", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("**1. Semantic Matching**")
            st.caption("Sentence-BERT semantic embeddings capture conceptual skill alignment beyond keyword matching.")
        with col2:
            st.markdown("**2. Evidence Validation**")
            st.caption("Verifies skills against contextual resume evidence across projects, experience, and certifications.")
        with col3:
            st.markdown("**3. Bias Reduction**")
            st.caption("Best-effort PII masking (Name, Email, Phone, Address, Dates) with Fairlearn audit metrics.")
        with col4:
            st.markdown("**4. Explainable AI**")
            st.caption("Transparent multi-factor weighted scoring breakdown and data-driven candidate rationales.")


def render_sidebar():
    """Renders sidebar controls and system configuration."""
    st.sidebar.title("⚙️ Control Panel")

    st.sidebar.markdown("### 📊 Scoring Weights")
    st.sidebar.caption("Transparent, configurable evaluation criteria")
    st.sidebar.info(
        f"- **Semantic Skill Match:** {SCORING_WEIGHTS['semantic_skill_match']*100:.0f}%\n"
        f"- **Skill Evidence Validation:** {SCORING_WEIGHTS['skill_evidence']*100:.0f}%\n"
        f"- **Experience Relevance:** {SCORING_WEIGHTS['experience_relevance']*100:.0f}%\n"
        f"- **Education Relevance:** {SCORING_WEIGHTS['education_relevance']*100:.0f}%\n"
        f"- **Projects & Certifications:** {SCORING_WEIGHTS['projects_certs']*100:.0f}%"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧪 Academic Demo Mode")
    load_demo = st.sidebar.button("🚀 Load Sample Job & 3 Resumes", help="Loads Senior ML Engineer job and 3 pre-configured resumes (Alex Turner, Priya Sharma, John Doe)")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔒 Privacy & Bias Reduction")
    show_names = st.sidebar.checkbox("Reveal Candidate Names", value=False, help="Toggle between Anonymized IDs (ANON-XXXX) and candidate names.")
    st.sidebar.caption("When unchecked, recruiters evaluate candidates purely by anonymized qualifications.")

    return load_demo, show_names


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


def main():
    render_header()
    load_demo, show_names = render_sidebar()
    service = get_screening_service()

    if "screening_results" not in st.session_state:
        st.session_state.screening_results = None
    if "job_text_input" not in st.session_state:
        st.session_state.job_text_input = ""

    # Handle demo loading
    if load_demo:
        demo_job, demo_resumes, demo_filenames, demo_job_name = load_sample_data()
        with st.spinner("Processing 3 demo candidates through full screening pipeline..."):
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
        "⚖️ 4. Fairness & Bias Audit",
    ])

    with tab_input:
        st.subheader("Step 1: Provide Job Description")
        col_jd_text, col_jd_file = st.columns([3, 2])

        with col_jd_text:
            job_text = st.text_area(
                "Paste Job Description Text",
                value=st.session_state.job_text_input,
                height=220,
                placeholder="Paste the full job requirements, skills, experience, and responsibilities...",
            )

        with col_jd_file:
            job_file = st.file_uploader(
                "Or Upload Job Description (PDF / DOCX)",
                type=["pdf", "docx", "txt"],
                key="jd_file_uploader",
            )
            if job_file:
                st.caption(f"Uploaded file: `{job_file.name}`")

        st.markdown("---")
        st.subheader("Step 2: Upload Candidate Resumes")
        resume_uploads = st.file_uploader(
            "Upload Resumes (PDF, DOCX, TXT - Multiple Allowed)",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            key="resume_uploader",
        )

        if resume_uploads:
            st.caption(f"📂 {len(resume_uploads)} resume file(s) selected.")

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
                    with st.spinner("Running complete screening, skill validation, and bias reduction pipeline..."):
                        resumes_data = [r.read() for r in resume_uploads]
                        resume_fnames = [r.name for r in resume_uploads]
                        batch_res = service.process_screening(
                            job_input=active_job_input,
                            resume_files=resumes_data,
                            job_filename=job_fname,
                            resume_filenames=resume_fnames,
                        )
                        st.session_state.screening_results = batch_res
                        st.success(f"Successfully processed and ranked {batch_res.total_candidates} candidate(s)!")
                except ValidationError as ve:
                    st.error(f"Validation Error: {str(ve)}")
                except Exception as e:
                    st.error(f"An error occurred during processing: {str(e)}")

    # Display Results if Available
    results: BatchScreeningResult = st.session_state.screening_results

    if results and results.ranked_candidates:
        with tab_results:
            st.subheader(f"🏆 Candidate Ranking for: {results.job_title}")

            # Top Summary Metrics
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            with mcol1:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-val">{results.total_candidates}</div><div class="metric-label">Candidates Evaluated</div></div>',
                    unsafe_allow_html=True,
                )
            with mcol2:
                top_cand = results.ranked_candidates[0]
                disp_name = top_cand.candidate_name if show_names else top_cand.anonymized_id
                st.markdown(
                    f'<div class="metric-card"><div class="metric-val">{top_cand.overall_score:.1f}%</div><div class="metric-label">Top Score ({disp_name})</div></div>',
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
                    f'<div class="metric-card"><div class="metric-val">{avg_score:.1f}%</div><div class="metric-label">Average Score</div></div>',
                    unsafe_allow_html=True,
                )

            st.markdown("### 📋 Leaderboard Table")

            # Build leaderboard DataFrame
            leaderboard_data = []
            for c in results.ranked_candidates:
                cand_label = c.candidate_name if show_names else c.anonymized_id
                leaderboard_data.append({
                    "Rank": f"#{c.rank}",
                    "Candidate ID": c.anonymized_id,
                    "Candidate Name": cand_label,
                    "Overall Match": f"{c.overall_score:.1f}%",
                    "Semantic Skill Score": f"{c.score_breakdown.semantic_skill_match_score:.1f}%" if c.score_breakdown else "N/A",
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
            selected_idx = st.selectbox("Select Candidate to Inspect:", range(len(cand_options)), format_func=lambda i: cand_options[i])
            cand = results.ranked_candidates[selected_idx]
            cand_display_label = cand.candidate_name if show_names else cand.anonymized_id

            col_p1, col_p2 = st.columns([1, 2])

            with col_p1:
                st.markdown(f"#### 👤 {cand_display_label}")
                st.caption(f"Internal ID: `{cand.candidate_id}` | Recruiter ID: `{cand.anonymized_id}`")

                rec_badge_class = {
                    "HIGH RELEVANCE": "badge-high",
                    "RELEVANT": "badge-rel",
                    "MODERATE RELEVANCE": "badge-mod",
                    "LOW RELEVANCE": "badge-low",
                }.get(cand.recommendation, "badge-mod")

                st.markdown(
                    f'<p>Recommendation: <span class="{rec_badge_class}">{cand.recommendation}</span></p>',
                    unsafe_allow_html=True,
                )
                st.progress(min(1.0, cand.overall_score / 100.0), text=f"Overall Score: {cand.overall_score:.1f}%")

                if cand.score_breakdown:
                    st.markdown("##### Transparent Score Breakdown")
                    st.write(f"- 🧠 **Semantic Skill Match:** {cand.score_breakdown.semantic_skill_match_score:.1f}%")
                    st.write(f"- 🔬 **Skill Evidence Validation:** {cand.score_breakdown.skill_evidence_score:.1f}%")
                    st.write(f"- ⏳ **Experience Relevance:** {cand.score_breakdown.experience_relevance_score:.1f}%")
                    st.write(f"- 🎓 **Education Relevance:** {cand.score_breakdown.education_relevance_score:.1f}%")
                    st.write(f"- 🛠️ **Projects & Certs:** {cand.score_breakdown.projects_certs_score:.1f}%")

                # PII Masking Summary
                if cand.pii_summary:
                    st.markdown("##### 🛡️ Masked PII Elements")
                    for pii_type, count in cand.pii_summary.items():
                        st.caption(f"✓ {pii_type.title()}: {count} instance(s) masked")

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

                st.markdown("#### 🔬 Verified Skill Evidence (Novelty Pillar)")
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

        with tab_analytics:
            st.subheader("📊 Visual Analytics & Candidate Comparison")

            # Chart 1: Candidate Overall Scores Comparison
            c_names = [c.candidate_name if show_names else c.anonymized_id for c in results.ranked_candidates]
            c_scores = [c.overall_score for c in results.ranked_candidates]
            c_colors = ['#16A34A' if s >= 80 else '#2563EB' if s >= 65 else '#D97706' if s >= 50 else '#DC2626' for s in c_scores]

            fig_scores = go.Figure(
                data=[go.Bar(x=c_names, y=c_scores, marker_color=c_colors, text=[f"{s:.1f}%" for s in c_scores], textposition='auto')]
            )
            fig_scores.update_layout(
                title="Overall Candidate Match Scores",
                xaxis_title="Candidate",
                yaxis_title="Match Score (%)",
                yaxis=dict(range=[0, 100]),
                template="plotly_white",
            )
            st.plotly_chart(fig_scores, use_container_width=True)

            # Chart 2: Multi-Factor Breakdown Grouped Bar Chart
            breakdown_records = []
            for c in results.ranked_candidates:
                c_lbl = c.candidate_name if show_names else c.anonymized_id
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
                    title="Transparent 5-Pillar Score Breakdown per Candidate",
                    template="plotly_white",
                )
                fig_bd.update_layout(yaxis=dict(range=[0, 100]))
                st.plotly_chart(fig_bd, use_container_width=True)

        with tab_fairness:
            st.subheader("⚖️ Fairness, Bias Reduction & Audit Metrics")
            st.info(
                "**Bias Reduction Policy:** This system masks Personally Identifiable Information (PII) before evaluation "
                "and uses strictly merit-based qualification features to calculate candidate rankings."
            )

            bias_stmt = BiasReductionPipeline.get_bias_mitigation_statement()
            st.markdown(f"**Methodology:** {bias_stmt['methodology']}")
            st.markdown(f"**Scoring Basis:** {bias_stmt['scoring_basis']}")
            st.caption(f"**Academic Note:** {bias_stmt['disclaimer']}")

            st.markdown("---")
            st.subheader("📈 Batch Fairness Audit (Fairlearn Metrics)")

            f_metrics = results.fairness_metrics
            fc1, fc2, fc3, fc4 = st.columns(4)
            with fc1:
                st.metric("Batch Size", f_metrics.get("total_candidates_audited", 0))
            with fc2:
                st.metric("Selection Rate", f"{f_metrics.get('selection_rate_pct', 0.0):.1f}%")
            with fc3:
                st.metric("Score Mean", f"{f_metrics.get('score_mean', 0.0):.1f}")
            with fc4:
                st.metric("Demographic Parity Diff", f"{f_metrics.get('demographic_parity_difference', 0.0):.3f}")

            st.write(f"**Audit Status:** `{f_metrics.get('disparity_status', 'Evaluated')}`")

            # Export options
            st.markdown("---")
            st.subheader("💾 Export Screening Results")
            export_df = pd.DataFrame(leaderboard_data)
            csv_data = export_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv_data,
                file_name="screening_results.csv",
                mime="text/csv",
            )
    else:
        with tab_results:
            st.info("👋 Upload a job description and resumes in Tab 1, or click 'Load Sample Job & 3 Resumes' in the sidebar to start.")


if __name__ == "__main__":
    main()
