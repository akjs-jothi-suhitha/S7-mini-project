"""
Script to generate realistic DOCX and PDF resume files and job descriptions.
"""

from pathlib import Path
import docx

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESUMES_DIR = DATA_DIR / "sample_resumes"
JOBS_DIR = DATA_DIR / "sample_job_descriptions"

RESUMES_DIR.mkdir(parents=True, exist_ok=True)
JOBS_DIR.mkdir(parents=True, exist_ok=True)

# 1. Candidate A DOCX
doc_a = docx.Document()
doc_a.add_heading("Alex Turner", 0)
doc_a.add_paragraph("Email: alex.turner@techmail.io | Phone: +1-555-019-2834 | DOB: 14/08/1995")
doc_a.add_paragraph("Address: 742 Evergreen Terrace, Seattle, WA 98101 | LinkedIn: https://linkedin.com/in/alexturner-ai")

doc_a.add_heading("Professional Summary", level=1)
doc_a.add_paragraph(
    "Senior AI Engineer with 4.5 years of industry experience developing production Machine Learning, "
    "Deep Learning, and NLP systems. Proven track record deploying transformer-based microservices to "
    "AWS cloud infrastructure using Docker and Kubernetes."
)

doc_a.add_heading("Technical Skills", level=1)
doc_a.add_paragraph("• Programming: Python, SQL, C++, Bash")
doc_a.add_paragraph("• ML/AI: Machine Learning, Deep Learning, Natural Language Processing, PyTorch, Scikit-learn, TensorFlow, spaCy, Transformers, LLMs")
doc_a.add_paragraph("• Cloud & DevOps: AWS, Docker, Kubernetes, Git, CI/CD, Linux")
doc_a.add_paragraph("• Databases: PostgreSQL, MongoDB, Redis")
doc_a.add_paragraph("• Web & APIs: FastAPI, REST API, Microservices")

doc_a.add_heading("Work Experience", level=1)
p1 = doc_a.add_paragraph()
p1.add_run("Senior Machine Learning Engineer | DataCore AI Labs (2021 - Present)\n").bold = True
p1.add_run("- Designed and engineered scalable NLP pipelines processing 1.5M text records daily using Python, PyTorch, and Hugging Face transformers.\n")
p1.add_run("- Implemented deep learning classification and semantic search models using cosine similarity and vector embeddings, improving retrieval accuracy by 28%.\n")
p1.add_run("- Containerized model inference microservices with Docker and deployed them to AWS EKS (Kubernetes) with automated CI/CD pipelines.\n")
p1.add_run("- Reduced inference latency by 40% via model quantization and FastAPI asynchronous endpoints.")

p2 = doc_a.add_paragraph()
p2.add_run("AI Developer | CloudSphere Tech (2019 - 2021)\n").bold = True
p2.add_run("- Developed machine learning predictive models using Scikit-learn, Pandas, and NumPy for customer churn prediction.\n")
p2.add_run("- Built automated ETL data pipelines querying PostgreSQL databases with optimized SQL queries.\n")
p2.add_run("- Collaborated with engineering teams to deploy RESTful APIs and monitor production model drift.")

doc_a.add_heading("Projects", level=1)
p3 = doc_a.add_paragraph()
p3.add_run("AI-Driven Semantic Search & Resume Screening System:\n").bold = True
p3.add_run("Developed an automated recruitment screening system using Python, spaCy, and Sentence-BERT transformers. Containerized with Docker and integrated Fairlearn bias mitigation metrics.\n")
p3.add_run("Multi-Modal NLP Document Analyzer:\n").bold = True
p3.add_run("Built deep learning text parsing application using PyTorch, FastAPI, and PostgreSQL with high-throughput batch inference on AWS.")

doc_a.add_heading("Education", level=1)
doc_a.add_paragraph("Master of Science in Computer Science (Specialization in AI/ML) | University of Washington, 2019")
doc_a.add_paragraph("Bachelor of Technology in Information Technology | State University, 2017")

doc_a.add_heading("Certifications", level=1)
doc_a.add_paragraph("• AWS Certified Machine Learning - Specialty (2022)")
doc_a.add_paragraph("• Deep Learning Specialization - Coursera / DeepLearning.AI")

docx_a_path = RESUMES_DIR / "candidate_a_alex_turner.docx"
doc_a.save(str(docx_a_path))

# 2. Candidate B DOCX
doc_b = docx.Document()
doc_b.add_heading("Priya Sharma", 0)
doc_b.add_paragraph("Email: priya.sharma@domainexample.com | Phone: +1-555-837-4921 | DOB: 22/05/1997")
doc_b.add_paragraph("Address: 1204 Pine Street, San Jose, CA 95112 | LinkedIn: https://linkedin.com/in/priyasharma-data")

doc_b.add_heading("Professional Summary", level=1)
doc_b.add_paragraph(
    "Data Scientist with 2.5 years of experience developing machine learning models, statistical analysis workflows, "
    "and data pipelines using Python and SQL. Passionate about natural language processing and applied data analytics."
)

doc_b.add_heading("Technical Skills", level=1)
doc_b.add_paragraph("• Programming: Python, SQL, R")
doc_b.add_paragraph("• Machine Learning & Data: Machine Learning, Scikit-learn, Pandas, NumPy, Data Analysis, Natural Language Processing, Statistical Analysis")
doc_b.add_paragraph("• Frameworks & Tools: Flask, Git, GitHub, Tableau, Power BI")
doc_b.add_paragraph("• Databases: MySQL, SQLite, PostgreSQL")

doc_b.add_heading("Work Experience", level=1)
pb1 = doc_b.add_paragraph()
pb1.add_run("Data Scientist | Insight Analytics Corp (2022 - Present)\n").bold = True
pb1.add_run("- Developed machine learning models using Python and Scikit-learn to forecast sales revenue and customer retention.\n")
pb1.add_run("- Created data pipelines and extracted insights using complex SQL queries on PostgreSQL and MySQL databases.\n")
pb1.add_run("- Performed text mining and sentiment analysis using Python and NLTK on customer feedback reviews.\n")
pb1.add_run("- Built interactive dashboards and data visualizations using Plotly and Tableau for executive decision-making.")

pb2 = doc_b.add_paragraph()
pb2.add_run("Junior Data Analyst | Apex Data Systems (2021 - 2022)\n").bold = True
pb2.add_run("- Implemented exploratory data analysis and predictive modeling using Pandas, NumPy, and Scikit-learn.\n")
pb2.add_run("- Assisted in building lightweight REST APIs with Flask to serve prediction results.")

doc_b.add_heading("Projects", level=1)
pb3 = doc_b.add_paragraph()
pb3.add_run("Customer Feedback NLP Classifier:\n").bold = True
pb3.add_run("Implemented a text classification project using Python, Scikit-learn, and TF-IDF to categorize 50k customer support tickets.\n")
pb3.add_run("Financial Forecasting Web App:\n").bold = True
pb3.add_run("Built a forecasting dashboard in Python and Flask using machine learning regression models and SQLite.")

doc_b.add_heading("Education", level=1)
doc_b.add_paragraph("Bachelor of Technology in Computer Science | National Institute of Technology, 2021")

doc_b.add_heading("Certifications", level=1)
doc_b.add_paragraph("• Machine Learning Specialization - Coursera")
doc_b.add_paragraph("• Microsoft Certified: Data Analyst Associate")

docx_b_path = RESUMES_DIR / "candidate_b_priya_sharma.docx"
doc_b.save(str(docx_b_path))

# 3. Candidate C DOCX
doc_c = docx.Document()
doc_c.add_heading("John Doe", 0)
doc_c.add_paragraph("Email: john.doe@emailservice.net | Phone: +1-555-672-9102 | DOB: 10/11/1998")
doc_c.add_paragraph("Address: 45 Elm Court, Austin, TX 78701 | LinkedIn: https://linkedin.com/in/johndoe-webdev")

doc_c.add_heading("Professional Summary", level=1)
doc_c.add_paragraph(
    "Junior Web Developer with 1 year of experience building responsive websites and enterprise web interfaces using "
    "Java, HTML, CSS, and JavaScript. Seeking new opportunities in software development."
)

doc_c.add_heading("Technical Skills", level=1)
doc_c.add_paragraph("• Programming: Java, HTML, CSS, JavaScript")
doc_c.add_paragraph("• Frameworks: Spring Boot, Bootstrap")
doc_c.add_paragraph("• Databases: MySQL")
doc_c.add_paragraph("• Tools: Git, Eclipse, Windows")

doc_c.add_heading("Work Experience", level=1)
pc1 = doc_c.add_paragraph()
pc1.add_run("Junior Java Developer | Legacy Web Solutions (2023 - Present)\n").bold = True
pc1.add_run("- Developed responsive web interfaces using HTML, CSS, and JavaScript.\n")
pc1.add_run("- Built basic backend CRUD endpoints using Java and Spring Boot connected to MySQL.\n")
pc1.add_run("- Maintained legacy website codebases and fixed styling defects across browser versions.")

doc_c.add_heading("Projects", level=1)
pc2 = doc_c.add_paragraph()
pc2.add_run("E-Commerce Catalog Website:\n").bold = True
pc2.add_run("Created an online storefront product catalog utilizing HTML, CSS, JavaScript, and Java Spring Boot with MySQL database.\n")
pc2.add_run("Student Management Portal:\n").bold = True
pc2.add_run("Built a web portal for managing student attendance and grades using Java, HTML, and CSS.")

doc_c.add_heading("Education", level=1)
doc_c.add_paragraph("Bachelor of Science in Information Systems | City College, 2023")

doc_c.add_heading("Certifications", level=1)
doc_c.add_paragraph("• Oracle Certified Associate, Java SE Programmer")

docx_c_path = RESUMES_DIR / "candidate_c_john_doe.docx"
doc_c.save(str(docx_c_path))

# 4. Senior ML Engineer Job Description DOCX
doc_job = docx.Document()
doc_job.add_heading("Senior Machine Learning Engineer", 0)
doc_job.add_paragraph("Company: NextGen AI Solutions | Location: Remote / Hybrid")
doc_job.add_heading("About the Role", level=1)
doc_job.add_paragraph("We are seeking an experienced Senior Machine Learning Engineer to design, build, and deploy state-of-the-art NLP and predictive AI solutions. You will work on cutting-edge deep learning models, transformer architectures, and scalable data pipelines.")

doc_job.add_heading("Requirements & Must-Haves", level=1)
doc_job.add_paragraph("• Strong proficiency in Python and SQL.")
doc_job.add_paragraph("• Deep expertise in Machine Learning, Natural Language Processing (NLP), and Deep Learning.")
doc_job.add_paragraph("• Hands-on experience with PyTorch, Scikit-learn, and Transformers (Hugging Face / spaCy).")
doc_job.add_paragraph("• Experience containerizing and deploying models using Docker and Kubernetes.")
doc_job.add_paragraph("• Familiarity with Cloud platforms (AWS or Azure).")
doc_job.add_paragraph("• Minimum 3+ years of professional experience in machine learning or AI engineering.")
doc_job.add_paragraph("• Bachelor's or Master's degree in Computer Science, Data Science, or related engineering discipline.")

doc_job.add_heading("Preferred & Good to Have", level=1)
doc_job.add_paragraph("• Experience with FastAPI, Microservices, and REST API development.")
doc_job.add_paragraph("• Knowledge of MLOps, CI/CD pipelines, and Git.")
doc_job.add_paragraph("• Experience with Large Language Models (LLMs) and vector databases.")

doc_job.add_heading("Key Responsibilities", level=1)
doc_job.add_paragraph("• Design and implement scalable machine learning models and NLP pipelines.")
doc_job.add_paragraph("• Collaborate with software engineers to integrate AI models into production microservices.")
doc_job.add_paragraph("• Optimize inference latency, monitor model performance, and maintain automated retraining workflows.")

docx_job_path = JOBS_DIR / "senior_ml_engineer.docx"
doc_job.save(str(docx_job_path))

print("Successfully generated DOCX files:")
print(f"- {docx_a_path}")
print(f"- {docx_b_path}")
print(f"- {docx_c_path}")
print(f"- {docx_job_path}")
