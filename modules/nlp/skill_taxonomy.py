"""
Skill Taxonomy Module.
Defines canonical technical skills, domain categories, and comprehensive alias/synonym mappings.
"""

from typing import Dict, List, Optional, Set, Tuple


class SkillTaxonomy:
    """Manages technical skill categorization, aliases, and normalization."""

    # Skill Categories and their Canonical Skills
    TAXONOMY: Dict[str, List[str]] = {
        "Programming Languages": [
            "Python", "Java", "C", "C++", "C#", "SQL", "JavaScript", "TypeScript",
            "Go", "Rust", "R", "PHP", "Ruby", "Kotlin", "Swift", "Scala", "Dart",
            "Shell Scripting", "Bash", "HTML", "CSS"
        ],
        "Machine Learning & AI": [
            "Machine Learning", "Deep Learning", "Artificial Intelligence",
            "Natural Language Processing", "Computer Vision", "Reinforcement Learning",
            "Large Language Models", "Generative AI", "Transformers", "Information Retrieval",
            "Predictive Modeling", "Neural Networks", "Feature Engineering", "Data Mining",
            "Recommendation Systems"
        ],
        "Frameworks & Libraries": [
            "TensorFlow", "PyTorch", "Keras", "Scikit-learn", "Pandas", "NumPy",
            "Matplotlib", "Seaborn", "Plotly", "SciPy", "spaCy", "NLTK", "Hugging Face",
            "FastAPI", "Django", "Flask", "React", "Next.js", "Vue.js", "Angular",
            "Node.js", "Express.js", "Spring Boot", "ASP.NET", ".NET", "Tailwind CSS",
            "Bootstrap", "GraphQL", "Streamlit"
        ],
        "Cloud & DevOps": [
            "AWS", "Azure", "Google Cloud Platform", "Docker", "Kubernetes", "Git",
            "GitHub", "GitLab", "CI/CD", "Jenkins", "Terraform", "Ansible", "Linux",
            "Unix", "Nginx", "Apache", "Kafka", "RabbitMQ", "Celery", "Airflow"
        ],
        "Databases & Storage": [
            "PostgreSQL", "MySQL", "MongoDB", "SQLite", "Redis", "Elasticsearch",
            "Oracle", "Cassandra", "DynamoDB", "Firebase", "Snowflake", "BigQuery"
        ],
        "Software Engineering & Architecture": [
            "REST API", "Microservices", "Object-Oriented Programming", "Design Patterns",
            "Data Structures", "Algorithms", "System Design", "Agile", "Scrum",
            "Test-Driven Development", "Unit Testing", "PyTest"
        ],
        "Data Science & Analytics": [
            "Data Analysis", "Data Science", "Data Visualization", "Statistical Analysis",
            "Business Intelligence", "Tableau", "Power BI", "ETL", "Data Pipelines"
        ],
    }

    # Alias / Synonym Dictionary mapping variations to Canonical Skill Name
    SYNONYM_MAP: Dict[str, str] = {
        # ML / AI aliases
        "ml": "Machine Learning",
        "machine learning": "Machine Learning",
        "dl": "Deep Learning",
        "deep learning": "Deep Learning",
        "ai": "Artificial Intelligence",
        "artificial intelligence": "Artificial Intelligence",
        "nlp": "Natural Language Processing",
        "natural language processing": "Natural Language Processing",
        "cv": "Computer Vision",
        "computer vision": "Computer Vision",
        "llm": "Large Language Models",
        "llms": "Large Language Models",
        "large language model": "Large Language Models",
        "large language models": "Large Language Models",
        "genai": "Generative AI",
        "generative ai": "Generative AI",
        "transformers": "Transformers",
        "huggingface": "Hugging Face",
        "hugging face": "Hugging Face",
        "neural network": "Neural Networks",
        "neural networks": "Neural Networks",

        # Frameworks & Libraries
        "pytorch": "PyTorch",
        "py torch": "PyTorch",
        "torch": "PyTorch",
        "tensorflow": "TensorFlow",
        "tf": "TensorFlow",
        "scikit learn": "Scikit-learn",
        "scikit-learn": "Scikit-learn",
        "sklearn": "Scikit-learn",
        "spacy": "spaCy",
        "nltk": "NLTK",
        "pandas": "Pandas",
        "numpy": "NumPy",
        "matplotlib": "Matplotlib",
        "seaborn": "Seaborn",
        "plotly": "Plotly",
        "scipy": "SciPy",
        "streamlit": "Streamlit",
        "fastapi": "FastAPI",
        "fast api": "FastAPI",
        "django": "Django",
        "flask": "Flask",
        "react": "React",
        "reactjs": "React",
        "react.js": "React",
        "nextjs": "Next.js",
        "next.js": "Next.js",
        "vue": "Vue.js",
        "vuejs": "Vue.js",
        "vue.js": "Vue.js",
        "angular": "Angular",
        "angularjs": "Angular",
        "nodejs": "Node.js",
        "node.js": "Node.js",
        "node": "Node.js",
        "express": "Express.js",
        "expressjs": "Express.js",
        "express.js": "Express.js",
        "spring boot": "Spring Boot",
        "springboot": "Spring Boot",
        "dotnet": ".NET",
        ".net": ".NET",
        "asp.net": "ASP.NET",

        # Languages
        "python": "Python",
        "py": "Python",
        "python3": "Python",
        "java": "Java",
        "c++": "C++",
        "cpp": "C++",
        "c#": "C#",
        "c sharp": "C#",
        "csharp": "C#",
        "c": "C",
        "sql": "SQL",
        "javascript": "JavaScript",
        "js": "JavaScript",
        "typescript": "TypeScript",
        "ts": "TypeScript",
        "golang": "Go",
        "go": "Go",
        "rust": "Rust",
        "r": "R",
        "php": "PHP",
        "ruby": "Ruby",
        "kotlin": "Kotlin",
        "swift": "Swift",
        "scala": "Scala",
        "html": "HTML",
        "html5": "HTML",
        "css": "CSS",
        "css3": "CSS",
        "bash": "Bash",
        "shell": "Shell Scripting",
        "shell scripting": "Shell Scripting",

        # Cloud & DevOps
        "aws": "AWS",
        "amazon web services": "AWS",
        "azure": "Azure",
        "microsoft azure": "Azure",
        "gcp": "Google Cloud Platform",
        "google cloud": "Google Cloud Platform",
        "google cloud platform": "Google Cloud Platform",
        "docker": "Docker",
        "k8s": "Kubernetes",
        "kubernetes": "Kubernetes",
        "git": "Git",
        "github": "GitHub",
        "gitlab": "GitLab",
        "ci/cd": "CI/CD",
        "cicd": "CI/CD",
        "jenkins": "Jenkins",
        "terraform": "Terraform",
        "ansible": "Ansible",
        "linux": "Linux",
        "unix": "Unix",
        "kafka": "Kafka",
        "rabbitmq": "RabbitMQ",
        "airflow": "Airflow",

        # Databases
        "postgres": "PostgreSQL",
        "postgresql": "PostgreSQL",
        "mysql": "MySQL",
        "mongodb": "MongoDB",
        "mongo": "MongoDB",
        "sqlite": "SQLite",
        "sqlite3": "SQLite",
        "redis": "Redis",
        "elasticsearch": "Elasticsearch",
        "elastic search": "Elasticsearch",
        "oracle": "Oracle",
        "dynamodb": "DynamoDB",
        "firebase": "Firebase",
        "snowflake": "Snowflake",
        "bigquery": "BigQuery",

        # Architecture & Engineering
        "rest": "REST API",
        "rest api": "REST API",
        "rest apis": "REST API",
        "restful": "REST API",
        "restful api": "REST API",
        "restful apis": "REST API",
        "microservices": "Microservices",
        "microservice": "Microservices",
        "oop": "Object-Oriented Programming",
        "object oriented programming": "Object-Oriented Programming",
        "dsa": "Data Structures",
        "data structures": "Data Structures",
        "algorithms": "Algorithms",
        "system design": "System Design",
        "tdd": "Test-Driven Development",
        "test driven development": "Test-Driven Development",
        "pytest": "PyTest",
        "unit test": "Unit Testing",
        "unit testing": "Unit Testing",

        # Data Science & Analytics
        "data analysis": "Data Analysis",
        "data science": "Data Science",
        "data visualization": "Data Visualization",
        "tableau": "Tableau",
        "power bi": "Power BI",
        "powerbi": "Power BI",
        "etl": "ETL",
        "data pipeline": "Data Pipelines",
        "data pipelines": "Data Pipelines",
    }

    @classmethod
    def normalize(cls, skill: str) -> str:
        """
        Normalizes a skill string to its canonical taxonomy name.

        Args:
            skill: Raw skill string.

        Returns:
            Canonical skill name if recognized, else cleaned title-cased string.
        """
        if not skill:
            return ""
        cleaned = skill.strip().lower()
        if cleaned in cls.SYNONYM_MAP:
            return cls.SYNONYM_MAP[cleaned]

        # Strip extra punctuation and check
        cleaned_simple = cleaned.replace("-", " ").replace(".", "").strip()
        if cleaned_simple in cls.SYNONYM_MAP:
            return cls.SYNONYM_MAP[cleaned_simple]

        # Return original with appropriate capitalization
        return skill.strip().title()

    @classmethod
    def get_category(cls, canonical_skill: str) -> str:
        """Finds the category for a given canonical skill name."""
        for category, skills in cls.TAXONOMY.items():
            if canonical_skill in skills or any(s.lower() == canonical_skill.lower() for s in skills):
                return category
        return "Other Technical Skills"

    @classmethod
    def get_all_canonical_skills(cls) -> Set[str]:
        """Returns a set of all canonical skills in the taxonomy."""
        all_skills = set()
        for skills in cls.TAXONOMY.values():
            all_skills.update(skills)
        return all_skills

    @classmethod
    def get_all_searchable_terms(cls) -> Dict[str, str]:
        """Returns mapping from all lowercase searchable terms to canonical names."""
        terms = {}
        for syn, canon in cls.SYNONYM_MAP.items():
            terms[syn] = canon
        for canon in cls.get_all_canonical_skills():
            terms[canon.lower()] = canon
        return terms
