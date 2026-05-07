from flask import Flask, render_template, request, session, redirect, flash, send_file
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

import os
import re
import secrets
import pdfplumber
import matplotlib.pyplot as plt

from io import BytesIO
from datetime import datetime, timedelta

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image
)

from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter

from functools import wraps

def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if 'user' not in session:
            return redirect("/login")
        return f(*args, **kwargs)
    return wrapper

# ================= APP =================
app = Flask(__name__)

app.secret_key = "super-secret-key"

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

app.config['SESSION_PERMANENT'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=2)

db = SQLAlchemy(app)

UPLOAD_FOLDER = "uploads"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ================= UNIVERSAL SKILLS =================
GLOBAL_SKILLS = [

    # FRONTEND
    "html", "css", "javascript", "typescript",
    "react", "next.js", "nextjs", "vue", "angular",
    "tailwind", "bootstrap", "redux",

    # BACKEND
    "node.js", "nodejs", "express", "express.js",
    "django", "flask", "fastapi", "spring boot",
    "laravel", ".net", "graphql", "rest api",

    # DATABASE
    "mysql", "postgresql", "mongodb", "firebase",
    "redis", "sqlite", "oracle", "supabase",

    # CLOUD / DEVOPS
    "aws", "azure", "gcp", "docker", "kubernetes",
    "jenkins", "terraform", "ci/cd", "nginx",

    # AI / ML
    "machine learning", "deep learning",
    "tensorflow", "pytorch", "langchain",
    "llm", "nlp", "computer vision",

    # PROGRAMMING
    "python", "java", "c", "c++", "c#",
    "go", "rust", "php", "ruby",
    "kotlin", "swift",

    # TOOLS
    "git", "github", "figma",
    "postman", "jira", "linux",
    "vercel", "netlify",

    # SOFT SKILLS
    "communication", "leadership",
    "teamwork", "problem solving",
    "analytical", "creative"
]


# ================= HELPERS =================
def extract_text(pdf_path):

    text = ""

    with pdfplumber.open(pdf_path) as pdf:

        for page in pdf.pages:

            t = page.extract_text()

            if t:
                text += t + " "

    return text.lower()


def normalize_text(text):

    cleaned = re.sub(
        r"[^a-z0-9+#.\s-]",
        " ",
        text.lower()
    )

    return re.sub(r"\s+", " ", cleaned).strip()


def extract_skills(text):

    found_skills = []

    text = text.lower()

    for skill in GLOBAL_SKILLS:

        pattern = r'\b' + re.escape(skill.lower()) + r'\b'

        if re.search(pattern, text):
            found_skills.append(skill)

    return list(set(found_skills))


# ================= AUTO SUGGESTIONS =================
def generate_auto_suggestions(missing, missing_keywords):

    suggestions = []

    if missing:

        suggestions.append(
            "Add missing skills like: " +
            ", ".join(missing[:5])
        )

    if missing_keywords:

        suggestions.append(
            "Add important keywords like: " +
            ", ".join(missing_keywords[:5])
        )

    suggestions.append(
        "Use strong action verbs like Developed, Built, Created"
    )

    suggestions.append(
        "Add measurable achievements with numbers"
    )

    suggestions.append(
        "Keep formatting simple and ATS-friendly"
    )

    return suggestions


# ================= ATS TIPS =================
def generate_ats_tips():

    return [

        "Use ATS-friendly fonts like Arial or Calibri",

        "Avoid tables, graphics, and images",

        "Use standard section headings",

        "Include relevant job keywords",

        "Keep resume formatting simple",

        "Use bullet points for achievements",

        "Save resume as PDF"
    ]


# ================= GRAPH =================
def create_score_graph(score):

    labels = ["Matched", "Missing"]

    values = [score, 100 - score]

    plt.figure(figsize=(4, 3))

    plt.bar(labels, values)

    plt.title("ATS Score Breakdown")

    graph_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        "score_graph.png"
    )

    plt.savefig(graph_path, bbox_inches='tight')

    plt.close()

    return graph_path


# ================= PDF REPORT =================
def generate_pdf_report(
    score,
    rating,
    suggestions,
    ats_tips
):

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter
    )

    styles = getSampleStyleSheet()

    story = []

    # TITLE
    story.append(
        Paragraph(
            "ATS Resume Analysis Report",
            styles['Title']
        )
    )

    story.append(Spacer(1, 20))

    # SCORE
    story.append(
        Paragraph(
            f"<b>ATS Score:</b> {score}%",
            styles['Heading2']
        )
    )

    story.append(
        Paragraph(
            f"<b>Rating:</b> {rating}",
            styles['BodyText']
        )
    )

    story.append(Spacer(1, 20))

    # GRAPH
    graph = create_score_graph(score)

    story.append(
        Paragraph(
            "Score Breakdown Graph",
            styles['Heading3']
        )
    )

    story.append(Spacer(1, 10))

    story.append(
        Image(graph, width=300, height=180)
    )

    story.append(Spacer(1, 20))

    # SUGGESTIONS
    story.append(
        Paragraph(
            "Suggestions",
            styles['Heading3']
        )
    )

    for s in suggestions:

        story.append(
            Paragraph(
                f"• {s}",
                styles['BodyText']
            )
        )

    story.append(Spacer(1, 20))

    # ATS TIPS
    story.append(
        Paragraph(
            "ATS Optimization Tips",
            styles['Heading3']
        )
    )

    for t in ats_tips:

        story.append(
            Paragraph(
                f"• {t}",
                styles['BodyText']
            )
        )

    story.append(Spacer(1, 20))

    story.append(
        Paragraph(
            "Generated by AI ATS Resume Analyzer",
            styles['Italic']
        )
    )

    doc.build(story)

    buffer.seek(0)

    return buffer


# ================= MODELS =================
class User(db.Model):

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    username = db.Column(
        db.String(100),
        unique=True
    )

    password = db.Column(
        db.String(200)
    )

    dark_mode = db.Column(
        db.Boolean,
        default=False
    )


class Resume(db.Model):

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    user_id = db.Column(db.Integer)

    name = db.Column(
        db.String(200)
    )

    best_score = db.Column(
        db.Integer,
        default=0
    )

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )


class AnalysisHistory(db.Model):

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    resume_id = db.Column(db.Integer)

    jd_title = db.Column(
        db.String(200)
    )

    match_score = db.Column(
        db.Integer
    )

    matched = db.Column(db.Text)

    missing = db.Column(db.Text)

    keywords = db.Column(db.Text)

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )


# ================= HOME =================
@app.route("/")
def home():

    if 'user' not in session:
        return redirect("/login")

    user = User.query.filter_by(
        username=session['user']
    ).first()

    if not user:
        session.clear()
        return redirect("/login")

    return render_template(
        "index.html",
        dark_mode=user.dark_mode
    )


# ================= DASHBOARD =================
@app.route("/dashboard")
def dashboard():

    if 'user' not in session:
        return redirect("/login")

    user = User.query.filter_by(
        username=session['user']
    ).first()

    if not user:
        session.clear()
        return redirect("/login")

    # USER RESUMES
    resumes = Resume.query.filter_by(
        user_id=user.id
    ).order_by(
        Resume.created_at.desc()
    ).all()

    # USER ANALYSES
    analyses = AnalysisHistory.query.join(
        Resume,
        Resume.id == AnalysisHistory.resume_id
    ).filter(
        Resume.user_id == user.id
    ).order_by(
        AnalysisHistory.created_at.desc()
    ).all()

    # ADD RATING TO ANALYSIS
    for analysis in analyses:

        if analysis.match_score >= 80:
            analysis.rating = "Excellent"

        elif analysis.match_score >= 60:
            analysis.rating = "Good"

        else:
            analysis.rating = "Needs Improvement"

    return render_template(
        "dashboard.html",
        user=user,
        resumes=resumes,
        analyses=analyses,
        dark_mode=user.dark_mode
    )


# ================= PROFILE =================
@app.route("/profile")
def profile():

    if 'user' not in session:
        return redirect("/login")

    user = User.query.filter_by(
        username=session['user']
    ).first()

    if not user:
        session.clear()
        return redirect("/login")

    # USER RESUMES
    resumes = Resume.query.filter_by(
        user_id=user.id
    ).all()

    # USER ANALYSES
    analyses = AnalysisHistory.query.join(
        Resume,
        Resume.id == AnalysisHistory.resume_id
    ).filter(
        Resume.user_id == user.id
    ).all()

    total_resumes = len(resumes)

    total_analyses = len(analyses)

    # BEST SCORE
    if resumes:
        best_score = max(
            r.best_score for r in resumes
        )
    else:
        best_score = 0

    # AVERAGE SCORE
    if analyses:

        average_score = int(
            sum(a.match_score for a in analyses)
            / len(analyses)
        )

    else:
        average_score = 0

    return render_template(
        "profile.html",
        user=user,
        total_resumes=total_resumes,
        total_analyses=total_analyses,
        best_score=best_score,
        average_score=average_score,
        dark_mode=user.dark_mode
    )


# ================= DELETE RESUME =================
@app.route("/delete-resume/<int:id>", methods=["POST"])
def delete_resume(id):

    if 'user' not in session:
        return redirect("/login")

    resume = Resume.query.get(id)

    if resume:

        # DELETE ANALYSIS HISTORY
        AnalysisHistory.query.filter_by(
            resume_id=resume.id
        ).delete()

        db.session.delete(resume)

        db.session.commit()

    return redirect("/dashboard")


# ================= DELETE ANALYSIS =================
@app.route("/delete-analysis/<int:id>", methods=["POST"])
def delete_analysis(id):

    if 'user' not in session:
        return redirect("/login")

    analysis = AnalysisHistory.query.get(id)

    if analysis:

        db.session.delete(analysis)

        db.session.commit()

    return redirect("/dashboard")
# ================= LOGIN =================
@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        user = User.query.filter_by(
            username=request.form["username"]
        ).first()

        if not user or not check_password_hash(
            user.password,
            request.form["password"]
        ):

            flash("Invalid credentials")

            return redirect("/login")

        session["user"] = user.username

        return redirect("/")

    return render_template("login.html")


# ================= REGISTER =================
@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        existing = User.query.filter_by(
            username=request.form["username"]
        ).first()

        if existing:

            flash("User already exists")

            return redirect("/register")

        user = User(
            username=request.form["username"],
            password=generate_password_hash(
                request.form["password"]
            )
        )

        db.session.add(user)

        db.session.commit()

        return redirect("/login")

    return render_template("register.html")





# ================= LOGOUT =================
@app.route("/logout")
def logout():

    session.clear()

    return redirect("/login")


@app.route("/settings", methods=["GET", "POST"])
@login_required
def settings():

    user = User.query.filter_by(username=session['user']).first()

    if request.method == "POST":

        # checkbox returns "on" if checked, None if not
        user.dark_mode = "dark_mode" in request.form

        db.session.commit()

        flash("Settings updated successfully")
        return redirect("/settings")

    return render_template(
        "settings.html",
        user=user
    )


# ================= ANALYZE =================
@app.route("/analyze", methods=["POST"])
def analyze():

    if 'user' not in session:
        return redirect("/login")

    file = request.files["resume"]

    jd = normalize_text(
        request.form["jd"]
    )

    jd_title = request.form.get(
        "jd_title",
        "Untitled Job"
    )

    user = User.query.filter_by(
        username=session['user']
    ).first()

    # SAVE FILE
    filename = secrets.token_hex(5) + "_" + file.filename

    path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    file.save(path)

    # EXTRACT TEXT
    resume_text = normalize_text(
        extract_text(path)
    )

    # TFIDF SCORE
    tfidf = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 2),
        stop_words='english'
    )

    try:

        vectors = tfidf.fit_transform([
            resume_text,
            jd
        ])

        tfidf_score = cosine_similarity(
            vectors[0:1],
            vectors[1:2]
        )[0][0] * 100

    except Exception:

        tfidf_score = 0

    # EXTRACT SKILLS
    resume_skills = extract_skills(
        resume_text
    )

    jd_skills = extract_skills(
        jd
    )

    matched = []

    missing = []

    for skill in jd_skills:

        if skill in resume_skills:
            matched.append(skill)

        else:
            missing.append(skill)

    # KEYWORDS
    resume_words = set(
        resume_text.split()
    )

    jd_words = set(
        jd.split()
    )

    missing_keywords = []

    for word in jd_words:

        if (
            len(word) > 4
            and word not in resume_words
            and word not in GLOBAL_SKILLS
        ):

            missing_keywords.append(word)

    missing_keywords = list(
        set(missing_keywords)
    )[:10]

    # ================= BETTER ATS SCORE =================

    # SKILL SCORE
    if len(jd_skills) > 0:

        skill_score = (
            len(matched) / len(jd_skills)
        ) * 100

    else:

        skill_score = 0

    # KEYWORD SCORE
    matched_keywords = len(jd_words) - len(missing_keywords)

    keyword_score = (
        matched_keywords / max(len(jd_words), 1)
    ) * 100

    keyword_score = max(
        0,
        min(100, keyword_score)
    )

    # FINAL SCORE
    score = int(round(

        (skill_score * 0.70) +

        (keyword_score * 0.15) +

        (tfidf_score * 0.15)

    ))

    # STRICT LIMITS
    if len(matched) <= 2:
        score = min(score, 35)

    if len(matched) == 0:
        score = min(score, 15)

    score = max(0, min(100, score))

    # ================= RATING =================
    if score >= 80:

        rating = "Excellent"

    elif score >= 60:

        rating = "Good"

    else:

        rating = "Needs Improvement"

    # ================= SUGGESTIONS =================
    suggestions = generate_auto_suggestions(
        missing,
        missing_keywords
    )

    ats_tips = generate_ats_tips()

    # SAVE RESUME
    resume = Resume(
        user_id=user.id,
        name=file.filename,
        best_score=score
    )

    db.session.add(resume)

    db.session.commit()

    # SAVE ANALYSIS
    analysis = AnalysisHistory(
        resume_id=resume.id,
        jd_title=jd_title,
        match_score=score,
        matched=",".join(matched),
        missing=",".join(missing),
        keywords=",".join(missing_keywords)
    )

    db.session.add(analysis)

    db.session.commit()

    return render_template(
        "result.html",

        score=score,

        rating=rating,

        matched=matched,

        missing=missing,

        missing_keywords=missing_keywords,

        suggestions=suggestions,

        ats_tips=ats_tips,

        analysis_id=str(analysis.id),

        dark_mode=user.dark_mode
    )


# ================= DOWNLOAD REPORT =================
@app.route("/download-report/<int:id>")
def download(id):

    a = AnalysisHistory.query.get(id)

    if not a:
        return "Analysis not found"

    rating = (
        "Excellent"
        if a.match_score >= 80
        else "Good"
        if a.match_score >= 60
        else "Needs Improvement"
    )

    suggestions = [
        "Add more relevant technical skills",
        "Improve keyword optimization",
        "Use stronger action verbs",
        "Add measurable achievements"
    ]

    ats_tips = generate_ats_tips()

    pdf = generate_pdf_report(
        a.match_score,
        rating,
        suggestions,
        ats_tips
    )

    return send_file(
        pdf,
        as_attachment=True,
        download_name="ATS_Report.pdf"
    )


# ================= RUN =================
if __name__ == "__main__":

    with app.app_context():
        db.create_all()

    app.run(debug=True)