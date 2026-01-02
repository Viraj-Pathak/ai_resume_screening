import os
from flask import Flask, render_template, request
from model_utils import ResumeMatcher

app = Flask(__name__)
matcher = ResumeMatcher()

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    strengths = []
    gaps = []
    resume_text = ""
    jd_text = ""

    if request.method == "POST":
        jd_text = request.form.get("job_description", "").strip()
        resume_text = request.form.get("resume_text", "").strip()

        resume_file = request.files.get("resume_file")
        if resume_file and resume_file.filename:
            try:
                uploaded_text = resume_file.read().decode("utf-8", errors="ignore")
                resume_text = uploaded_text
            except Exception:
                pass

        if jd_text and resume_text:
            result = round(matcher.similarity(resume_text, jd_text), 2)
            strengths, gaps = matcher.compare_bullets(resume_text, jd_text)

    return render_template("index.html",
        result=result,
        strengths=strengths,
        gaps=gaps,
        resume_text=resume_text,
        jd_text=jd_text,
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
