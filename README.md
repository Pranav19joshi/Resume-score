# ResumeAI — Smart ATS Analyzer

ResumeAI is an intelligent, Streamlit-based web application designed to evaluate and optimize resumes against specific job descriptions. Powered by the `gemini-2.0-flash` model, it calculates realistic ATS compatibility scores, extracts structural summaries, and detects profile strengths. 

The app blends deep AI insights with local, privacy-friendly NLP metrics—like TF-IDF keyword heuristics and regex contact parsing—to give users immediate feedback.

### Key Features
* **AI Match & Analysis:** Pinpoints critical skills gaps and maps targeted learning plans.
* **Local NLP Insights:** Computes standalone mathematical text similarities instantly.
* **Actionable Roadmap:** Outlines a 4-week structured skill-building timeline.

### Quick Start
1. Configure your `GOOGLE_API_KEY` in a local `.env` file.
2. Launch the dashboard:  
   `streamlit run app.py`
