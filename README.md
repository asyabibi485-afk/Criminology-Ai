# Criminology + AI (Gemini)

Two hackathon concepts in one Streamlit app:
- **White-Collar Crime + Rehabilitation**: chatbot, anonymous reporting, fraud-triangle risk assessment, rehab plan
- **Serial Offender Analysis** (synthetic data): chatbot, case linkage, geographic profiling, investigation simulation

## Structure
    app.py                 # entry point (navigation)
    frontend/views.py      # all Streamlit UI
    backend/ai.py          # Gemini client + offline fallback
    backend/knowledge.py   # system prompts + offline knowledge base
    backend/wcc.py         # risk scoring, rehab plans
    backend/serial.py      # synthetic cases, linkage, geo-profiling, simulation

## Run locally
    pip install -r requirements.txt
    mkdir -p .streamlit && cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # add your key
    streamlit run app.py

## Deploy (Streamlit Community Cloud)
1. Push to GitHub (secrets.toml is git-ignored).
2. share.streamlit.io -> New app -> repo, branch, main file `app.py`.
3. App settings -> Secrets -> `GEMINI_API_KEY = "..."`.

> Educational prototype with fictional data. Not legal advice; profiling outputs are not evidence.
