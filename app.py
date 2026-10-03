"""Criminology + AI (Gemini) - single-file version (no frontend/backend imports needed)."""
import os, itertools, datetime as dt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Criminology + AI", page_icon="⚖️", layout="centered",
                   initial_sidebar_state="collapsed")

CSS = """
<style>
#MainMenu, footer {visibility: hidden;}
section[data-testid="stSidebar"], div[data-testid="stSidebarCollapsedControl"] {display: none;}
.block-container {padding-top: 1.2rem; padding-bottom: 3rem; max-width: 980px;}
html, body, [class*="css"] {font-family: "Inter", "Segoe UI", system-ui, sans-serif;}

/* hero */
.hero {background: linear-gradient(135deg,#0f172a 0%,#1e3a8a 55%,#7c3aed 100%);
  border-radius: 22px; padding: 28px 24px; color: #fff; margin-bottom: 18px;
  box-shadow: 0 12px 30px rgba(30,58,138,.35);}
.hero h1 {margin: 0 0 6px 0; font-size: 2rem; line-height: 1.15; color:#fff; padding:0;}
.hero p {margin: 0 0 14px 0; color: #dbe4ff; font-size: 1rem;}
.chip {display:inline-block; padding: 4px 12px; margin: 3px 6px 3px 0; border-radius: 999px;
  background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.25); font-size: .8rem; color:#fff;}

/* cards */
.card {border-radius: 18px; padding: 18px; margin-bottom: 14px;
  background: rgba(124,58,237,.07); border: 1px solid rgba(124,58,237,.25);}
.card h3 {margin: 0 0 8px 0; font-size: 1.1rem;}
.card ul {margin: 0; padding-left: 18px;}
.card.alt {background: rgba(30,64,175,.07); border-color: rgba(30,64,175,.28);}

/* page title */
.ptitle {font-size: 1.7rem; font-weight: 800; margin: 6px 0 14px 0;
  background: linear-gradient(90deg,#1e3a8a,#7c3aed); -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;}

/* top navigation (radio as pills) */
div[role="radiogroup"] {gap: 8px; flex-wrap: wrap;}
div[role="radiogroup"] > label {padding: 7px 14px; border-radius: 999px; cursor: pointer;
  border: 1px solid rgba(124,58,237,.35); background: rgba(124,58,237,.06); margin: 0 !important;}
div[role="radiogroup"] > label > div:first-child {display: none;}
div[role="radiogroup"] > label:has(input:checked) {
  background: linear-gradient(90deg,#1e3a8a,#7c3aed); border-color: transparent;}
div[role="radiogroup"] > label:has(input:checked) p {color:#fff !important; font-weight: 700;}
div[role="radiogroup"] > label p {font-size: .88rem; margin: 0;}

/* buttons, metrics, inputs */
.stButton > button, .stDownloadButton > button, div[data-testid="stFormSubmitButton"] > button {
  border-radius: 12px; border: 0; color: #fff; font-weight: 700; padding: .55rem 1.2rem;
  background: linear-gradient(90deg,#1e3a8a,#7c3aed); box-shadow: 0 6px 16px rgba(124,58,237,.3);}
.stButton > button:hover, .stDownloadButton > button:hover {filter: brightness(1.1); color:#fff;}
div[data-testid="stMetric"] {background: rgba(124,58,237,.07); border: 1px solid rgba(124,58,237,.25);
  padding: 14px 16px; border-radius: 16px;}
div[data-testid="stChatMessage"] {border-radius: 16px; background: rgba(124,58,237,.05);
  border: 1px solid rgba(124,58,237,.15);}
div[data-testid="stAlert"] {border-radius: 14px;}
.foot {text-align:center; opacity:.6; font-size:.8rem; margin-top: 30px;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

def page_title(text):
    st.markdown(f'<div class="ptitle">{text}</div>', unsafe_allow_html=True)


# ===================== BACKEND: knowledge =====================
SYSTEM = {
    "White-collar crime & rehabilitation": (
        "You are a criminology assistant on white-collar crime (corruption, bribery, embezzlement, scholarship/fund "
        "manipulation, abuse of authority, fraud). Explain theories (Sutherland, Cressey's Fraud Triangle, "
        "differential association, neutralization), prevention, whistleblowing, institutional controls and offender "
        "rehabilitation (restorative justice, CBT, ethics training, restitution, reintegration). Be concise and "
        "non-judgmental. Do not help anyone commit or conceal crimes. You are not a lawyer."),
    "Serial offender analysis": (
        "You are a criminology assistant on serial offending for students and analysts. Cover typologies and their "
        "critiques, linkage analysis (MO vs signature), geographic profiling (Rossmo, distance decay, buffer zone), "
        "investigative psychology and the limits of profiling. Do not glorify offenders or give guidance on "
        "committing crimes or evading detection. Be concise, note uncertainty. You are not a lawyer."),
}

KB = {
    "White-collar crime & rehabilitation": {
        "fraud triangle": "Fraud Triangle (Cressey): **Pressure + Opportunity + Rationalization**. Prevention targets all three.",
        "white collar": "White-collar crime (Sutherland, 1939): crime by respectable people in the course of their occupation.",
        "bribery": "Bribery: offering/accepting value to influence an official act. Controls: rotation, e-procurement, transparency, whistleblower protection.",
        "scholarship": "Scholarship manipulation: ghost beneficiaries, forged documents, favoritism. Controls: independent verification, public merit lists, audits.",
        "rehab": "Rehab: restitution, ethics training, CBT on rationalizations, restorative justice, supervised reintegration.",
        "whistle": "Whistleblowing works with anonymous channels, anti-retaliation rules and an independent receiver.",
    },
    "Serial offender analysis": {
        "mo": "MO is learned and evolves; signature expresses psychological needs and is more stable. Linkage favors rare, stable behaviors.",
        "organized": "Organized/disorganized typology is widely criticised as oversimplified; modern work is behavior-based.",
        "geographic": "Geographic profiling uses crime sites, distance decay and the buffer zone to prioritise search areas.",
        "linkage": "Case linkage compares behavior across crimes while accounting for base rates; beware linkage blindness across jurisdictions.",
        "profiling": "Profiling is an investigative aid with limited validated accuracy.",
    },
}

def offline(mode, q):
    q = q.lower()
    hits = [v for k, v in KB[mode].items() if k in q]
    return "\n\n".join(hits) or ("Offline mode (no Gemini key). Try: " + ", ".join(KB[mode]) + ".")


# ===================== BACKEND: Gemini =====================

def _get(secrets, name, default=""):
    try:
        v = secrets.get(name, "")
    except Exception:
        v = ""
    return v or os.getenv(name, default)

def ask(mode, history, secrets):
    """history: [{'role': 'user'|'assistant', 'content': str}]"""
    key = _get(secrets, "GEMINI_API_KEY")
    if not key:
        return offline(mode, history[-1]["content"])
    try:
        from google import genai
        from google.genai import types
        model = _get(secrets, "GEMINI_MODEL", "gemini-2.5-flash")
        contents = [types.Content(role="user" if m["role"] == "user" else "model",
                                  parts=[types.Part(text=m["content"])]) for m in history[-12:]]
        r = genai.Client(api_key=key).models.generate_content(
            model=model, contents=contents,
            config=types.GenerateContentConfig(system_instruction=SYSTEM[mode], max_output_tokens=900))
        return r.text or offline(mode, history[-1]["content"])
    except Exception as e:
        return f"Gemini unavailable ({type(e).__name__}). Fallback:\n\n" + offline(mode, history[-1]["content"])

# ===================== BACKEND: white-collar =====================
QUESTIONS = {
    "Pressure": ["Staff face unrealistic targets or financial stress.", "Pay/benefits are seen as unfair."],
    "Opportunity": ["One person controls approval, payment and records.", "Audits are rare or predictable.",
                    "There is no safe anonymous reporting channel."],
    "Rationalization": ["\"Everyone does it\" is commonly heard.", "Past misconduct went unpunished.",
                        "Leaders model bending rules."],
}
CONTEXTS = ["Bribery / corruption", "Scholarship / fund manipulation", "Abuse of authority", "Embezzlement / fraud"]

def level(total):
    return "Low" if total < 33 else "Moderate" if total < 66 else "High"

def rehab_plan(context, lvl):
    base = ["Restitution / corrective action where applicable", "Ethics & integrity workshop", "Clear written code of conduct"]
    extra = {"Low": ["Annual refresher training"],
             "Moderate": ["CBT-style sessions on rationalizations", "Dual approval for payments", "Quarterly audit"],
             "High": ["Structured counselling / restorative justice", "Remove sole control over funds",
                      "Monthly independent audit", "Supervised reintegration with mentor", "Anonymous whistleblower line"]}
    ctx = {"Bribery / corruption": ["Job rotation", "E-procurement & transparency portal"],
           "Scholarship / fund manipulation": ["Independent beneficiary verification", "Public merit lists"],
           "Abuse of authority": ["Oversight committee", "Clear grievance path for subordinates"],
           "Embezzlement / fraud": ["Segregation of duties", "Surprise cash/stock counts"]}
    return base + extra[lvl] + ctx[context]


# ===================== BACKEND: serial offender =====================

ANCHOR = (31.52, 74.35)  # fictional
FEATS = ["approach", "time", "site", "ligature", "trophy", "staging"]
WEIGHT = dict(approach=1, time=1, site=1, ligature=2, trophy=3, staging=3)

def make_cases():
    rng = np.random.default_rng(7)
    rows = []
    for i in range(6):
        d, a = rng.gamma(4, 0.012), rng.uniform(0, 2 * np.pi)
        lat, lon = np.array(ANCHOR) + d * np.array([np.sin(a), np.cos(a)])
        rows.append(dict(case=f"C{i+1}", lat=lat, lon=lon,
                         approach=rng.choice(["ruse", "ruse", "surprise"]), time=rng.choice(["night", "night", "dusk"]),
                         site=rng.choice(["isolated road", "isolated road", "vacant lot"]),
                         ligature=rng.choice(["yes", "yes", "no"]), trophy=rng.choice(["yes", "no"]),
                         staging=rng.choice(["yes", "no"])))
    rows.append(dict(case="X7", lat=31.80, lon=74.10, approach="surprise", time="day", site="residence",
                     ligature="no", trophy="no", staging="no"))  # unrelated decoy
    return pd.DataFrame(rows)

def similarity_matrix(df):
    ids = list(df.case)
    M = pd.DataFrame(np.eye(len(ids)), index=ids, columns=ids)
    tw = sum(WEIGHT.values())
    for a, b in itertools.combinations(range(len(df)), 2):
        s = sum(w for f, w in WEIGHT.items() if df.loc[a, f] == df.loc[b, f]) / tw
        M.iloc[a, b] = M.iloc[b, a] = round(s, 2)
    return M

def geo_grid(sub, buffer_km, decay, n=60):
    la, lo = np.meshgrid(np.linspace(sub.lat.min() - .08, sub.lat.max() + .08, n),
                         np.linspace(sub.lon.min() - .08, sub.lon.max() + .08, n))
    score = np.zeros_like(la)
    for _, r in sub.iterrows():
        d = np.maximum(np.hypot((la - r.lat) * 111, (lo - r.lon) * 95), 1e-3)
        score += np.where(d > buffer_km, 1 / d ** decay, 1 / np.maximum(2 * buffer_km - d, 1e-3) ** decay)
    i, j = np.unravel_index(np.argsort(score, axis=None)[-30:], score.shape)
    top = pd.DataFrame({"lat": la[i, j], "lon": lo[i, j]})
    centre = (top.lat.mean(), top.lon.mean())
    err = float(np.hypot((centre[0] - ANCHOR[0]) * 111, (centre[1] - ANCHOR[1]) * 95))
    return top, centre, err

STEPS = [
    ("Three similar night-time incidents are reported by different stations. First move?",
     {"Share data & request cross-jurisdiction linkage review": 2, "Treat each as an isolated case": -1, "Announce a 'serial offender' to media": -2}),
    ("Linkage shows two shared rare behaviors. How do you use a profile?",
     {"Use it as one investigative aid; keep an open mind": 2, "Fix on a suspect that fits the profile": -2, "Ignore it completely": 0}),
    ("Geographic analysis highlights a search zone. Next?",
     {"Prioritise tips, CCTV and records in the zone": 2, "Raid every house in the zone": -2, "Wait for another incident": -1}),
    ("A suspect emerges. How do you proceed?",
     {"Seek forensic corroboration before any charge": 2, "Charge on behavioral match alone": -2, "Release information to the press": -1}),
]

# ===================== FRONTEND =====================

def home():
    st.markdown("""
    <div class="hero">
      <h1>⚖️ Criminology + AI</h1>
      <p>Understand, prevent and investigate crime with Gemini-powered analysis.</p>
      <span class="chip">🤖 Gemini AI</span><span class="chip">🛡️ Prevention</span>
      <span class="chip">🌱 Rehabilitation</span><span class="chip">🕵️ Investigation</span>
    </div>
    <div class="card">
      <h3>🏛️ White-Collar Crime + Rehabilitation</h3>
      <ul><li>AI chatbot on corruption, bribery &amp; fraud theory</li>
      <li>Anonymous incident reporting dashboard</li>
      <li>Fraud-triangle risk assessment</li>
      <li>Personalised rehabilitation plan</li></ul>
    </div>
    <div class="card alt">
      <h3>🕵️ Serial Offender Analysis <small>(synthetic data)</small></h3>
      <ul><li>AI chatbot on typologies &amp; profiling limits</li>
      <li>Case linkage similarity matrix</li>
      <li>Geographic profiling map</li>
      <li>Investigation decision simulation</li></ul>
    </div>
    """, unsafe_allow_html=True)
    st.warning("Educational prototype. Profiling outputs are investigative aids, never evidence.")

def chat():
    page_title("💬 Criminology Chatbot")
    mode = st.selectbox("Topic", list(SYSTEM))
    key = f"chat::{mode}"
    st.session_state.setdefault(key, [])
    for m in st.session_state[key]:
        st.chat_message(m["role"]).write(m["content"])
    if p := st.chat_input("Ask a question..."):
        st.session_state[key].append({"role": "user", "content": p})
        st.chat_message("user").write(p)
        with st.spinner("Thinking..."):
            a = ask(mode, st.session_state[key], st.secrets)
        st.session_state[key].append({"role": "assistant", "content": a})
        st.chat_message("assistant").write(a)

def report():
    page_title("📝 Anonymous Incident Report")
    st.caption("Demo: stored in this session; download as CSV.")
    st.session_state.setdefault("reports", [])
    with st.form("rep"):
        cat = st.selectbox("Category", ["Bribery", "Embezzlement", "Scholarship manipulation", "Abuse of authority", "Fraud / forgery", "Other"])
        inst = st.text_input("Institution (optional)")
        sev = st.slider("Severity", 1, 5, 3)
        desc = st.text_area("What happened?")
        if st.form_submit_button("Submit") and desc.strip():
            st.session_state.reports.append(dict(time=dt.datetime.now().isoformat(timespec="seconds"),
                                                 category=cat, institution=inst, severity=sev, description=desc))
            st.success("Report logged.")
    if st.session_state.reports:
        df = pd.DataFrame(st.session_state.reports)
        st.dataframe(df, use_container_width=True)
        st.bar_chart(df["category"].value_counts())
        st.download_button("Download CSV", df.to_csv(index=False), "reports.csv")

def risk():
    page_title("📊 Fraud-Triangle Risk Assessment")
    st.caption("Rate each statement 0 (not true) to 4 (very true).")
    scores = {}
    for k, qs in QUESTIONS.items():
        st.subheader(k)
        scores[k] = sum(st.slider(q, 0, 4, 0, key=q) for q in qs) / (4 * len(qs)) * 100
    total = sum(scores.values()) / 3
    st.bar_chart(pd.Series(scores, name="Risk %"))
    lvl = level(total)
    st.metric("Overall risk", f"{total:.0f}%", lvl)
    st.session_state.level = lvl

def rehab():
    page_title("🌱 Rehabilitation & Prevention Plan")
    ctx = st.selectbox("Context", CONTEXTS)
    lv = ["Low", "Moderate", "High"]
    lvl = st.selectbox("Risk level", lv, index=lv.index(st.session_state.get("level", "Moderate")))
    plan = rehab_plan(ctx, lvl)
    for i, s in enumerate(plan, 1):
        st.write(f"{i}. {s}")
    st.download_button("Download plan", "\n".join(f"{i}. {s}" for i, s in enumerate(plan, 1)), "plan.txt")

@st.cache_data
def _cases():
    return make_cases()

def linkage():
    page_title("🔗 Case Linkage Analysis")
    cases = _cases()
    st.dataframe(cases[["case"] + FEATS], use_container_width=True)
    M = similarity_matrix(cases)
    st.dataframe(M.style.background_gradient(cmap="Blues", vmin=0, vmax=1), use_container_width=True)
    thr = st.slider("Linkage threshold", 0.3, 0.9, 0.6, 0.05)
    avg = ((M.sum() - 1) / (len(M) - 1)).round(2)
    st.bar_chart(avg)
    st.dataframe(pd.DataFrame({"avg similarity": avg, "likely series": avg >= thr}))
    st.info("Similarity ≠ proof. Confirm with forensic evidence.")

def geo():
    page_title("🗺️ Geographic Profiling (simplified)")
    cases = _cases()
    use = st.multiselect("Crime sites", list(cases.case), default=list(cases.case[:6]))
    B = st.slider("Buffer zone (km)", 0.5, 5.0, 2.0, 0.5)
    f = st.slider("Decay exponent", 0.5, 3.0, 1.5, 0.1)
    sub = cases[cases.case.isin(use)]
    if len(sub) < 3:
        return st.warning("Select at least 3 sites.")
    top, c, err = geo_grid(sub, B, f)
    st.write(f"Priority search area centre: **{c[0]:.3f}, {c[1]:.3f}**")
    pts = pd.concat([sub[["lat", "lon"]].assign(c="#d62728", s=120), top.assign(c="#1f77b4", s=30)])
    st.map(pts, latitude="lat", longitude="lon", color="c", size="s")
    st.caption("Red = crime sites, blue = top probability cells (fictional).")
    with st.expander("Validation: reveal fictional anchor"):
        st.write(f"Model centre is **{err:.1f} km** from the simulated anchor.")

def simulation():
    page_title("🎮 Case Alpha — Investigation Simulation")
    st.session_state.setdefault("step", 0)
    st.session_state.setdefault("score", 0)
    s, n = st.session_state.step, len(STEPS)
    if s < n:
        st.progress(s / n)
        q, opts = STEPS[s]
        ch = st.radio(q, list(opts), key=f"q{s}")
        if st.button("Confirm"):
            st.session_state.score += opts[ch]
            st.session_state.step += 1
            st.rerun()
    else:
        sc = st.session_state.score
        st.metric("Investigation quality", f"{sc} / {2 * n}")
        st.success("Evidence-led, bias-aware!" if sc >= 6 else "Avoid tunnel vision, profile-fitting and premature publicity.")
        if st.button("Restart"):
            st.session_state.step = st.session_state.score = 0
            st.rerun()

# ===================== NAVIGATION =====================
PAGES = {
    "🏠 Home": home,
    "💬 Chatbot": chat,
    "📝 Report": report,
    "📊 Risk": risk,
    "🌱 Rehab": rehab,
    "🔗 Linkage": linkage,
    "🗺️ Geo": geo,
    "🎮 Simulation": simulation,
}
choice = st.radio("Module", list(PAGES), horizontal=True, label_visibility="collapsed")
st.divider()
PAGES[choice]()
st.markdown('<div class="foot">Educational prototype · fictional data · not legal advice</div>', unsafe_allow_html=True)
