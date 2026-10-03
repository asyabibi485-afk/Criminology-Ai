import datetime as dt
import pandas as pd
import streamlit as st
from backend import ai, wcc, serial
from backend.knowledge import SYSTEM

def home():
    st.title("⚖️ Criminology + AI")
    st.write("Two concepts, one platform, powered by **Gemini**.")
    a, b = st.columns(2)
    a.info("**White-Collar Crime + Rehabilitation**\n\nChatbot · Reporting · Risk assessment · Rehab plan")
    b.info("**Serial Offender Analysis** (synthetic)\n\nChatbot · Case linkage · Geo-profiling · Simulation")
    st.warning("Educational prototype. Profiling outputs are investigative aids, never evidence.")

def chat():
    st.title("💬 Criminology Chatbot")
    mode = st.selectbox("Topic", list(SYSTEM))
    key = f"chat::{mode}"
    st.session_state.setdefault(key, [])
    for m in st.session_state[key]:
        st.chat_message(m["role"]).write(m["content"])
    if p := st.chat_input("Ask a question..."):
        st.session_state[key].append({"role": "user", "content": p})
        st.chat_message("user").write(p)
        with st.spinner("Thinking..."):
            a = ai.ask(mode, st.session_state[key], st.secrets)
        st.session_state[key].append({"role": "assistant", "content": a})
        st.chat_message("assistant").write(a)

def report():
    st.title("📝 Anonymous Incident Report")
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
    st.title("📊 Fraud-Triangle Risk Assessment")
    st.caption("Rate each statement 0 (not true) to 4 (very true).")
    scores = {}
    for k, qs in wcc.QUESTIONS.items():
        st.subheader(k)
        scores[k] = sum(st.slider(q, 0, 4, 0, key=q) for q in qs) / (4 * len(qs)) * 100
    total = sum(scores.values()) / 3
    st.bar_chart(pd.Series(scores, name="Risk %"))
    lvl = wcc.level(total)
    st.metric("Overall risk", f"{total:.0f}%", lvl)
    st.session_state.level = lvl

def rehab():
    st.title("🌱 Rehabilitation & Prevention Plan")
    ctx = st.selectbox("Context", wcc.CONTEXTS)
    lv = ["Low", "Moderate", "High"]
    lvl = st.selectbox("Risk level", lv, index=lv.index(st.session_state.get("level", "Moderate")))
    plan = wcc.rehab_plan(ctx, lvl)
    for i, s in enumerate(plan, 1):
        st.write(f"{i}. {s}")
    st.download_button("Download plan", "\n".join(f"{i}. {s}" for i, s in enumerate(plan, 1)), "plan.txt")

@st.cache_data
def _cases():
    return serial.make_cases()

def linkage():
    st.title("🔗 Case Linkage Analysis")
    cases = _cases()
    st.dataframe(cases[["case"] + serial.FEATS], use_container_width=True)
    M = serial.similarity_matrix(cases)
    st.dataframe(M.style.background_gradient(cmap="Blues", vmin=0, vmax=1), use_container_width=True)
    thr = st.slider("Linkage threshold", 0.3, 0.9, 0.6, 0.05)
    avg = ((M.sum() - 1) / (len(M) - 1)).round(2)
    st.bar_chart(avg)
    st.dataframe(pd.DataFrame({"avg similarity": avg, "likely series": avg >= thr}))
    st.info("Similarity ≠ proof. Confirm with forensic evidence.")

def geo():
    st.title("🗺️ Geographic Profiling (simplified)")
    cases = _cases()
    use = st.multiselect("Crime sites", list(cases.case), default=list(cases.case[:6]))
    B = st.slider("Buffer zone (km)", 0.5, 5.0, 2.0, 0.5)
    f = st.slider("Decay exponent", 0.5, 3.0, 1.5, 0.1)
    sub = cases[cases.case.isin(use)]
    if len(sub) < 3:
        return st.warning("Select at least 3 sites.")
    top, c, err = serial.geo_grid(sub, B, f)
    st.write(f"Priority search area centre: **{c[0]:.3f}, {c[1]:.3f}**")
    pts = pd.concat([sub[["lat", "lon"]].assign(c="#d62728", s=120), top.assign(c="#1f77b4", s=30)])
    st.map(pts, latitude="lat", longitude="lon", color="c", size="s")
    st.caption("Red = crime sites, blue = top probability cells (fictional).")
    with st.expander("Validation: reveal fictional anchor"):
        st.write(f"Model centre is **{err:.1f} km** from the simulated anchor.")

def simulation():
    st.title("🎮 Case Alpha — Investigation Simulation")
    st.session_state.setdefault("step", 0)
    st.session_state.setdefault("score", 0)
    s, n = st.session_state.step, len(serial.STEPS)
    if s < n:
        st.progress(s / n)
        q, opts = serial.STEPS[s]
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
