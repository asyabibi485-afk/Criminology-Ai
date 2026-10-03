import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from frontend import views

st.set_page_config(page_title="Criminology + AI", page_icon="⚖️", layout="wide")

PAGES = {
    "🏠 Home": views.home,
    "💬 Chatbot": views.chat,
    "— White-collar —": None,
    "📝 Report": views.report,
    "📊 Risk Assessment": views.risk,
    "🌱 Rehab Plan": views.rehab,
    "— Serial offender —": None,
    "🔗 Case Linkage": views.linkage,
    "🗺️ Geo Profiling": views.geo,
    "🎮 Simulation": views.simulation,
}
choice = st.sidebar.radio("Module", [k for k in PAGES if PAGES[k]])
st.sidebar.caption("Educational prototype · fictional data · not legal advice")
PAGES[choice]()
