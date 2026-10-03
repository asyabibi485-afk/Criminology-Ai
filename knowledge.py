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
