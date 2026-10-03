import itertools
import numpy as np
import pandas as pd

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
