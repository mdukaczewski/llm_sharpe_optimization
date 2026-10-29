# Kept onml candidates (read by final.py).  fs = feature set name in final.SETS; members = targets x retrain x seeds,
# rank-normalised per day and averaged; k / kw = book construction (book2 in olib.py).
KEEP = {
    "onml_f28cal_q_k20":    dict(fs="f28cal", targets=["yr"], retrain=["Q"], seeds=[0], k=20,
                                 note="LGB rank target, FEATS18+EXTRA10+calendar(dow,dme,qend,dsgap), quarterly expanding retrain, universe(1500), L/S top20 eq-weight"),
    "onml_f28cal_q3_k20":   dict(fs="f28cal", targets=["yr"], retrain=["Q"], seeds=[0, 1, 2], k=20,
                                 note="same features, 3-seed averaged predictions (rank-avg), L/S top20 eq-weight"),
    "onml_f28cal_q3_k30p":  dict(fs="f28cal", targets=["yr"], retrain=["Q"], seeds=[0, 1, 2], k=30, kw=dict(weight="pred"),
                                 note="same 3-seed predictions, L/S top30 prediction-weighted"),
}
