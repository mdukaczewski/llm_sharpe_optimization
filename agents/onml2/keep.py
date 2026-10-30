# Kept onml2 candidates (read by final.py).  MODELS: name -> wf2 kwargs (LightGBM params / window / recency; feature set
# defaults to F26CAL = f28cal minus zmoc/vmoc_share).  KEEP: candidate -> models x seeds rank-averaged, retrain 'Y'/'Q',
# book = book3 kwargs (L/S top-k equal weight), note.  Saved series are NET of spread, borrow and the $0.4M commission.
MODELS = {
    "base":    dict(),                                         # 300 trees, lr 0.03, 15 leaves, min_child_samples 200, colsample 0.8
    "mcs1000": dict(params=dict(min_child_samples=1000)),
    "cs05":    dict(params=dict(colsample_bytree=0.5)),
}
KEEP = {
    "onml2_f26cal_y_k10":  dict(models=["base"], seeds=[0], retrain="Y", book=dict(k=10),
                                note="LGB rank target, f26cal (f28cal minus zmoc/vmoc_share), yearly expanding retrain, universe(2000), L/S top10 eq-weight, net of $3/order commission on $0.4M"),
    "onml2_f26cal_y3_k10": dict(models=["base", "mcs1000", "cs05"], seeds=[0], retrain="Y", book=dict(k=10),
                                note="rank-average of 3 model variants (base, min_child_samples 1000, colsample 0.5), otherwise as onml2_f26cal_y_k10"),
    "onml2_f26cal_y_k20":  dict(models=["base"], seeds=[0], retrain="Y", book=dict(k=20),
                                note="same model as onml2_f26cal_y_k10, L/S top20 eq-weight (the baseline construction) for reference"),
}
