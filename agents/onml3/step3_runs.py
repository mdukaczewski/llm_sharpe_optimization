"""run table shared by step3 / step4: key -> (desc, features, target)"""
from lib3 import F26X, LIQF
RUNS = {"R1": ("f26x_oo", F26X, "yr_oo"), "R2": ("f40_oo", F26X + LIQF, "yr_oo"), "R3": ("f40_oon", F26X + LIQF, "yr_oon"),
        "R4": ("f40_on", F26X + LIQF, "yr"), "R5": ("f26x_on", F26X, "yr")}
