"""Frozen study specification. Written and committed BEFORE any outcome was computed.

Every choice below was made from schema/resolution inspection only (strike spacing,
1-minute move distribution, session coverage) -- never from forward returns.
"""

# --- instruments -----------------------------------------------------------
# NQ/ES futures are NOT in the dataset. The primary outcome is therefore measured on
# the source underlying itself (QQQ), with SPY and SPXW (SPX index price) as separate
# replications and the 47 single names as an exploratory pool. Nothing here is
# "NQ hedge demand".
PRIMARY = "QQQ"
REPLICATIONS = ["SPY", "SPXW"]
EXCLUDE_FROM_POOL = ["VIX"]  # VIX options settle on a different underlying (VIX futures)

# --- ATM band (percentage points of price) ------------------------------------
# Default b = 0.05. SPXW strikes are 5 pts apart (~0.065%), so a +/-0.05 band would
# overlap neighbouring strikes' bands; b = 0.025 (a predeclared sensitivity) is used
# as SPXW's primary. Chosen from strike spacing alone.
BAND = {"default": 0.05, "SPXW": 0.025}
BAND_SENSITIVITIES = [0.025, 0.05, 0.10]
ARM_MULT = 2.0            # arm after an observed |x| >= 2b exterior close
COOLDOWN_MIN = 5          # rearm only >= 5 minutes after the prior entry at that strike

# --- timing ------------------------------------------------------------------
# Bars are labelled by their start minute (09:30 ... 15:59, ET). A bar's close is
# observable at label + 1 min. Decision time tau = entry-bar label + 1 min.
# Executable entry = OPEN of the bar labelled tau (first trade after the decision).
# Gamma snapshot used = latest snapshot with time_et <= tau (README: snapshot is the
# latest at or before its listed time, median lag 1 s).
SESSION_OPEN = "09:30"
SESSION_LAST_BAR = "15:59"
FIRST_SNAPSHOT = "10:00"
HORIZONS = [1, 5, 15, 30]
PRIMARY_HORIZON = 15
PRE_WINDOW = 10           # descriptive only
BARRIER_PCT = 0.10        # symmetric +/- barrier from executable entry, 15-min deadline

# --- splits (chronological, whole trading days; label horizons are intraday so no
# purge gap is needed across day boundaries) ---------------------------------
SPLITS = {
    "dev": ("2026-07-01", "2026-07-31"),
    "val": ("2026-08-01", "2026-08-31"),
    "holdout": ("2026-09-01", "2026-09-30"),
}

# --- features ---------------------------------------------------------------
LOCAL_REGION_PCT = 0.25   # "nearby region" around the approached strike
SHOCKS = [0.05, 0.10, 0.20]   # percent; 0.10 primary (proxy windows, see features.py)

# --- decision rules frozen before the holdout ----------------------------------
MIN_MEANINGFUL_SKILL = 0.005      # 1 - MSE(M3)/MSE(M2) must be >= 0.5% AND CI > 0
ROUND_TRIP_COST_PCT = {"QQQ": 0.010, "SPY": 0.010, "SPXW": 0.010, "default": 0.050}
MIN_MEANINGFUL_TRADE_PCT = 0.010  # >= 1 bp/trade net improvement vs price-only rule
BOOTSTRAP_REPS = 2000
SEED = 20261001
