"""Build the event dataset: ATM-band entries (real strikes + placebo mid-strikes),
band jumps, landmarks, forward outcomes and pre-decision features.

Timing (see spec.py): bar label t_e = entry bar; decision tau = t_e + 1 min;
executable entry = open of bar tau; features use closes <= t_e and the latest gamma
snapshot with time <= tau. Outcome columns are prefixed  y_ ; everything prefixed
f_ is a pre-decision feature. tests/test_leakage.py re-computes features on data
truncated at tau and asserts equality.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from study import spec
from study.events import run_state_machine

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "watchlist"
OUT = ROOT / "results" / "cache"
NMIN = 390                        # 09:30 .. 15:59
SNAP_MIN = {m: m for m in range(30, 331, 30)}   # 10:00 .. 15:00 as minutes since open


def band(symbol, override=None):
    if override is not None:
        return override
    return spec.BAND.get(symbol, spec.BAND["default"])


def load():
    g = pd.read_parquet(RAW / "gamma_snapshots.parquet")
    p = pd.read_parquet(RAW / "price_bars_1min.parquet")
    for d in (g, p):
        d["symbol"] = d["symbol"].astype(str)
        d["date"] = d["time_et"].dt.normalize()
        d["minute"] = ((d["time_et"] - d["date"]).dt.total_seconds() // 60 - 570).astype(int)
    g["node_type"] = g["node_type"].astype(str)
    return g, p


def day_arrays(pday):
    arr = {k: np.full(NMIN, np.nan) for k in ("open", "high", "low", "close")}
    idx = pday["minute"].to_numpy()
    for k in arr:
        arr[k][idx] = pday[k].to_numpy()
    return arr


def last_valid(c, upto):
    """Index of last non-NaN close at or before `upto` (or -1)."""
    seg = c[: upto + 1]
    nz = np.flatnonzero(~np.isnan(seg))
    return nz[-1] if len(nz) else -1


# ----------------------------------------------------------------------------- features
def price_features(a, te, s, x_entry):
    """Pre-decision price-path features. Uses only bars with label <= te."""
    c, o, h, l = a["close"], a["open"], a["high"], a["low"]
    ce = c[te]
    f = {}
    for k in (1, 5, 15, 30):
        j = last_valid(c, max(te - k, 0))
        f[f"f_ret{k}"] = s * 100 * (ce / c[j] - 1) if j >= 0 and j < te else 0.0
    first_open = o[np.flatnonzero(~np.isnan(o))[0]]
    f["f_dayret"] = s * 100 * (ce / first_open - 1)
    seg = c[max(0, te - 30): te + 1]
    seg = seg[~np.isnan(seg)]
    f["f_rv30"] = float(np.std(np.diff(np.log(seg))) * 100) if len(seg) > 2 else np.nan
    f["f_range30"] = float((np.nanmax(h[max(0, te - 29): te + 1]) -
                            np.nanmin(l[max(0, te - 29): te + 1])) / ce * 100)
    f["f_speed5"] = abs(f["f_ret5"])
    f["f_accel"] = f["f_ret5"] - f["f_ret15"] / 3
    f["f_dist"] = s * x_entry          # <0: still on the approach side
    f["f_tod"] = te
    f["f_s"] = s
    return f


def snap_index(snaps, tau):
    """Latest snapshot minute <= tau (decision time)."""
    ok = [m for m in snaps if m <= tau]
    return max(ok) if ok else None


def gamma_features(snap, prev, K, S, s):
    strikes, gam, node = snap["strike"], snap["gamma"], snap["node"]
    gross = np.abs(gam).sum()
    net = gam.sum()
    f = {"f_net_ratio": net / gross if gross else 0.0,
         "f_log_gross": np.log10(gross) if gross else 0.0}
    at = np.isclose(strikes, K)
    gK = gam[at].sum()
    f["f_loc_share"] = gK / gross if gross else 0.0
    rel = 100 * (strikes - K) / K
    reg = np.abs(rel) <= spec.LOCAL_REGION_PCT
    rnet, rgross = gam[reg].sum(), np.abs(gam[reg]).sum()
    f["f_reg_net"] = rnet / gross if gross else 0.0
    f["f_reg_gross"] = rgross / gross if gross else 0.0
    f["f_reg_cancel"] = abs(rnet) / rgross if rgross else 0.0
    nodeK = node[at][0] if at.any() else "none"
    for nt in ("king", "gatekeeper", "pika", "barney"):
        f[f"f_node_{nt}"] = float(nodeK == nt)
    # --- M3 proxies: gamma profile in shock windows ahead/behind the approached strike
    srel = s * rel                      # >0 = beyond K in the approach direction
    for w in spec.SHOCKS:
        ah = (srel > 0) & (srel <= w + 1e-9)
        bh = (srel < 0) & (srel >= -w - 1e-9)
        f[f"f_ahead{w}"] = gam[ah].sum() / gross if gross else 0.0
        f[f"f_behind{w}"] = gam[bh].sum() / gross if gross else 0.0
        f[f"f_asym{w}"] = f[f"f_ahead{w}"] - f[f"f_behind{w}"]
    ki = np.flatnonzero(node == "king")
    if len(ki):
        kk = ki[0]
        f["f_king_dist"] = s * 100 * (strikes[kk] - S) / S
        f["f_king_sign"] = float(np.sign(gam[kk]))
        f["f_king_is_K"] = float(np.isclose(strikes[kk], K))
    else:
        f["f_king_dist"], f["f_king_sign"], f["f_king_is_K"] = 0.0, 0.0, 0.0
    if prev is not None:
        pg = prev["gamma"][np.isclose(prev["strike"], K)].sum()
        pgross = np.abs(prev["gamma"]).sum()
        f["f_dloc"] = gK / gross - (pg / pgross if pgross else 0.0) if gross else 0.0
        f["f_dnet_ratio"] = f["f_net_ratio"] - (prev["gamma"].sum() / pgross if pgross else 0.0)
        f["f_has_prev"] = 1.0
    else:
        f["f_dloc"], f["f_dnet_ratio"], f["f_has_prev"] = 0.0, 0.0, 0.0
    return f


# ----------------------------------------------------------------------------- outcomes
def outcomes(a, te, s, K, b):
    """Forward outcomes from the executable entry (open of bar te+1)."""
    c, o, h, l = a["close"], a["open"], a["high"], a["low"]
    m0 = te + 1
    y = {}
    if m0 >= NMIN or np.isnan(o[m0]):
        y["y_exec"] = 0.0
        return y
    p0 = o[m0]
    y["y_exec"] = 1.0
    y["y_p0"] = p0
    for hz in spec.HORIZONS:
        last = m0 + hz - 1
        if last >= NMIN:
            y[f"y_r{hz}"] = np.nan          # censored by session end -- never truncated
            continue
        j = last_valid(c, last)
        y[f"y_r{hz}"] = s * 100 * (c[j] / p0 - 1) if j >= m0 else np.nan
    # barrier, excursions, range, volatility over 15 minutes
    H = spec.PRIMARY_HORIZON
    if m0 + H - 1 < NMIN:
        hi, lo = h[m0:m0 + H], l[m0:m0 + H]
        up, dn = p0 * (1 + spec.BARRIER_PCT / 100), p0 * (1 - spec.BARRIER_PCT / 100)
        fav_hit = (hi >= up) if s > 0 else (lo <= dn)
        adv_hit = (lo <= dn) if s > 0 else (hi >= up)
        fi = np.flatnonzero(np.nan_to_num(fav_hit, nan=0)).tolist()
        ai = np.flatnonzero(np.nan_to_num(adv_hit, nan=0)).tolist()
        f1, a1 = (fi[0] if fi else 99), (ai[0] if ai else 99)
        if f1 == 99 and a1 == 99:
            y["y_barrier"] = "neither"
        elif f1 == a1:
            y["y_barrier"] = "ambiguous"
        else:
            y["y_barrier"] = "target" if f1 < a1 else "stop"
        y["y_t_target"] = f1 + 1 if f1 < 99 else np.nan
        y["y_t_stop"] = a1 + 1 if a1 < 99 else np.nan
        fav = (np.nanmax(hi) / p0 - 1) * 100 if s > 0 else (1 - np.nanmin(lo) / p0) * 100
        adv = (1 - np.nanmin(lo) / p0) * 100 if s > 0 else (np.nanmax(hi) / p0 - 1) * 100
        y["y_mfe15"], y["y_mae15"] = fav, adv
        y["y_range15"] = (np.nanmax(hi) - np.nanmin(lo)) / p0 * 100
        seg = c[te:m0 + H]
        seg = seg[~np.isnan(seg)]
        y["y_rv15"] = float(np.std(np.diff(np.log(seg))) * 100) if len(seg) > 2 else np.nan
    # strike crossing after entry (closes), 30-min window, censored at session end
    xs = s * 100 * (c[m0:min(m0 + 30, NMIN)] - K) / c[m0:min(m0 + 30, NMIN)]
    cr = np.flatnonzero(np.nan_to_num(xs, nan=-1) > 0)
    y["y_cross_t"] = cr[0] + 1 if len(cr) else np.nan
    y["y_cross_censored"] = float(m0 + 30 > NMIN and not len(cr))
    ex = np.flatnonzero(np.abs(np.nan_to_num(xs, nan=0)) > b)
    if len(ex):
        y["y_exit_t"] = ex[0] + 1
        y["y_exit_far"] = float(xs[ex[0]] > 0)
    else:
        y["y_exit_t"], y["y_exit_far"] = np.nan, np.nan
    return y


def paths(a, te, s):
    """Event-aligned path (descriptive): k=-10..30 relative to executable entry price.
    Pre-event points use closes of bars te-10..te; post points closes of m0..m0+29."""
    c, o = a["close"], a["open"]
    m0 = te + 1
    if m0 >= NMIN or np.isnan(o[m0]):
        return None
    p0 = o[m0]
    out = np.full(41, np.nan)
    for i, k in enumerate(range(-10, 31)):
        j = te + k if k <= 0 else m0 + k - 1
        if 0 <= j < NMIN:
            out[i] = s * 100 * (c[j] / p0 - 1)
    return out


# ----------------------------------------------------------------------------- driver
def placebo_levels(strikes):
    ks = np.sort(np.unique(strikes))
    mids = (ks[:-1] + ks[1:]) / 2
    return mids[~np.isin(mids, ks)]


def build_symbol(sym, g, p, b_override=None, want_paths=False, snap_lag=0, delay=0):
    """snap_lag: use the snapshot `snap_lag` steps older (stale-data test).
    delay: extra minutes between decision and executable entry (latency test)."""
    b = band(sym, b_override)
    gs, ps = g[g.symbol == sym], p[p.symbol == sym]
    rows = []
    for date, pday in ps.groupby("date"):
        a = day_arrays(pday)
        gd = gs[gs.date == date]
        snaps = {}
        for m, sd in gd.groupby("minute"):
            sd = sd.sort_values("strike")
            snaps[m] = {"strike": sd.strike.to_numpy(), "gamma": sd.net_gamma.to_numpy(float),
                        "node": sd.node_type.to_numpy(), "spot": sd.spot.iloc[0]}
        smins = sorted(snaps)
        real = np.sort(gd.strike.unique())
        for kind, levels in (("real", real), ("placebo", placebo_levels(real))):
            c = a["close"]
            x = 100 * (c[:, None] - levels[None, :]) / c[:, None]
            evs = run_state_machine(x, np.arange(NMIN), b, spec.ARM_MULT, spec.COOLDOWN_MIN)
            nprior = {}
            for te, k, typ, s in evs:
                K = levels[k]
                key = (k, typ)
                n = nprior.get(key, 0)
                nprior[key] = n + 1
                tau = te + 1
                r = {"symbol": sym, "date": date, "t_entry_bar": te, "tau": tau,
                     "K": K, "b": b, "level": kind, "event": typ, "s": s,
                     "n_prior": n, "x_entry": x[te, k]}
                si = snap_index(smins, tau)
                if si is not None and snap_lag:
                    i = smins.index(si) - snap_lag
                    si = smins[i] if i >= 0 else None
                r["snap_min"] = si
                # eligibility uses only strikes LISTED in the snapshot available at tau:
                # a real K must be listed; a placebo level needs both neighbours listed.
                if si is None:
                    ok = False
                elif kind == "real":
                    ok = bool(np.isclose(snaps[si]["strike"], K).any())
                else:
                    ok = bool(np.isclose(placebo_levels(snaps[si]["strike"]), K).any())
                r["eligible"] = float(ok)
                if typ == "entry":
                    f = price_features(a, te, s, x[te, k])
                    f["f_n_prior"] = n
                    sp = spacing(snaps[si]["strike"]) if si is not None else spacing(real)
                    f["f_round5"] = float(np.isclose((K / sp) % 5, 0))
                    f["f_round10"] = float(np.isclose((K / sp) % 10, 0))
                    if si is not None:
                        prev = smins[smins.index(si) - 1] if smins.index(si) > 0 else None
                        f.update(gamma_features(snaps[si], snaps.get(prev), K, c[te], s))
                        f["f_snap_age"] = tau - si
                    r.update(f)
                r.update(outcomes(a, te + delay, s, K, b))
                if want_paths and typ == "entry":
                    pth = paths(a, te, s)
                    if pth is not None:
                        r.update({f"p{k}": v for k, v in zip(range(-10, 31), pth)})
                rows.append(r)
    return pd.DataFrame(rows)


def spacing(strikes):
    d = np.diff(np.sort(strikes))
    return float(np.median(d)) if len(d) else 1.0


def add_ids(df):
    df = df.copy()
    et = (df["date"] + pd.to_timedelta(570 + df["tau"], unit="m")).dt.tz_localize("America/New_York")
    df["decision_time_utc"] = et.dt.tz_convert("UTC")
    df["decision_time_et"] = et
    df["event_id"] = (df.symbol + "|" + df.date.dt.strftime("%Y-%m-%d") + "|" +
                      df.K.map("{:g}".format) + "|" + df.t_entry_bar.astype(str) + "|" + df.level)
    # impulse clusters: same symbol/day/direction, entries within 5 minutes of the previous
    df = df.sort_values(["symbol", "level", "date", "s", "tau"])
    gap = df.groupby(["symbol", "level", "date", "s"]).tau.diff().fillna(999) > 5
    df["cluster_id"] = gap.cumsum()
    df["split"] = "holdout"
    for name, (lo, hi) in spec.SPLITS.items():
        df.loc[(df.date >= lo) & (df.date <= hi), "split"] = name
    return df.sort_values(["symbol", "date", "tau", "K"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", type=float, default=None, help="override band (sensitivity)")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--snap-lag", type=int, default=0)
    ap.add_argument("--delay", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    g, p = load()
    syms = args.symbols or sorted(p.symbol.unique())
    frames = []
    for sym in syms:
        df = build_symbol(sym, g, p, args.band,
                          want_paths=sym in (spec.PRIMARY, *spec.REPLICATIONS),
                          snap_lag=args.snap_lag, delay=args.delay)
        frames.append(df)
        print(sym, len(df), flush=True)
    ev = add_ids(pd.concat(frames, ignore_index=True))
    tag = ("" if args.band is None else f"_b{args.band}") + \
          (f"_lag{args.snap_lag}" if args.snap_lag else "") + (f"_delay{args.delay}" if args.delay else "")
    ev.to_parquet(OUT / f"events{tag}.parquet")
    print("events", len(ev))


if __name__ == "__main__":
    main()
