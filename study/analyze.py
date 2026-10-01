"""Run the primary and secondary analyses on the cached event datasets.

Writes machine-readable results to results/tables/*.csv and results/results.json.
All uncertainty is a trading-day block bootstrap (days resampled with replacement),
so repeated events, nearby strikes and impulse clusters on one day are never treated
as independent evidence.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from study import build, spec

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "results" / "cache"
TAB = ROOT / "results" / "tables"
RNG = np.random.default_rng(spec.SEED)
B = spec.BOOTSTRAP_REPS
RES = {}


# ----------------------------------------------------------------------------- bootstrap
def _day_weights(days):
    n = len(days)
    return RNG.multinomial(n, np.full(n, 1 / n), size=B)          # (B, n_days)


def boot_mean(df, col):
    d = df.dropna(subset=[col])
    if d.empty:
        return dict(mean=np.nan, lo=np.nan, hi=np.nan, n=0, days=0)
    agg = d.groupby("date")[col].agg(["sum", "count"])
    w = _day_weights(agg.index)
    bs = (w @ agg["sum"].to_numpy()) / np.maximum(w @ agg["count"].to_numpy(), 1)
    return dict(mean=d[col].mean(), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5),
                n=len(d), days=d.date.nunique())


def boot_diff(a, b, col):
    """mean(a) - mean(b), days resampled jointly (paired by trading day)."""
    a, b = a.dropna(subset=[col]), b.dropna(subset=[col])
    days = sorted(set(a.date) | set(b.date))
    A = a.groupby("date")[col].agg(["sum", "count"]).reindex(days, fill_value=0)
    Bb = b.groupby("date")[col].agg(["sum", "count"]).reindex(days, fill_value=0)
    w = _day_weights(days)
    bs = (w @ A["sum"].to_numpy()) / np.maximum(w @ A["count"].to_numpy(), 1) - \
         (w @ Bb["sum"].to_numpy()) / np.maximum(w @ Bb["count"].to_numpy(), 1)
    est = a[col].mean() - b[col].mean()
    p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    return dict(diff=est, lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5),
                p_boot=max(p, 1 / B), n_a=len(a), n_b=len(b), days=len(days))


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


# ----------------------------------------------------------------------------- data
def entries(path="events.parquet", symbols=None):
    e = pd.read_parquet(CACHE / path)
    e = e[(e.event == "entry") & (e.eligible == 1) & (e.y_exec == 1)]
    if symbols is not None:
        e = e[e.symbol.isin(symbols)]
    e = e.copy()
    e["y_cont"] = (e.y_r15 > 0).astype(float).where(e.y_r15.notna())
    e["y_abs15"] = e.y_r15.abs()
    e["y_target"] = (e.y_barrier == "target").astype(float).where(e.y_barrier.notna())
    e["y_stop"] = (e.y_barrier == "stop").astype(float).where(e.y_barrier.notna())
    e["y_ambig"] = (e.y_barrier == "ambiguous").astype(float).where(e.y_barrier.notna())
    pre = e.f_dist < 0                                          # still on approach side
    e["y_cross15"] = ((e.y_cross_t <= 15).astype(float)).where(pre & (e.tau + 15 <= build.NMIN))
    e["y_cross30"] = ((e.y_cross_t <= 30).astype(float)).where(pre & (e.tau + 30 <= build.NMIN))
    e["f_tod2"] = (e.f_tod - 195) ** 2 / 1e4
    return e


# ----------------------------------------------------------------------------- A. descriptive
OUTS = ["y_r1", "y_r5", "y_r15", "y_r30", "y_cont", "y_target", "y_stop", "y_ambig",
        "y_mfe15", "y_mae15", "y_range15", "y_rv15", "y_cross15", "y_cross30", "y_exit_far"]


def descriptive(e, sym):
    rows = []
    real, plac = e[e.level == "real"], e[e.level == "placebo"]
    for col in OUTS:
        r, p, d = boot_mean(real, col), boot_mean(plac, col), boot_diff(real, plac, col)
        rows.append(dict(symbol=sym, outcome=col, real=r["mean"], real_lo=r["lo"], real_hi=r["hi"],
                         placebo=p["mean"], placebo_lo=p["lo"], placebo_hi=p["hi"],
                         diff=d["diff"], diff_lo=d["lo"], diff_hi=d["hi"], p_boot=d["p_boot"],
                         n_real=r["n"], n_placebo=p["n"], days=d["days"]))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- C. models
M0 = ["f_s", "f_dist", "f_ret1", "f_ret5", "f_ret15", "f_ret30", "f_dayret", "f_rv30",
      "f_range30", "f_speed5", "f_accel", "f_tod", "f_tod2", "f_n_prior", "f_round5", "f_round10"]
M1 = M0 + ["f_net_ratio", "f_log_gross", "f_snap_age", "f_dnet_ratio"]
M2 = M1 + ["f_loc_share", "f_reg_net", "f_reg_gross", "f_reg_cancel", "f_node_king",
           "f_node_gatekeeper", "f_node_pika", "f_node_barney", "f_dloc",
           "i_loc_dist", "i_reg_dist", "i_loc_net", "i_reg_rv"]
M3 = M2 + ["f_ahead0.1", "f_behind0.1", "f_asym0.1", "f_ahead0.2", "f_behind0.2", "f_asym0.2",
           "f_king_dist", "f_king_sign", "f_king_is_K", "i_king", "i_asym_dist"]
MODELS = {"M0": M0, "M1": M1, "M2": M2, "M3": M3}
ALPHAS = [0.1, 1, 10, 100, 1000, 1e4, 1e5]


def add_interactions(e):
    e = e.copy()
    e["i_loc_dist"] = e.f_loc_share * e.f_dist
    e["i_reg_dist"] = e.f_reg_net * e.f_dist
    e["i_loc_net"] = e.f_loc_share * e.f_net_ratio
    e["i_reg_rv"] = e.f_reg_net * e.f_rv30
    e["i_king"] = e.f_king_dist * e.f_king_sign
    e["i_asym_dist"] = e["f_asym0.2"] * e.f_dist
    # pooled single names: centre log-gross per symbol on DEVELOPMENT data only
    dev_mean = e[e.split == "dev"].groupby("symbol").f_log_gross.mean()
    e["f_log_gross"] = e.f_log_gross - e.symbol.map(dev_mean).fillna(e.f_log_gross.mean())
    return e


def fit_predict(kind, cols, tr, te, alpha=None):
    if kind == "ridge":
        mdl = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    else:   # identical capacity for every nested model
        mdl = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=150,
                                            min_samples_leaf=60, l2_regularization=1.0,
                                            random_state=spec.SEED)
    mdl.fit(tr[cols].to_numpy(), tr["y"].to_numpy())
    return mdl.predict(te[cols].to_numpy())


def evaluate_models(e, target="y_r15", kind="ridge", tag=""):
    e = add_interactions(e[e.level == "real"]).dropna(subset=[target]).copy()
    e["y"] = e[target]
    feats = sorted(set(sum(MODELS.values(), [])))
    e[feats] = e[feats].fillna(0.0)
    dev, val, hold = (e[e.split == s] for s in ("dev", "val", "holdout"))
    preds = {"val": {}, "holdout": {}, "val_refit": {}}
    chosen = {}
    for name, cols in MODELS.items():
        if kind == "ridge":
            mse = {a: np.mean((val.y - fit_predict(kind, cols, dev, val, a)) ** 2) for a in ALPHAS}
            a = min(mse, key=mse.get)
        else:
            a = None
        chosen[name] = a
        preds["val"][name] = fit_predict(kind, cols, dev, val, a)
        both = pd.concat([dev, val])
        preds["holdout"][name] = fit_predict(kind, cols, both, hold, a)
        preds["val_refit"][name] = fit_predict(kind, cols, both, val, a)   # threshold only
    train_mean = {"val": dev.y.mean(), "holdout": pd.concat([dev, val]).y.mean()}
    out = []
    for part, df in (("val", val), ("holdout", hold)):
        days = sorted(df.date.unique())
        w = _day_weights(days)
        sse = {}
        for name in MODELS:
            err = (df.y.to_numpy() - preds[part][name]) ** 2
            sse[name] = pd.Series(err, index=df.index).groupby(df.date).sum().reindex(days).to_numpy()
        # null forecast = training-period mean (zero is not a sensible null for |r| or vol)
        zero = ((df.y - train_mean[part]) ** 2).groupby(df.date).sum().reindex(days).to_numpy()
        for name in MODELS:
            base = {"M0": zero, "M1": sse["M0"], "M2": sse["M1"], "M3": sse["M2"]}[name]
            bname = {"M0": "train_mean", "M1": "M0", "M2": "M1", "M3": "M2"}[name]
            skill = 1 - sse[name].sum() / base.sum()
            bs = 1 - (w @ sse[name]) / (w @ base)
            vs_zero = 1 - sse[name].sum() / zero.sum()
            bz = 1 - (w @ sse[name]) / (w @ zero)
            mae = np.mean(np.abs(df.y.to_numpy() - preds[part][name]))
            out.append(dict(target=target, kind=kind, part=part, model=name, baseline=bname,
                            skill_vs_prev=skill, lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5),
                            skill_vs_mean=vs_zero, mean_lo=np.percentile(bz, 2.5),
                            mean_hi=np.percentile(bz, 97.5), mae=mae, alpha=chosen[name],
                            n=len(df), days=len(days)))
        # the primary contrast M3 vs M2 (identical rows)
    res = pd.DataFrame(out)
    res["tag"] = tag
    return res, preds, (dev, val, hold)


def trading(preds, val, hold, sym, cost):
    """Rule: trade in the forecast direction when |forecast| exceeds the 70th percentile
    of the final (dev+val) model's |forecasts| on the validation month -- frozen before
    the holdout. M3 rule vs the price-only M0 rule."""
    rows = []
    for name in ("M0", "M3"):
        thr = np.percentile(np.abs(preds["val_refit"][name]), 70)
        f = preds["holdout"][name]
        take = np.abs(f) > thr
        side = np.sign(f[take])
        pnl = side * hold.y.to_numpy()[take] - cost
        d = hold[take].assign(pnl=pnl, price_side=side * hold.s.to_numpy()[take])
        r = boot_mean(d, "pnl")
        rows.append(dict(symbol=sym, model=name, trades=int(take.sum()), days=d.date.nunique(),
                         net_mean=r["mean"], lo=r["lo"], hi=r["hi"],
                         long_mean=d[d.price_side > 0].pnl.mean(), n_long=int((d.price_side > 0).sum()),
                         short_mean=d[d.price_side < 0].pnl.mean(), n_short=int((d.price_side < 0).sum()),
                         cost=cost, threshold=thr))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- E. H1 landmarks
def landmark_panel(e_sym, sym):
    """Minute-by-minute panel inside each entry episode (decisions at entry + j, j=0..15).
    For decision j the close of bar te+j is observed; outcome = s-adjusted 15-min return
    from the open of bar te+j+1. Retains episodes that never cross."""
    g, p = build.load()
    p = p[p.symbol == sym]
    arrs = {d: build.day_arrays(pd_) for d, pd_ in p.groupby("date")}
    rows = []
    for _, r in e_sym.iterrows():
        a = arrs[r.date]
        c, o = a["close"], a["open"]
        crossed = False
        exited = False
        for j in range(0, 16):
            t = r.t_entry_bar + j
            if t + 15 >= build.NMIN or np.isnan(c[t]) or np.isnan(o[t + 1]):
                break
            x = r.s * 100 * (c[t] - r.K) / c[t]
            first_cross = (not crossed) and x > 0
            crossed = crossed or x > 0
            first_exit = (not exited) and abs(x) > r.b
            exited = exited or abs(x) > r.b
            j15 = build.last_valid(c, t + 15)
            y = r.s * 100 * (c[j15] / o[t + 1] - 1)
            rows.append(dict(date=r.date, level=r.level, event_id=r.event_id, j=j, x=x,
                             crossed=float(crossed), first_cross=float(first_cross),
                             first_exit=float(first_exit), exit_far=float(first_exit and x > 0),
                             exit_back=float(first_exit and x < 0), y=y, b=r.b))
    return pd.DataFrame(rows)


def crossing_effect(panel, b):
    """Within-episode regression: y ~ crossed + distance bins + minutes-since-entry.
    Day-block bootstrap of the 'crossed' coefficient, real and placebo separately."""
    out = {}
    for lvl in ("real", "placebo"):
        d = panel[panel.level == lvl].copy()
        d["dbin"] = pd.cut(d.x, np.linspace(-4 * b, 4 * b, 17), labels=False).fillna(-1)
        X = pd.get_dummies(d[["dbin", "j"]].astype(int).astype(str), drop_first=True).astype(float)
        X["crossed"] = d.crossed
        X["const"] = 1.0
        Xv, yv = X.to_numpy(), d.y.to_numpy()
        days = d.date.to_numpy()
        ud = np.unique(days)
        idx = {u: np.flatnonzero(days == u) for u in ud}
        beta = np.linalg.lstsq(Xv, yv, rcond=None)[0]
        ci = X.columns.get_loc("crossed")
        bs = []
        for _ in range(400):
            pick = np.concatenate([idx[u] for u in RNG.choice(ud, len(ud))])
            bs.append(np.linalg.lstsq(Xv[pick], yv[pick], rcond=None)[0][ci])
        out[lvl] = dict(coef=beta[ci], lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5),
                        n=len(d), episodes=d.event_id.nunique(), days=len(ud))
    return out


# ----------------------------------------------------------------------------- I. distance curve
def distance_curve(sym, max_x=0.35):
    """All-minute panel: for each decision minute (10:00..15:44) and each threshold
    (real listed strike from the latest snapshot, or placebo mid-strike) within max_x,
    y = 15-min return measured TOWARD the threshold (positive = moved toward/through)."""
    g, p = build.load()
    g, p = g[g.symbol == sym], p[p.symbol == sym]
    rows = []
    for date, pday in p.groupby("date"):
        a = build.day_arrays(pday)
        c, o = a["close"], a["open"]
        gd = g[g.date == date]
        snaps = {m: np.sort(sd.strike.to_numpy()) for m, sd in gd.groupby("minute")}
        sm = sorted(snaps)
        for t in range(29, build.NMIN - 16, 1):       # decision tau = t+1 in [30, 374]
            tau = t + 1
            si = build.snap_index(sm, tau)
            if si is None or np.isnan(c[t]) or np.isnan(o[tau]):
                continue
            ks = snaps[si]
            j15 = build.last_valid(c, tau + 14)
            ret = 100 * (c[j15] / o[tau] - 1)
            for lvl, levels in (("real", ks), ("placebo", build.placebo_levels(ks))):
                x = 100 * (c[t] - levels) / c[t]
                m = np.abs(x) <= max_x
                for xi in x[m]:
                    rows.append((date, lvl, abs(xi), -np.sign(xi) * ret if xi != 0 else np.nan))
    d = pd.DataFrame(rows, columns=["date", "level", "absx", "y"]).dropna()
    edges = np.arange(0, max_x + 1e-9, 0.025)
    d["bin"] = pd.cut(d.absx, edges, include_lowest=True)
    out = []
    for (lvl, bn), sub in d.groupby(["level", "bin"], observed=True):
        r = boot_mean(sub, "y")
        out.append(dict(symbol=sym, level=lvl, x_mid=bn.mid, mean=r["mean"], lo=r["lo"],
                        hi=r["hi"], n=r["n"], days=r["days"]))
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- group tests
def group_table(e, sym, defs, col="y_r15"):
    rows = []
    for fam, name, mask_a, mask_b in defs:
        a, b_ = e[mask_a], e[mask_b]
        d = boot_diff(a, b_, col)
        rows.append(dict(symbol=sym, family=fam, contrast=name, outcome=col, **d,
                         mean_a=a[col].mean(), mean_b=b_[col].mean()))
    return pd.DataFrame(rows)


def secondary_groups(e, sym):
    r = e[e.level == "real"]
    pre = r.f_dist < 0
    defs = [
        ("H3", "local&aggregate agree vs disagree", np.sign(r.f_reg_net) == np.sign(r.f_net_ratio),
         np.sign(r.f_reg_net) != np.sign(r.f_net_ratio)),
        ("H3", "near-zero total: low gross vs high gross",
         (r.f_net_ratio.abs() < 0.15) & (r.f_log_gross < r.f_log_gross.median()),
         (r.f_net_ratio.abs() < 0.15) & (r.f_log_gross >= r.f_log_gross.median())),
        ("H3", "aggregate net>0 vs net<0", r.f_net_ratio > 0, r.f_net_ratio < 0),
        ("H2", "local strike gamma >0 vs <0", r.f_loc_share > 0, r.f_loc_share < 0),
        ("H2", "approached strike is king vs not", r.f_node_king == 1, r.f_node_king == 0),
        ("H2", "upward vs downward approach", r.f_s > 0, r.f_s < 0),
        ("H5", "first encounter vs repeat", r.f_n_prior == 0, r.f_n_prior > 0),
        ("H5", "high cancellation region vs concentrated", r.f_reg_cancel < 0.5, r.f_reg_cancel >= 0.5),
        ("H2", "0DTE clock: last 90 min vs earlier", r.f_tod >= 300, r.f_tod < 300),
    ]
    out = []
    for col in ("y_r15", "y_rv15"):
        t = group_table(r, sym, defs, col)
        out.append(t)
    out = pd.concat(out)
    out["p_holm_family"] = np.nan
    for (fam, col), idx in out.groupby(["family", "outcome"]).groups.items():
        out.loc[idx, "p_holm_family"] = holm(out.loc[idx, "p_boot"])
    return out


def regime_check(e, sym):
    """Sign-convention check: does vendor net gamma > 0 coincide with lower subsequent
    realised volatility (the textbook dealer-long-gamma signature)?"""
    r = e[e.level == "real"].dropna(subset=["y_rv15"])
    q = pd.qcut(r.f_net_ratio, 4, labels=["Q1 (most negative)", "Q2", "Q3", "Q4 (most positive)"])
    rows = []
    for lab, sub in r.groupby(q, observed=True):
        m = boot_mean(sub, "y_rv15")
        rows.append(dict(symbol=sym, quartile=str(lab), rv15=m["mean"], lo=m["lo"], hi=m["hi"],
                         n=m["n"], net_ratio_mid=sub.f_net_ratio.median(),
                         rv30_pre=sub.f_rv30.mean()))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- driver
def main():
    TAB.mkdir(parents=True, exist_ok=True)
    syms = [spec.PRIMARY, *spec.REPLICATIONS]
    E = entries()
    desc, models, trades, groups, regimes, curves, xeff = [], [], [], [], [], [], {}
    for sym in syms:
        e = E[E.symbol == sym]
        desc.append(descriptive(e, sym))
        regimes.append(regime_check(e, sym))
        groups.append(secondary_groups(e, sym))
        for kind in ("ridge", "gbm"):
            for target in ("y_r15", "y_rv15", "y_abs15"):
                if kind == "gbm" and target != "y_r15":
                    continue
                res, preds, (dev, val, hold) = evaluate_models(e, target, kind)
                res["symbol"] = sym
                models.append(res)
                if kind == "ridge" and target == "y_r15":
                    cost = spec.ROUND_TRIP_COST_PCT.get(sym, spec.ROUND_TRIP_COST_PCT["default"])
                    trades.append(trading(preds, val, hold, sym, cost))
                    if sym == spec.PRIMARY:
                        hold.assign(**{f"pred_{k}": v for k, v in preds["holdout"].items()}) \
                            [["event_id", "date", "y"] + [f"pred_{k}" for k in MODELS]] \
                            .to_csv(TAB / "primary_holdout_predictions.csv", index=False)
        panel = landmark_panel(e[e.split.isin(["dev", "val", "holdout"])], sym)
        xeff[sym] = crossing_effect(panel, e.b.iloc[0])
        lm = []
        for lab, col in (("entry (j=0)", None), ("first crossing", "first_cross"),
                         ("exit far side", "exit_far"), ("exit back", "exit_back")):
            for lvl in ("real", "placebo"):
                sub = panel[panel.level == lvl]
                sub = sub[sub.j == 0] if col is None else sub[sub[col] == 1]
                m = boot_mean(sub, "y")
                lm.append(dict(symbol=sym, landmark=lab, level=lvl, **m))
        pd.DataFrame(lm).to_csv(TAB / f"landmarks_{sym}.csv", index=False)
        curves.append(distance_curve(sym))
        print("done", sym, flush=True)

    # robustness: band sensitivities, stale snapshot, execution delay (primary + reps)
    rob = []
    for tag, path in (("band0.025", "events_b0.025.parquet"), ("band0.05", "events_b0.05.parquet"),
                      ("band0.10", "events_b0.1.parquet"), ("stale_snapshot", "events_lag1.parquet"),
                      ("delay2min", "events_delay2.parquet")):
        Ev = entries(path)
        for sym in syms:
            e = Ev[Ev.symbol == sym]
            res, _, _ = evaluate_models(e, "y_r15", "ridge", tag)
            res["symbol"] = sym
            rob.append(res)
            d = descriptive(e, sym)
            d["tag"] = tag
            d.to_csv(TAB / f"descriptive_{sym}_{tag}.csv", index=False)
    # exploratory pool of single names
    pool = E[~E.symbol.isin(syms + spec.EXCLUDE_FROM_POOL)]
    desc.append(descriptive(pool, "POOL"))
    res, _, _ = evaluate_models(pool, "y_r15", "ridge")
    res["symbol"] = "POOL"
    models.append(res)
    regimes.append(regime_check(pool, "POOL"))

    pd.concat(desc).to_csv(TAB / "descriptive.csv", index=False)
    pd.concat(models).to_csv(TAB / "models.csv", index=False)
    pd.concat(rob).to_csv(TAB / "robustness_models.csv", index=False)
    pd.concat(trades).to_csv(TAB / "trading.csv", index=False)
    pd.concat(groups).to_csv(TAB / "secondary_groups.csv", index=False)
    pd.concat(regimes).to_csv(TAB / "regime_check.csv", index=False)
    pd.concat(curves).to_csv(TAB / "distance_curve.csv", index=False)
    json.dump({k: {l: {kk: float(vv) for kk, vv in d.items()} for l, d in v.items()}
               for k, v in xeff.items()}, open(TAB / "crossing_effect.json", "w"), indent=1)
    # event-aligned mean paths
    pcols = [f"p{k}" for k in range(-10, 31)]
    paths = []
    for sym in syms:
        for lvl in ("real", "placebo"):
            sub = E[(E.symbol == sym) & (E.level == lvl)]
            day = sub.groupby("date")[pcols].mean()
            w = _day_weights(day.index)
            cnt = sub.groupby("date")[pcols].count().to_numpy()
            sm = sub.groupby("date")[pcols].sum().to_numpy()
            bs = (w @ sm) / np.maximum(w @ cnt, 1)
            for i, k in enumerate(range(-10, 31)):
                paths.append(dict(symbol=sym, level=lvl, k=k, mean=sub[pcols[i]].mean(),
                                  lo=np.percentile(bs[:, i], 2.5), hi=np.percentile(bs[:, i], 97.5)))
    pd.DataFrame(paths).to_csv(TAB / "paths.csv", index=False)
    # manifest numbers
    man = E.groupby(["symbol", "level", "split"]).agg(n=("event_id", "size"),
                                                      days=("date", "nunique")).reset_index()
    man.to_csv(TAB / "event_counts.csv", index=False)
    run_controlled()
    print("analysis complete")


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------------- controlled vol
def controlled_vol(e, sym):
    """rv15 ~ indicator + pre-event controls (rv30, range30, |day return|, time-of-day
    bins). Separates a node/encounter 'effect' from already-quiet conditions."""
    r = e[e.level == "real"].dropna(subset=["y_rv15", "f_rv30"]).copy()
    contrasts = {"king": r.f_node_king, "first_encounter": (r.f_n_prior == 0).astype(float),
                 "agg_net_pos": (r.f_net_ratio > 0).astype(float),
                 "upward": (r.f_s > 0).astype(float)}
    tod = pd.get_dummies(pd.cut(r.f_tod, [0, 60, 120, 180, 240, 300, 390]), drop_first=True).astype(float)
    base = pd.concat([r[["f_rv30", "f_range30"]], r.f_dayret.abs().rename("absday"), tod], axis=1)
    days = r.date.to_numpy()
    ud = np.unique(days)
    idx = {u: np.flatnonzero(days == u) for u in ud}
    rows = []
    for name, ind in contrasts.items():
        for ctrl in (False, True):
            X = pd.concat([ind.rename("ind"), base], axis=1) if ctrl else ind.rename("ind").to_frame()
            X = X.assign(const=1.0).to_numpy(float)
            y = r.y_rv15.to_numpy()
            b0 = np.linalg.lstsq(X, y, rcond=None)[0][0]
            bs = [np.linalg.lstsq(X[p], y[p], rcond=None)[0][0]
                  for p in (np.concatenate([idx[u] for u in RNG.choice(ud, len(ud))]) for _ in range(500))]
            rows.append(dict(symbol=sym, contrast=name, controls=ctrl, coef=b0,
                             lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5),
                             rel_to_mean=b0 / y.mean(), n=len(r), days=len(ud)))
    return pd.DataFrame(rows)


def run_controlled():
    E = entries()
    out = [controlled_vol(E[E.symbol == s], s) for s in (spec.PRIMARY, *spec.REPLICATIONS)]
    pool = E[~E.symbol.isin([spec.PRIMARY, *spec.REPLICATIONS, *spec.EXCLUDE_FROM_POOL])]
    out.append(controlled_vol(pool, "POOL"))
    pd.concat(out).to_csv(TAB / "controlled_vol.csv", index=False)
