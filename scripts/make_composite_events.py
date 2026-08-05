"""
Composite-event preprocessing for pyvorotomo.

Optionally run after loading raw events/arrivals to reduce event volume and
average down pick noise. Events whose hypocentres fall within a small radius
`r_km` of a reference event are merged into one "composite" event:

  * The reference is the cluster member with the most arrivals; its hypocentre
    and origin time define the composite.
  * All member arrivals are pooled. Each arrival's time is shifted onto the
    reference's clock by (reference_origin - member_origin), which is equivalent
    to averaging travel times -- the part that is a property of the path, not
    the event timing.
  * Arrivals sharing a (network, station, phase) key are merged to one row by
    averaging the shifted time (and any other listed numeric columns).

Merged-away events are discarded (no traceability kept, by design).

See:
    Lin, G., Shearer, P. M., Hauksson, E., & Thurber, C. H. (2007).
    A three‐dimensional crustal seismic velocity model for southern California
    from a composite event method.
    Journal of Geophysical Research: Solid Earth, 112(B11).

Clustering is greedy: seed on the highest-arrival-count unclaimed event, claim
everything within `r_km` of it, repeat. This guarantees every cluster member is
within `r_km` of its reference (no single-linkage chaining).
"""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

_KM_PER_DEG = 111.195


def _robust_time(times, min_n=3):
    """Rough outlier guard: median for groups of >= min_n picks (a single bad
    pick can't drag the median), plain mean for smaller groups where there are
    too few points to judge an outlier."""
    t = np.asarray(times, dtype=float)
    return float(np.median(t)) if len(t) >= min_n else float(np.mean(t))


def _flag_outliers(times, n_sigma=2.0, min_n=3):
    """Return a boolean mask of picks that deviate from the group median by more
    than n_sigma robust deviations (MAD-scaled). Only acts on groups of >= min_n;
    smaller groups can't support an outlier judgement, so nothing is flagged.

    Detection is independent of the merged value (which is always the median) --
    this only identifies picks to log."""
    t = np.asarray(times, dtype=float)
    if len(t) < min_n:
        return np.zeros(len(t), dtype=bool)
    med = np.median(t)
    mad = np.median(np.abs(t - med))
    if mad == 0:
        return np.zeros(len(t), dtype=bool)        # no robust spread -> flag nothing
    mod_z = 0.6745 * (t - med) / mad               # MAD-scaled deviation (~sigma units)
    return np.abs(mod_z) > n_sigma


def make_composite_events(events, arrivals, r_km=1.5,
                          average_cols=("time", "weight"),
                          outlier_csv=None, outlier_sigma=2.0,
                          verbose=True):
    """
    Parameters
    ----------
    events : DataFrame
        Must contain: event_id, latitude, longitude, depth, time.
    arrivals : DataFrame
        Must contain: event_id, network, station, phase, time.
        May contain: weight, arrival_id, and any other columns (carried through
        via 'first' on each merged group).
    r_km : float
        3D clustering radius in kilometres (isotropic). Should be small relative
        to model cell size / location uncertainty so that pooled arrivals share
        essentially the same ray path.
    average_cols : tuple of str
        Columns averaged when collapsing duplicate (network, station, phase)
        arrivals. Others take the first value in the group.
    outlier_csv : str or None
        If given, path to write a CSV of arrivals flagged as outliers during
        merging (picks deviating > outlier_sigma robust deviations from their
        group median). The merged value is unaffected -- it is always the median;
        this is a diagnostic log only. Times are written as ISO 8601 UTC strings
        (input `time` columns are treated as epoch seconds, per the package
        convention).
    outlier_sigma : float
        MAD-scaled deviation threshold for flagging outliers (only on groups of
        3+ picks).
    verbose : bool
        Print a one-line before/after summary.

    Returns
    -------
    (events_out, arrivals_out) : tuple of DataFrame
    """
    events = events.reset_index(drop=True).copy()
    arrivals = arrivals.copy()

    # arrivals-per-event from the arrivals table (robust to a missing/ stale
    # n_arrivals column on the events frame)
    counts = arrivals.groupby("event_id").size()
    n_arr = events["event_id"].map(counts).fillna(0).astype(int).values

    # local cartesian (km) for 3D distance; flat-earth approx is fine at small r
    lat0 = np.deg2rad(events["latitude"].mean())
    km_lat = _KM_PER_DEG
    km_lon = _KM_PER_DEG * np.cos(lat0)
    x = (events["longitude"].values - events["longitude"].mean()) * km_lon
    y = (events["latitude"].values - events["latitude"].mean()) * km_lat
    z = events["depth"].values.astype(float)
    xyz = np.column_stack([x, y, z])

    tree = cKDTree(xyz)

    # greedy clustering: highest-arrival-count event seeds each cluster
    order = np.argsort(-n_arr)            # descending arrival count
    claimed = np.zeros(len(events), dtype=bool)
    clusters = []                          # (anchor_row, [member_rows])
    for i in order:
        if claimed[i]:
            continue
        members = [m for m in tree.query_ball_point(xyz[i], r_km) if not claimed[m]]
        if i not in members:
            members.append(i)
        claimed[members] = True
        clusters.append((i, members))

    origin = dict(zip(events["event_id"], events["time"]))
    eid_by_row = events["event_id"].values
    src_by_eid = (dict(zip(events["event_id"], events["source_id"]))
                  if "source_id" in events.columns else {})

    avg_cols = [c for c in average_cols if c in arrivals.columns]
    key_cols = ["network", "station", "phase"]
    other_cols = [c for c in arrivals.columns
                  if c not in avg_cols + key_cols + ["event_id"]]
    # 'time' uses the median-based robust reducer; other averaged cols (e.g.
    # weight) use a plain mean.
    agg = {c: "first" for c in other_cols}
    for c in avg_cols:
        agg[c] = (_robust_time if c == "time" else "mean")

    arr_by_event = {eid: df for eid, df in arrivals.groupby("event_id")}

    out_events = []
    out_arr_frames = []
    outlier_rows = []                          # collected flagged picks for the CSV
    for anchor_row, member_rows in clusters:
        anchor_eid = eid_by_row[anchor_row]
        anchor_origin = origin[anchor_eid]

        frames = []
        for m in member_rows:
            eid = eid_by_row[m]
            df = arr_by_event.get(eid)
            if df is None or len(df) == 0:
                continue
            df = df.copy()
            # preserve originals BEFORE shifting/re-IDing, for the outlier log
            df["_orig_event_id"] = eid
            df["_orig_time"] = df["time"].values if "time" in df.columns else np.nan
            if "time" in df.columns:
                df["time"] = df["time"] + (anchor_origin - origin[eid])
            df["event_id"] = anchor_eid
            frames.append(df)

        ev = events.loc[anchor_row].copy()
        if not frames:
            ev["n_arrivals"] = 0
            out_events.append(ev)
            continue

        pooled = pd.concat(frames, ignore_index=True)

        # Flag outliers per (net,sta,phase) group from the shifted times, and
        # record the originating pick details for the log. Does not affect the
        # merged value (still the median below).
        if outlier_csv is not None and "time" in pooled.columns:
            for keys, grp in pooled.groupby(key_cols):
                mask = _flag_outliers(grp["time"].values, n_sigma=outlier_sigma)
                if not mask.any():
                    continue
                med = float(np.median(grp["time"].values))
                net, sta, pha = keys
                for (_, row), is_out in zip(grp.iterrows(), mask):
                    if not is_out:
                        continue
                    oeid = row["_orig_event_id"]
                    outlier_rows.append({
                        "orig_event_id": oeid,
                        "orig_source_id": src_by_eid.get(oeid, None),
                        "composite_event_id": anchor_eid,
                        "network": net, "station": sta, "phase": pha,
                        "arrival_time": row["_orig_time"],          # epoch s (original)
                        "origin_time": origin.get(oeid, np.nan),    # epoch s
                        "travel_time": row["_orig_time"] - origin.get(oeid, np.nan),
                        "group_median_tt": med - anchor_origin,     # median travel time
                        "deviation_s": (row["time"] - med),         # shifted-time deviation
                    })

        merged = pooled.drop(columns=["_orig_event_id", "_orig_time"]) \
                       .groupby(key_cols, as_index=False).agg(agg)
        merged["event_id"] = anchor_eid

        # if derived event-location columns exist, set them to the anchor's
        if "event_latitude" in merged.columns:
            merged["event_latitude"] = events.loc[anchor_row, "latitude"]
        if "event_longitude" in merged.columns:
            merged["event_longitude"] = events.loc[anchor_row, "longitude"]
        if "event_depth" in merged.columns:
            merged["event_depth"] = events.loc[anchor_row, "depth"]

        ev["n_arrivals"] = len(merged)
        out_events.append(ev)
        out_arr_frames.append(merged)

    events_out = pd.DataFrame(out_events).reset_index(drop=True)
    arrivals_out = (pd.concat(out_arr_frames, ignore_index=True)
                    if out_arr_frames else arrivals.iloc[0:0].copy())

    if outlier_csv is not None:
        cols = ["orig_event_id", "orig_source_id", "composite_event_id",
                "network", "station", "phase",
                "arrival_time", "origin_time", "travel_time",
                "group_median_tt", "deviation_s"]
        outliers = pd.DataFrame(outlier_rows, columns=cols)
        # round travel-time columns to millisecond precision (3 dp)
        for c in ("travel_time", "group_median_tt", "deviation_s"):
            outliers[c] = outliers[c].round(3)
        # epoch seconds -> ISO 8601 UTC strings, millisecond precision
        for c in ("arrival_time", "origin_time"):
            outliers[c] = (pd.to_datetime(outliers[c], unit="s", utc=True,
                                          errors="coerce")
                           .dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
                           .str.slice(0, -3) + "Z")
        outliers.to_csv(outlier_csv, index=False)
        if verbose:
            print(f"  flagged {len(outliers)} outlier arrivals "
                  f"(>{outlier_sigma} robust sigma) -> {outlier_csv}")

    if verbose:
        print(f"composite events: {len(events)} -> {len(events_out)} events, "
              f"{len(arrivals)} -> {len(arrivals_out)} arrivals "
              f"({len(clusters)} clusters, r={r_km} km)")

    return events_out, arrivals_out


def analyze_outliers(outliers, top_n=30, min_events_systematic=3,
                     consistency_ratio=1.0, csv_prefix=None, verbose=True):
    """
    Higher-level QC summary of the outlier records produced by
    make_composite_events. Operates purely on the flagged picks.

    Parameters
    ----------
    outliers : DataFrame or str
        The outlier records, either as a DataFrame or a path to the CSV written
        by make_composite_events. Must contain: orig_event_id, orig_source_id,
        network, station, phase, deviation_s.
    top_n : int
        How many rows to print for the event and station tables.
    min_events_systematic : int
        A station must have flagged picks from at least this many DISTINCT
        original events to be considered for a systematic-issue flag (so one bad
        event can't make a station look systematically broken).
    consistency_ratio : float
        A station is flagged systematic when |mean(deviation)| / (std + eps)
        exceeds this. ~0 => deviations scatter around zero (just noisy);
        large => coherently signed (timing/location bias).
    csv_prefix : str or None
        If given, write the three tables to '{prefix}_events.csv',
        '{prefix}_stations.csv', '{prefix}_systematic.csv'.
    verbose : bool
        Print the tables.

    Returns
    -------
    dict with 'events', 'stations', 'systematic' DataFrames.
    """
    if isinstance(outliers, str):
        outliers = pd.read_csv(outliers)

    if len(outliers) == 0:
        empty = pd.DataFrame()
        if verbose:
            print("No outliers to analyze.")
        return {"events": empty, "stations": empty, "systematic": empty}

    absdev = outliers["deviation_s"].abs()

    # --- worst events: biggest summed |deviation| (most + largest fixables) ---
    ev = (outliers.assign(absdev=absdev)
          .groupby("orig_event_id")
          .agg(orig_source_id=("orig_source_id", "first"),
               n_outliers=("deviation_s", "size"),
               sum_abs_dev=("absdev", "sum"),
               max_abs_dev=("absdev", "max"))
          .reset_index()
          .sort_values("sum_abs_dev", ascending=False)
          .reset_index(drop=True))
    ev[["sum_abs_dev", "max_abs_dev"]] = ev[["sum_abs_dev", "max_abs_dev"]].round(3)

    # --- station ranking: population in the outlier list, worst -> best ---
    st = (outliers.groupby("station")
          .agg(network=("network", "first"),
               n_outliers=("deviation_s", "size"),
               n_events=("orig_event_id", "nunique"),
               mean_signed_dev=("deviation_s", "mean"))
          .reset_index()
          .sort_values("n_outliers", ascending=False)
          .reset_index(drop=True))
    st["mean_signed_dev"] = st["mean_signed_dev"].round(3)

    # --- systematic station issues: coherently-signed deviation, multi-event ---
    def _sysrow(g):
        m = g["deviation_s"].mean()
        s = g["deviation_s"].std(ddof=0)
        return pd.Series({
            "network": g["network"].iloc[0],
            "n_outliers": len(g),
            "n_events": g["orig_event_id"].nunique(),
            "mean_signed_dev": m,
            "std_dev": s,
            "consistency": abs(m) / (s + 1e-9),
        })

    sysg = outliers.groupby("station").apply(_sysrow, include_groups=False).reset_index()
    systematic = (sysg[(sysg["n_events"] >= min_events_systematic) &
                       (sysg["consistency"] >= consistency_ratio)]
                  .sort_values("mean_signed_dev", key=lambda c: c.abs(),
                               ascending=False)
                  .reset_index(drop=True))

    # per-phase signed-mean breakdown for the flagged systematic stations
    if len(systematic):
        phase_means = (outliers[outliers["station"].isin(systematic["station"])]
                       .groupby(["station", "phase"])["deviation_s"]
                       .mean().round(3).unstack(fill_value=np.nan))
        phase_means.columns = [f"mean_dev_{c}" for c in phase_means.columns]
        systematic = systematic.merge(phase_means, on="station", how="left")
    for c in ("mean_signed_dev", "std_dev", "consistency"):
        systematic[c] = systematic[c].round(3)

    if verbose:
        print(f"\n=== WORST EVENTS (by summed |deviation|, top {top_n}) ===")
        print(ev.head(top_n).to_string(index=False))
        print(f"\n=== STATION RANKING (by outlier count, top {top_n}) ===")
        print(st.head(top_n).to_string(index=False))
        print(f"\n=== SYSTEMATIC STATION ISSUES "
              f"(coherent sign, >={min_events_systematic} events) ===")
        if len(systematic):
            print(systematic.to_string(index=False))
        else:
            print("  none flagged")

    if csv_prefix is not None:
        ev.to_csv(f"{csv_prefix}_events.csv", index=False)
        st.to_csv(f"{csv_prefix}_stations.csv", index=False)
        systematic.to_csv(f"{csv_prefix}_systematic.csv", index=False)
        if verbose:
            print(f"\nwrote {csv_prefix}_{{events,stations,systematic}}.csv")

    return {"events": ev, "stations": st, "systematic": systematic}


"""

example:

events_comp, arrivals_comp = make_composite_events(events,arrivals,r_km=1.5,outlier_csv='composite_outliers.1.csv')
analyze_outliers("composite_outliers.csv", top_n=50, csv_prefix="qc")

out = 'wa.composite.1.h5'
events_comp.to_hdf(out,key='events',complevel=5,complib='zlib')
arrivals_comp.to_hdf(out,key='arrivals',complevel=5,complib='zlib')


"""