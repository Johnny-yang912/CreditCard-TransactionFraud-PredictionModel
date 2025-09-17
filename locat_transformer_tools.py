from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Optional
from collections import deque, Counter
from sklearn.base import BaseEstimator, TransformerMixin


# ---------- utilities ----------
def _to_datetime(s: pd.Series) -> pd.Series:
    """Safely convert to pandas datetime (NaT for invalid)."""
    if np.issubdtype(getattr(s, "dtype", np.dtype("O")), np.datetime64):
        return s
    return pd.to_datetime(s, errors="coerce")


def _haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Vectorized haversine distance (km). Inputs can be array-like/Series."""
    R = 6371.0  # Earth radius in km
    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lon1 = np.radians(np.asarray(lon1, dtype=float))
    lat2 = np.radians(np.asarray(lat2, dtype=float))
    lon2 = np.radians(np.asarray(lon2, dtype=float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def _as_tuple(x):
    return (x,) if isinstance(x, str) else tuple(x)

@dataclass
class _Event:
    ts: pd.Timestamp
    city: Any
    merch_key: Tuple[Any, ...]
    merch_lat: float
    merch_long: float


# ---------- main transformer ----------
class GeoTemporalFeatures(BaseEstimator, TransformerMixin):
    """
    Derive leakage-safe geo + temporal features for fraud detection / transaction modeling.

    Outputs (new columns):
        - distance_home_to_merchant (km)
        - travel_speed_kmh (km/h), per card (cc_num) from previous txn location -> current merchant
        - uniq_cities_24h, uniq_merchants_24h within (t-24h, t) per card

    Leakage safety:
        - fit(X) builds training history only.
        - transform(X) uses training history + earlier rows within current batch; never looks ahead.

    Parameters
    ----------
    lat_col, lon_col, merch_lat_col, merch_lon_col : str
        Coordinate columns for home (lat/lon) and merchant (merch_lat/merch_long).
    city_col : str
        City/categorical location column per transaction.
    cc_col : str
        Card/account id for grouping history (e.g., card number hashed/obfuscated).
    time_col : str
        Timestamp column (any parseable to datetime).
    merchant_key_cols : Sequence[str], default=("merch_lat", "merch_long")
        Columns that define a merchant identity (e.g., ("merchant",) if you have a merchant id).
    hours_window : int, default=24
        Sliding window size in hours for diversity counts.
    first_speed_fill : {"nan","zero"}, default="nan"
        Fill for speed when there's no previous txn available.
    return_dataframe : bool, default=True
        If True, returns DataFrame; else returns numpy array.
    append_original : bool, default=True
        If True, return original columns along with newly derived features.
    drop_time_col_in_output : bool, default=False
        If True, remove time_col from outputs (to avoid feeding datetime into model).
    validate_cols : bool, default=True
        If True, check for presence of required columns.
    """

    NEW_COLS = [
        "distance_home_to_merchant",
        "travel_speed_kmh",
        "uniq_cities_24h",
        "uniq_merchants_24h",
    ]

    def __init__(
        self,
        lat_col: str = "lat",
        lon_col: str = "long",
        merch_lat_col: str = "merch_lat",
        merch_lon_col: str = "merch_long",
        city_col: str = "city",
        cc_col: str = "cc_num",
        time_col: str = "trans_date_trans_time",
        merchant_key_cols: Sequence[str] = ("merch_lat", "merch_long"),
        hours_window: int = 24,
        first_speed_fill: str = "nan",
        return_dataframe: bool = True,
        append_original: bool = True,
        drop_time_col_in_output: bool = False,
        validate_cols: bool = True,
    ):
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.merch_lat_col = merch_lat_col
        self.merch_lon_col = merch_lon_col
        self.city_col = city_col
        self.cc_col = cc_col
        self.time_col = time_col
        self.merchant_key_cols = _as_tuple(merchant_key_cols)
        self.hours_window = int(hours_window)
        self.first_speed_fill = first_speed_fill
        self.return_dataframe = return_dataframe
        self.append_original = append_original
        self.drop_time_col_in_output = drop_time_col_in_output
        self.validate_cols = validate_cols

        # internal
        self._fitted_: bool = False
        self._history_: Dict[Any, List[_Event]] = {}

    # -------------- scikit-learn API --------------
    def fit(self, X, y=None):
        X = self._to_df(X)
        if self.validate_cols:
            self._check_required_columns(X)
        self._history_ = self._build_history(X)  # training history only
        self._fitted_ = True
        return self

    def transform(self, X):
        if not self._fitted_:
            # allow transform on fresh instance (not typical in pipelines but safe)
            self._history_ = {}

        X = self._to_df(X).copy()
        if self.validate_cols:
            self._check_required_columns(X)
        idx = X.index

        # ensure datetime
        X[self.time_col] = _to_datetime(X[self.time_col])

        # (1) home -> merchant distance (km)
        dist_home = _haversine_km(
            X[self.lat_col].astype(float),
            X[self.lon_col].astype(float),
            X[self.merch_lat_col].astype(float),
            X[self.merch_lon_col].astype(float),
        )

        out = pd.DataFrame(index=idx, data={
            "distance_home_to_merchant": dist_home,
            "travel_speed_kmh": np.nan,
            "uniq_cities_24h": 0,
            "uniq_merchants_24h": 0,
        })

        # (2)(3) per-card time-ordered scan for speed + 24h diversity
        hours = pd.Timedelta(hours=self.hours_window)
        X["_row_id_"] = np.arange(len(X))
        X_sorted = X.sort_values([self.cc_col, self.time_col, "_row_id_"], kind="mergesort")

        prior_hist = self._history_

        for cc, grp in X_sorted.groupby(self.cc_col, sort=False):
            window: deque[_Event] = deque()
            city_counter: Counter = Counter()
            merch_counter: Counter = Counter()

            prior_events = prior_hist.get(cc, [])
            prior_idx = 0
            last_evt_for_speed: Optional[_Event] = None

            for i, (_, r) in enumerate(grp.iterrows()):
                t = r[self.time_col]
                city = r[self.city_col]
                mk = tuple(r[c] for c in self.merchant_key_cols)

                # include all training prior events up to time t
                while prior_idx < len(prior_events) and prior_events[prior_idx].ts <= t:
                    ev = prior_events[prior_idx]
                    window.append(ev)
                    city_counter[ev.city] += 1
                    merch_counter[ev.merch_key] += 1
                    last_evt_for_speed = ev
                    prior_idx += 1

                # prune to (t-24h, t)
                cutoff = t - hours
                while window and (window[0].ts < cutoff):
                    old = window.popleft()
                    city_counter[old.city] -= 1
                    if city_counter[old.city] == 0:
                        del city_counter[old.city]
                    merch_counter[old.merch_key] -= 1
                    if merch_counter[old.merch_key] == 0:
                        del merch_counter[old.merch_key]

                uniq_cities = len(city_counter)
                uniq_merchants = len(merch_counter)

                # travel speed: prev (within batch) else last training prior
                if i > 0:
                    prev = grp.iloc[i - 1]
                    prev_lat = float(prev[self.merch_lat_col])
                    prev_lon = float(prev[self.merch_lon_col])
                    prev_t = prev[self.time_col]
                else:
                    if last_evt_for_speed is not None:
                        prev_lat = float(last_evt_for_speed.merch_lat)
                        prev_lon = float(last_evt_for_speed.merch_long)
                        prev_t = last_evt_for_speed.ts
                    else:
                        prev_t = None

                if prev_t is None or pd.isna(prev_t):
                    speed_val = np.nan if self.first_speed_fill == "nan" else 0.0
                else:
                    dist_km = _haversine_km(
                        [prev_lat], [prev_lon],
                        [float(r[self.merch_lat_col])],
                        [float(r[self.merch_lon_col])]
                    )[0]
                    dt_hours = (t - prev_t).total_seconds() / 3600.0
                    speed_val = np.nan if dt_hours <= 0 else dist_km / dt_hours

                oi = r.name
                out.at[oi, "travel_speed_kmh"] = speed_val
                out.at[oi, "uniq_cities_24h"] = uniq_cities
                out.at[oi, "uniq_merchants_24h"] = uniq_merchants

                # push current as history for subsequent rows in the same batch
                cur_event = _Event(
                    ts=t, city=city, merch_key=mk,
                    merch_lat=float(r[self.merch_lat_col]),
                    merch_long=float(r[self.merch_lon_col]),
                )
                window.append(cur_event)
                city_counter[city] += 1
                merch_counter[mk] += 1

        # combine with original if requested
        if self.append_original:
            result = pd.concat([X.drop(columns=["_row_id_"]), out], axis=1)
            if self.drop_time_col_in_output and self.time_col in result.columns:
                result = result.drop(columns=[self.time_col])
        else:
            result = out

        return result if self.return_dataframe else result.values

    # -------------- helpers --------------
    def _to_df(self, X) -> pd.DataFrame:
        return X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

    def _check_required_columns(self, X: pd.DataFrame) -> None:
        req = {
            self.lat_col, self.lon_col,
            self.merch_lat_col, self.merch_lon_col,
            self.city_col, self.cc_col, self.time_col, *self.merchant_key_cols
        }
        missing = [c for c in req if c not in X.columns]
        if missing:
            raise ValueError(f"[GeoTemporalFeatures] Missing columns: {missing}")

    def _build_history(self, X: pd.DataFrame) -> Dict[Any, List[_Event]]:
        # Only essential columns to reduce memory
        cols = [
            self.cc_col, self.time_col, self.city_col,
            *self.merchant_key_cols, self.merch_lat_col, self.merch_lon_col,
        ]
        df = X.loc[:, [c for c in cols if c in X.columns]].copy()
        df = df.dropna(subset=[self.cc_col, self.time_col])
        df[self.time_col] = _to_datetime(df[self.time_col])
        df = df.sort_values([self.cc_col, self.time_col], kind="mergesort")

        history: Dict[Any, List[_Event]] = {}
        for cc, grp in df.groupby(self.cc_col, sort=False):
            evs: List[_Event] = []
            for _, r in grp.iterrows():
                mk = tuple(r.get(c) for c in self.merchant_key_cols)
                evs.append(_Event(
                    ts=r[self.time_col],
                    city=r[self.city_col],
                    merch_key=mk,
                    merch_lat=float(r[self.merch_lat_col]) if pd.notna(r[self.merch_lat_col]) else np.nan,
                    merch_long=float(r[self.merch_lon_col]) if pd.notna(r[self.merch_lon_col]) else np.nan,
                ))
            history[cc] = evs
        return history

    # -------------- feature names --------------
    def get_feature_names_out(self, input_features: Optional[Iterable[str]] = None) -> np.ndarray:
        """
        If append_original=True: return [input_features (+ maybe without time_col if dropped)] + NEW_COLS
        Else: return NEW_COLS
        """
        input_feats = list(input_features) if input_features is not None else None
        if not self.append_original:
            return np.array(self.NEW_COLS, dtype=object)

        # try best-effort: if caller provided features, respect them; else infer nothing (caller should handle)
        if input_feats is None:
            # unknown input feature order; just return new ones
            base = []
        else:
            base = input_feats.copy()

        if self.drop_time_col_in_output and self.time_col in base:
            base = [c for c in base if c != self.time_col]

        return np.array([*base, *self.NEW_COLS], dtype=object)

