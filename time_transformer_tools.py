from __future__ import annotations
from typing import List, Optional, Union, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class TimeFeaturesTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        datetime_col: str,
        group_cols: Union[str, List[str], Tuple[str, ...]],
        bucket_col_name: str = "time_bucket",
        delta_col_name: str = "delta_sec_prev_tx",
        unusual_col_name: str = "is_unusual_hour",
        one_hot: bool = True,
        drop_datetime: bool = False,
        fill_first_delta: Optional[Union[int, float]] = None,
    ):
        # ⚠️ 保留原參數，完全不動（讓 sklearn clone 安心）
        self.datetime_col = datetime_col
        self.group_cols = group_cols
        self.bucket_col_name = bucket_col_name
        self.delta_col_name = delta_col_name
        self.unusual_col_name = unusual_col_name
        self.one_hot = one_hot
        self.drop_datetime = drop_datetime
        self.fill_first_delta = fill_first_delta

        # 在 fit 後建立的屬性
        self.group_hour_quantiles_: Optional[pd.DataFrame] = None
        self.global_hour_bounds_: Optional[Tuple[float, float]] = None
        self.bucket_categories_ = ["early_morning", "morning", "afternoon", "night"]

        # 內部使用，不屬於參數，不影響 clone
        self._group_cols_tuple_: Optional[Tuple[str, ...]] = None

    # -------------------- helpers --------------------
    @staticmethod
    def _ensure_dt(s: pd.Series) -> pd.Series:
        if not np.issubdtype(s.dtype, np.datetime64):
            return pd.to_datetime(s, errors="coerce", utc=False)
        return s

    @staticmethod
    def _bucket_by_hour(hour: pd.Series) -> pd.Series:
        # 用 pandas.cut 分桶，不會有 np.select 的 dtype 衝突問題
        bins = [0, 6, 12, 18, 24]
        labels = ["early_morning", "morning", "afternoon", "night"]
        out = pd.cut(hour, bins=bins, labels=labels, right=False, include_lowest=True)
        # 回傳 Categorical（允許 NaN）
        return out


    def _ensure_group_tuple(self):
        """在 fit/transform 開頭呼叫：把 group_cols 轉成不可變 tuple 存到 _group_cols_tuple_。"""
        if self._group_cols_tuple_ is None:
            gc = self.group_cols
            if isinstance(gc, str):
                self._group_cols_tuple_ = (gc,)
            elif isinstance(gc, tuple):
                self._group_cols_tuple_ = gc
            else:
                # list 或其他可迭代 → 轉 tuple（不回寫到 self.group_cols）
                self._group_cols_tuple_ = tuple(gc)

    def _lookup_bounds(self, grp_key: Tuple) -> Tuple[float, float]:
        try:
            row = self.group_hour_quantiles_.loc[grp_key]
            return float(row["low"]), float(row["high"])
        except Exception:
            return self.global_hour_bounds_

    # -------------------- sklearn API --------------------
    def fit(self, X: pd.DataFrame, y=None):
        if self.datetime_col not in X.columns:
            raise KeyError(f"datetime_col '{self.datetime_col}' not in columns")

        self._ensure_group_tuple()
        for c in self._group_cols_tuple_:
            if c not in X.columns:
                raise KeyError(f"group col '{c}' not in columns")

        df = X.copy()
        dt = self._ensure_dt(df[self.datetime_col])
        hour = dt.dt.hour

        # 依 group 計算 5% 與 95% 分位數
        tmp = df[list(self._group_cols_tuple_)].copy()
        tmp["_hour_"] = hour.values
        self.group_hour_quantiles_ = (
            tmp.groupby(list(self._group_cols_tuple_), sort=False)["_hour_"]
               .quantile([0.05, 0.95])
               .unstack(level=-1)
               .rename(columns={0.05: "low", 0.95: "high"})
        )

        # 全域備援
        self.global_hour_bounds_ = (float(hour.quantile(0.05)), float(hour.quantile(0.95)))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.group_hour_quantiles_ is None or self.global_hour_bounds_ is None:
            raise RuntimeError("Transformer is not fitted yet. Call fit before transform.")
        self._ensure_group_tuple()

        df = X.copy()

        # 1) 時段分類（含 one-hot）
        dt = self._ensure_dt(df[self.datetime_col])
        hour = dt.dt.hour

        if self.one_hot:
            bucket = self._bucket_by_hour(hour).astype(
                pd.CategoricalDtype(categories=self.bucket_categories_)
            )
            dummies = pd.get_dummies(bucket, prefix=self.bucket_col_name, dummy_na=False)
            fixed = [f"{self.bucket_col_name}_{c}" for c in self.bucket_categories_]
            for col in fixed:
                if col not in dummies.columns:
                    dummies[col] = 0
            dummies = dummies[fixed]
            df = pd.concat([df, dummies], axis=1)
        else:
            df[self.bucket_col_name] = self._bucket_by_hour(hour)


        # 2) 同組相鄰交易秒差
        sort_cols = list(self._group_cols_tuple_) + [self.datetime_col]
        df_sorted = df.sort_values(sort_cols, kind="mergesort")
        delta = (
            df_sorted.groupby(list(self._group_cols_tuple_), sort=False)[self.datetime_col]
                     .apply(lambda s: self._ensure_dt(s).diff().dt.total_seconds())
                     .reset_index(level=list(self._group_cols_tuple_), drop=True)
        )
        if self.fill_first_delta is not None:
            delta = delta.fillna(self.fill_first_delta)
        df_sorted[self.delta_col_name] = delta.values
        df = df_sorted.sort_index()

        # 3) 不尋常時段標記
        grp_index = df[list(self._group_cols_tuple_)].apply(tuple, axis=1)
        lo, hi = [], []
        for g in grp_index:
            l, h = self._lookup_bounds(g)
            lo.append(l); hi.append(h)
        lo = pd.Series(lo, index=df.index)
        hi = pd.Series(hi, index=df.index)
        df[self.unusual_col_name] = ((hour < lo) | (hour > hi)).astype(int)

        if self.drop_datetime:
            df = df.drop(columns=[self.datetime_col])

        return df


