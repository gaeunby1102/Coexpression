## Lasy Updated : 25.11.02 by GEB
# v2.1 - Added groupwise plotting

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import sparse
from typing import Optional, Sequence, Dict, List, Tuple, Union
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

try:
    from scipy.stats import pearsonr, ttest_ind, ranksums
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


class GeneSetCorrTester:
    """

    Parameters
    ----------
    layer : Optional[str]
        None -> .X 
    groupby : Optional[str]
    method : {'vectorized', 'scipy'}
    batch_size : int
    random_state : Optional[int]
    """

    def __init__(
        self,
        layer: Optional[str] = None,
        groupby: Optional[str] = None,
        method: str = "vectorized",
        batch_size: int = 5000,
        random_state: Optional[int] = 42,
    ):
        assert method in ("vectorized", "scipy")
        if method == "scipy" and not _HAS_SCIPY:
            raise ImportError("Install scipy.")

        self.layer = layer
        self.groupby = groupby
        self.method = method
        self.batch_size = int(batch_size)
        self.rng = np.random.default_rng(random_state)

        self._adata = None
        self._target_gene: Optional[str] = None
        self._r_all: Optional[pd.Series] = None  # index=gene, values=r

    # ------------------ Helper ------------------
    @staticmethod
    def _to_dense(a) -> np.ndarray:
        return a.toarray() if sparse.issparse(a) else np.asarray(a)

    @staticmethod
    def _pearson_corr_vectorized_block(X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        X: (n_samples, n_genes_block), y: (n_samples,)
        """
        y = y.astype(float)
        X = X.astype(float)

        # NaN-safe center/std
        y_centered = y - np.nanmean(y)
        X_centered = X - np.nanmean(X, axis=0, keepdims=True)

        y_std = np.nanstd(y, ddof=1)
        X_std = np.nanstd(X, axis=0, ddof=1)

        cov = np.nanmean(X_centered * y_centered[:, None], axis=0)

        r = np.full(X.shape[1], np.nan)
        valid = (
            np.isfinite(y_std) & (y_std > 0) &
            np.isfinite(X_std) & (X_std > 0)
        )
        r[valid] = cov[valid] / (X_std[valid] * y_std)
        return r

    def _get_matrix(self, genes: Sequence[str]):
        if self.layer is None:
            return self._adata[:, genes].X
        else:
            if self.layer not in self._adata.layers:
                raise ValueError(f"Layer '{self.layer}' doesn't exist in adata.")
            return self._adata[:, genes].layers[self.layer]
        
    def get_groupwise_stats(
        self,
        gene_set: Sequence[str],
        ) -> pd.DataFrame:
        """
        Returns
        -------
        pd.DataFrame with columns: [group, mean_r_set, mean_r_bg, t_stat, p_value, q_value]
        """
        from scipy.stats import ttest_ind
        from statsmodels.stats.multitest import multipletests

        if self._r_all is None:
            self.compute_all_correlations()
        if self.groupby is None:
            raise ValueError("groupby is None")
        if self._adata is None or self._target_gene is None:
            raise RuntimeError("Need to call fit() first")

        df_set = self.compute_correlations_by_group(gene_set)
        bg_genes = [g for g in self._r_all.index if g not in gene_set]
        df_bg = self.compute_correlations_by_group(bg_genes)

        groups = sorted(df_set[self.groupby].unique())

        rows = []
        p_vals = []

        for grp in groups:
            r_set = df_set[df_set[self.groupby] == grp]["r"].dropna().values
            r_bg = df_bg[df_bg[self.groupby] == grp]["r"].dropna().values

            if r_set.size > 1 and r_bg.size > 1:
                t_stat, p_val = ttest_ind(r_set, r_bg, equal_var=False, nan_policy="omit")
            else:
                t_stat, p_val = np.nan, np.nan

            rows.append({
                self.groupby: grp,
                "mean_r_set": np.nanmean(r_set),
                "mean_r_bg": np.nanmean(r_bg),
                "t_stat": t_stat,
                "p_value": p_val
            })
            p_vals.append(p_val)
        # FDR correction
        p_vals_raw = p_vals[:]  
        has_finite = any(np.isfinite(p) for p in p_vals_raw)
        if has_finite:
            p_clean = [1.0 if (p is None or not np.isfinite(p)) else p for p in p_vals_raw]
            _, q_vals, _, _ = multipletests(p_clean, method="fdr_bh")
            q_vals = [q if np.isfinite(p) else np.nan for p, q in zip(p_vals_raw, q_vals)]
        else:
            q_vals = [np.nan] * len(p_vals_raw)

        for row, q in zip(rows, q_vals):
            row["q_value"] = q

        return pd.DataFrame(rows, columns=[self.groupby, "mean_r_set", "mean_r_bg", "t_stat", "p_value", "q_value"])


    
    # ------------------ public API ------------------
    def fit(self, adata, target_gene: str) -> "GeneSetCorrTester":
        self._adata = adata
        self._target_gene = target_gene

        if target_gene not in set(adata.var_names):
            raise ValueError(f"No target_gene '{target_gene}' in adata.var.")
        self._r_all = None
        return self

    
    ##sample level
    def compute_all_correlations(self) -> pd.DataFrame:
        from scipy.stats import t
        from statsmodels.stats.multitest import multipletests
        """
        Returns
        -------
        pd.DataFrame : index=gene, columns=['r', 'p', 'q']
        """
        if self._adata is None or self._target_gene is None:
            raise RuntimeError("Call fit(adata, target_gene) first.")

        genes_all = list(self._adata.var_names)
        genes_all = [g for g in genes_all if g != self._target_gene]

        y_mat = self._get_matrix([self._target_gene])
        y = self._to_dense(y_mat).ravel()

        r_values = np.full(len(genes_all), np.nan, dtype=float)
        p_values = np.full(len(genes_all), np.nan, dtype=float)

        if self.method == "vectorized":
            for start in range(0, len(genes_all), self.batch_size):
                end = min(start + self.batch_size, len(genes_all))
                batch_genes = genes_all[start:end]
                X_mat = self._get_matrix(batch_genes)
                X = self._to_dense(X_mat)

                r_batch = self._pearson_corr_vectorized_block(X, y)
                r_values[start:end] = r_batch

                valid_pairs = (~np.isnan(X)) & (~np.isnan(y)[:, None])
                n = np.sum(valid_pairs, axis=0)
                df = n - 2
                valid = (df > 0) & np.isfinite(r_batch)
                t_stat = np.full_like(r_batch, np.nan, dtype=float)
                with np.errstate(divide='ignore', invalid='ignore'):
                    t_stat[valid] = r_batch[valid] * np.sqrt(df[valid] / (1 - r_batch[valid] ** 2))
                p_values[start:end] = 2 * t.sf(np.abs(t_stat), df)

        else:  # scipy loop
            from scipy.stats import pearsonr
            y_std = np.nanstd(y, ddof=1)
            for i, g in enumerate(genes_all):
                xi = self._to_dense(self._get_matrix([g])).ravel()
                if y_std > 0 and np.nanstd(xi, ddof=1) > 0:
                    mask = ~(np.isnan(xi) | np.isnan(y))
                    if mask.sum() > 1:
                        r, p = pearsonr(xi[mask], y[mask])
                        r_values[i] = r
                        p_values[i] = p

        # FDR correction
        _, q_values, _, _ = multipletests(p_values, method="fdr_bh")

        self._r_all = pd.DataFrame({
            "r": r_values,
            "p": p_values,
            "q": q_values
        }, index=genes_all)

        return self._r_all

    
    
    ## by group
    def compute_correlations_by_group(
        self,
        gene_set: Sequence[str]
    ) -> pd.DataFrame:
    
        from scipy.stats import t
        from statsmodels.stats.multitest import multipletests

        if self._adata is None or self._target_gene is None:
            raise RuntimeError("adata is None or target_gene is None")
        if self.groupby is None:
            raise ValueError("groupby is None")
        if self.groupby not in self._adata.obs.columns:
            raise ValueError(f"{self.groupby} col in adata.obs")

        gene_set_present = [g for g in gene_set if g in self._adata.var_names]
        if not gene_set_present:
            raise ValueError("No valid gene in adata.var_names")

        X_target = self._get_matrix([self._target_gene])
        X_genes  = self._get_matrix(gene_set_present)

        y_all = self._to_dense(X_target).ravel()
        X_all = self._to_dense(X_genes)

        rows = []

        for grp, idx in self._adata.obs.groupby(self.groupby).indices.items():
            y_grp = y_all[idx]
            X_grp = X_all[idx, :]

            r_grp = self._pearson_corr_vectorized_block(X_grp, y_grp)

        
            finite_y = np.isfinite(y_grp)
            finite_X = np.isfinite(X_grp)
            n_grp = np.sum(finite_X & finite_y[:, None], axis=0).astype(float)

            df = n_grp - 2
            p_vals = np.full_like(r_grp, 1.0, dtype=float)  
            valid = (df > 0) & np.isfinite(r_grp)
            if np.any(valid):
                t_stat = np.empty_like(r_grp, dtype=float)
                t_stat.fill(np.nan)
        
                denom = np.clip(1.0 - r_grp[valid]**2, 1e-15, None)
                t_stat[valid] = r_grp[valid] * np.sqrt(df[valid] / denom)
                p_vals[valid] = 2 * t.sf(np.abs(t_stat[valid]), df[valid])

            df_grp = pd.DataFrame({
                self.groupby: grp,
                "gene": gene_set_present,
                "r": r_grp,
                "n": n_grp,
                "p": p_vals
            })

            p_for_fdr = np.nan_to_num(df_grp["p"].to_numpy(), nan=1.0, posinf=1.0, neginf=1.0)
            _, q_vals, _, _ = multipletests(p_for_fdr, method="fdr_bh")
            df_grp["q"] = q_vals

            rows.append(df_grp)

        df_result = pd.concat(rows, ignore_index=True)
        return df_result
    
    def subset_correlations(self, gene_list: Sequence[str]) -> pd.Series:
        if self._r_all is None:
            self.compute_all_correlations()
        genes = [g for g in gene_list if g in self._r_all.index]
        if len(genes) == 0:
            raise ValueError("No gene_list in adata.")
        return self._r_all.loc[genes].copy()


    # diff. test
    def test_gene_set_significance(
        self,
        gene_set: Sequence[str],
        alternative: str = "two-sided",  # 'greater', 'less', 'two-sided'
        use_ttest: bool = True, 
        use_wilcoxon: bool = True,
        use_permutation: bool = True,
        n_permutations: int = 10000,
        ) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Returns
        -------
        dict with keys:
        - 'n_set', 'n_bg', 'mean_set', 'mean_bg'
        - 'ttest_p', 'wilcoxon_p', perm_p'
        """
        if self._r_all is None:
            self.compute_all_correlations()

        set_genes = [g for g in gene_set if g in self._r_all.index]
        if len(set_genes) == 0:
            raise ValueError("No valid gene in gene set")
        r_set = self._r_all.loc[set_genes,"r"].values

        bg_genes = [g for g in self._r_all.index if g not in set_genes]
        r_bg = self._r_all.loc[bg_genes,"r"].values

        out = dict(
            n_set=len(r_set),
            n_bg=len(r_bg),
            mean_set=float(np.nanmean(r_set)),
            mean_bg=float(np.nanmean(r_bg)),
        )

        def to_two_tailed(p_raw):
            return min(1.0, p_raw * 2.0)

        if use_ttest and _HAS_SCIPY:
            t_stat, p_t = ttest_ind(r_set, r_bg, equal_var=False, nan_policy="omit")
            if alternative == "greater":
                # H1: mean of Geneset > mean BG
                p_adj = p_t / 2 if (np.nanmean(r_set) > np.nanmean(r_bg)) else 1 - p_t / 2
            elif alternative == "less":
                p_adj = p_t / 2 if (np.nanmean(r_set) < np.nanmean(r_bg)) else 1 - p_t / 2
            else:
                p_adj = p_t
            out["ttest_p"] = float(max(min(p_adj, 1.0), 0.0))

        # Wilcoxon rank-sum (ranksums)
        if use_wilcoxon and _HAS_SCIPY:
            u_stat, p_w = ranksums(r_set, r_bg)
            if alternative == "greater":
                p_adj = p_w / 2 if (np.nanmean(r_set) > np.nanmean(r_bg)) else 1 - p_w / 2
            elif alternative == "less":
                p_adj = p_w / 2 if (np.nanmean(r_set) < np.nanmean(r_bg)) else 1 - p_w / 2
            else:
                p_adj = p_w
            out["wilcoxon_p"] = float(max(min(p_adj, 1.0), 0.0))

        # permutation test
        if use_permutation:
            k = len(r_set)
            obs = float(np.nanmean(r_set) - np.nanmean(r_bg))

            perm_diffs = np.empty(n_permutations, dtype=float)
            bg_vals = r_bg[~np.isnan(r_bg)]
            if len(bg_vals) < k:
                raise ValueError("len(Background gene)< len(gene_set).")

            for i in range(n_permutations):
                sample = self.rng.choice(bg_vals, size=k, replace=False)
                perm_diffs[i] = float(np.mean(sample) - np.mean(bg_vals))

            if alternative == "greater":
                p_perm = float(np.mean(perm_diffs >= obs))
            elif alternative == "less":
                p_perm = float(np.mean(perm_diffs <= obs))
            else:  # two-sided
                p_perm = float(np.mean(np.abs(perm_diffs) >= abs(obs)))
            out["perm_p"] = max(min(p_perm, 1.0), 0.0)

        return out

    
    # ------------------ results ------------------
    def get_top_genes_by_group(
        self,
        topk: int = 10,
        gene_set: Optional[Sequence[str]] = None,
        direction: str = "pos",   # {"pos","neg","abs"} 
        ) -> pd.DataFrame:
        """
        Returns
        -------
        pd.DataFrame columns:
        [group, gene, r, q, rank]
        """
        if self._r_all is None:
            self.compute_all_correlations()
        if self.groupby is None:
            raise ValueError("groupby is None")
        if self._adata is None or self._target_gene is None:
            raise RuntimeError("Need to call fit() first")

        if gene_set is None:
            genes = list(self._r_all.index)
        else:
            genes = [g for g in gene_set if g in self._r_all.index]
            if not genes:
                raise ValueError("No valid gene in gene_set.")

        df = self.compute_correlations_by_group(genes)

        if direction == "pos":
            key = "r"; ascending = False
        elif direction == "neg":
            key = "r"; ascending = True
        elif direction == "abs":
            df = df.assign(abs_r=np.abs(df["r"]))
            key = "abs_r"; ascending = False
        else:
            raise ValueError("direction must be one of {'pos','neg','abs'}")

        rows = []
        for grp, sub in df.groupby(self.groupby):
            sub = sub.sort_values(key, ascending=ascending).head(topk).copy()
            sub["rank"] = np.arange(1, len(sub) + 1, dtype=int)
            rows.append(sub[[self.groupby, "gene", "r","q", "rank"]])
        out = pd.concat(rows, axis=0).reset_index(drop=True)
        return out
    # ------------------ plots ------------------
    
    ## Top in all genes
    def plot_top_overall(self, topk: int = 20, ax=None, figsize=(4, 6)):
        if self._r_all is None:
            self.compute_all_correlations()
        df = self._r_all.sort_values(by=["r", "q"], ascending=[False, True]).head(topk)
        df = df[::-1]
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        ax.barh(df.index, df["r"].values)
        ax.set_xlabel("Pearson r")
        ax.set_title(f"Top {topk} correlations with {self._target_gene}")
        ax.grid(False)
        return ax
    
    
    ## Top in geneset
    def plot_top_in_gene_set(
        self,
        gene_set: Sequence[str],
        topk: int = 20,
        ax=None,
        figsize=(4, 6),
        alpha_by_q: bool = True
        ):
        if self._r_all is None:
            self.compute_all_correlations()

        genes = [g for g in gene_set if g in self._r_all.index]
        df = self._r_all.loc[genes].sort_values(by=["r", "q"], ascending=[False, True]).head(topk)

        df = df[::-1] 

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        if alpha_by_q:
            # q → -log10(q) --> 0~1 normalization --> alpha
            q_log = -np.log10(df["q"].clip(lower=1e-300))
            q_norm = (q_log - q_log.min()) / (q_log.max() - q_log.min() + 1e-12)
            alphas = q_norm.values
        else:
            alphas = np.ones_like(df["r"].values)

        for i, (gene, r, alpha) in enumerate(zip(df.index, df["r"], alphas)):
            ax.barh(gene, r, alpha=alpha, color="tab:blue")

        ax.set_xlabel("Pearson r")
        ax.set_title(f"Top {topk} in gene set with {self._target_gene}")
        ax.grid(False)

        return 
        
    def plot_top_genes_by_group(
        self,
        topk: int = 10,
        gene_set: Optional[Sequence[str]] = None,
        direction: str = "pos",   # {"pos","neg","abs"}
        n_cols: int = 4,
        figsize_unit: Tuple[float, float] = (4.0, 3.2), 
        ):

        topdf = self.get_top_genes_by_group(topk=topk, gene_set=gene_set, direction=direction)

        groups = topdf[self.groupby].unique().tolist()
        n_groups = len(groups)
        n_rows = (n_groups + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols,
                                figsize=(figsize_unit[0]*n_cols, figsize_unit[1]*n_rows),
                                squeeze=False)
        axes = axes.flatten()

        for i, grp in enumerate(groups):
            ax = axes[i]
            sub = topdf[topdf[self.groupby] == grp].copy()
            
            sub = sub.sort_values("rank", ascending=False)
            ax.barh(sub["gene"], sub["r"])
            ax.set_title(f"{grp} (top {len(sub)})", fontsize=10)
            ax.set_xlabel("Pearson r")
            ax.grid(False)

        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])

        mode = {"pos":"Top positive", "neg":"Top negative", "abs":"Top |r|"}[direction]
        if gene_set is None:
            subtitle = "all genes"
        else:
            subtitle = f"{len([g for g in (gene_set or []) if g in self._r_all.index])} geneset"
        fig.suptitle(f"{mode} correlations by {self.groupby} (target: {self._target_gene}, {subtitle})",
                    y=1.02, fontsize=12)
        fig.tight_layout()
        return fig, axes
    
    
    ## Distribution
    def plot_compare_distributions(
        self,
        gene_set,
        ax=None,
        figsize=(6, 4),
        bins: int = 40,
        log: bool = False,  
        set_label: str = "Gene set",
        bg_label: str = "Background",
        set_color: str = "red",
        bg_color: str = "gray",
        show_stats: bool = True,
        set_name: Optional[str] = None,
        eps: float = 1e-12,   
        common_range: tuple = (-1.0, 1.0),  
        stats: Optional[dict] = None,  
        alternative: str = "two-sided", 
        ):
        if self._r_all is None:
            self.compute_all_correlations()

        genes = [g for g in gene_set if g in self._r_all.index]
        r_set = self._r_all.loc[genes, "r"].to_numpy() if len(genes) else np.array([])
        r_bg  = self._r_all.loc[~self._r_all.index.isin(genes), "r"].to_numpy()

        r_set = r_set[np.isfinite(r_set)]
        r_bg  = r_bg[np.isfinite(r_bg)]

        bin_edges = np.linspace(common_range[0], common_range[1], bins + 1)
        cnt_bg, _  = np.histogram(r_bg,  bins=bin_edges)
        cnt_set, _ = np.histogram(r_set, bins=bin_edges)

        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        width = (bin_edges[1] - bin_edges[0]) * 0.9

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        if log:
            y_bg  = np.where(cnt_bg  > 0, np.log10(cnt_bg),  np.nan)
            y_set = np.where(cnt_set > 0, np.log10(cnt_set), np.nan)
            ax.set_ylabel("log10(Count)")
        else:
            y_bg, y_set = cnt_bg, cnt_set
            ax.set_ylabel("Count")

        ax.bar(bin_centers, y_bg, width=width, alpha=0.5, label=bg_label,color=bg_color, zorder=1, edgecolor=None)
        ax.bar(bin_centers, y_set, width=width, alpha=0.8, label=set_label,color=set_color, zorder=3, edgecolor="black", linewidth=1.0)

        if r_bg.size:
            ax.axvline(np.nanmean(r_bg), linestyle="--", color="black",label=f"{bg_label} mean", linewidth=1.2, zorder=4)
        if r_set.size:
            ax.axvline(np.nanmean(r_set), linestyle="-", color=set_color,label=f"{set_label} mean", linewidth=2.0, zorder=5)

        ax.set_xlabel("Pearson r")
        set_name = set_name or "Gene set"
        ax.set_title(f"r distribution: {set_name} vs background ({self._target_gene})")
        ax.legend(frameon=False)
        ax.grid(False)
        
        if show_stats and r_set.size > 1 and r_bg.size > 1:
            if stats is None:
                stats = self.test_gene_set_significance(
                    gene_set=gene_set, alternative=alternative,
                    use_ttest=True, use_wilcoxon=True, use_permutation=True
                )
            text_lines = [
                f"mean(set)={stats['mean_set']:.3f}, \nmean(bg)={stats['mean_bg']:.3f}",
            ]
            if 'ttest_p' in stats:
                text_lines.append(f"t-test p={stats['ttest_p']:.2e}")
            if 'wilcoxon_p' in stats:
                text_lines.append(f"Wilcoxon p={stats['wilcoxon_p']:.2e}")
            if 'perm_p' in stats:
                text_lines.append(f"Perm p={stats['perm_p']:.2e}")
            ax.text(0.05, 0.95, "\n".join(text_lines), transform=ax.transAxes,
                    fontsize=10, va="top", ha="left",
                    bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.5))
        return ax
        
    def plot_compare_distributions_by_group(
        self,
        gene_set: Sequence[str],
        figsize: Tuple[float, float] = (12, 4),
        bins: int = 30,
        log: bool = False,
        set_color: str = "red",
        bg_color: str = "gray",
        set_name: Optional[str] = None,
        sharex: bool = True,
        sharey: bool = True,
        show_stats: bool = True,
        ):
        if self._adata is None or self._target_gene is None:
            raise RuntimeError("import fit().")
        if self.groupby is None:
            raise ValueError("No groupby value.")
        if self._r_all is None:
            self.compute_all_correlations()

        df_group_corr = self.compute_correlations_by_group(gene_set)
        if df_group_corr.empty:
            raise ValueError("No valid gene in gene_set.")
        bg_genes = [g for g in self._r_all.index if g not in gene_set]
        df_all_group_corr = self.compute_correlations_by_group(bg_genes)

        df_stats = self.get_groupwise_stats(gene_set) if show_stats else None
        q_map = dict(zip(df_stats[self.groupby], df_stats["q_value"])) if df_stats is not None else {}

        set_name = set_name or "Gene set"
        groups = sorted(df_group_corr[self.groupby].unique())
        n_groups = len(groups)

        n_cols = 5
        n_rows = (n_groups + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1] * n_rows),
                                sharex=sharex, sharey=sharey)
        axes = axes.flatten()

        for i, grp in enumerate(groups):
            ax = axes[i]
            r_set = df_group_corr[df_group_corr[self.groupby] == grp]["r"].dropna().values
            r_bg  = df_all_group_corr[df_all_group_corr[self.groupby] == grp]["r"].dropna().values

            ax.hist(r_bg, bins=bins, color=bg_color, alpha=0.5, label="Background", zorder=1)
            ax.hist(r_set, bins=bins, color=set_color, alpha=0.8, edgecolor="black", label=set_name, zorder=3)
            if r_bg.size:
                ax.axvline(np.nanmean(r_bg), color="black", linestyle="--", linewidth=1.2, label="BG mean")
            if r_set.size:
                ax.axvline(np.nanmean(r_set), color=set_color, linestyle="-", linewidth=2.0, label="Set mean")

            title = f"{grp} ({len(r_set)} genes)"
            if show_stats and grp in q_map and np.isfinite(q_map[grp]):
                ax.text(0.05, 0.95, f"FDR q = {q_map[grp]:.2e}", transform=ax.transAxes,
                        fontsize=9, va="top", ha="left",
                        bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.5))
            ax.set_title(title)
            ax.set_xlabel("Pearson r")
            if log:
                ax.set_yscale("log")
            ax.grid(False)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        axes[0].set_ylabel("Count" + (" (log scale)" if log else ""))
        fig.suptitle(f"{set_name} Pearson r distribution by {self.groupby} (vs {self._target_gene})")
        fig.tight_layout()
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        return fig, axes

    def plot_distribution_with_filtered_top10(
        self,
        fdr_threshold: float = 0.01,
        r_threshold: float = 0.7,
        direction: str = "abs",  # {"abs", "pos", "neg"}
        topk: int = 10,
        gene_set: Optional[Sequence[str]] = None,
        n_cols: int = 2,
        figsize_unit: Tuple[float, float] = (5.0, 4.0),
        bins: int = 30,
        log: bool = False,
        set_color: str = "red",
        bg_color: str = "gray",
        show_stats: bool = True,
    ):
        """
        Parameters
        ----------
        fdr_threshold : float
            FDR q-value filter threshold (default: 0.01)
        r_threshold : float
            corr. coefficient threshold (default: 0.7)
        direction : str
            - "abs" : |r| > r_threshold 
            - "pos" : r > r_threshold 
            - "neg" : r < -r_threshold 
        topk : int
        gene_set : Optional[Sequence[str]]
            if None, use all genes
        n_cols : int
        figsize_unit : Tuple[float, float]
        bins : int
        log : bool
        set_color : str
        bg_color : str
        show_stats : bool
            
        Returns
        -------
        fig, axes : matplotlib figure and axes
            (n_rows, n_cols) 형태의 axes 배열
        """
        if self._adata is None or self._target_gene is None:
            raise RuntimeError("Call fit(adata, target_gene) first.")
        if self.groupby is None:
            raise ValueError("groupby is None")
        if self._r_all is None:
            self.compute_all_correlations()


        if gene_set is None:
            genes_to_use = list(self._r_all.index)
        else:
            genes_to_use = [g for g in gene_set if g in self._r_all.index]
            if not genes_to_use:
                raise ValueError("No valid genes in gene_set.")

  
        df_group_corr = self.compute_correlations_by_group(genes_to_use)
        
        if df_group_corr.empty:
            raise ValueError("No valid correlations computed.")
        
        groups = sorted(df_group_corr[self.groupby].unique())
        n_groups = len(groups)
        

        n_rows = (n_groups + n_cols - 1) // n_cols
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(figsize_unit[0] * n_cols, figsize_unit[1] * n_rows),
            squeeze=False
        )
        axes = axes.flatten()
        
        for idx, grp in enumerate(groups):
            ax = axes[idx]
            
            
            grp_data = df_group_corr[df_group_corr[self.groupby] == grp].copy()
            r_vals = grp_data["r"].dropna().values
            q_vals = grp_data["q"].dropna().values
            
            
            if direction == "abs":
                mask = (np.abs(grp_data["r"]) > r_threshold) & (grp_data["q"] < fdr_threshold)
                direction_label = f"|r| > {r_threshold}"
            elif direction == "pos":
                mask = (grp_data["r"] > r_threshold) & (grp_data["q"] < fdr_threshold)
                direction_label = f"r > {r_threshold}"
            elif direction == "neg":
                mask = (grp_data["r"] < -r_threshold) & (grp_data["q"] < fdr_threshold)
                direction_label = f"r < {-r_threshold}"
            else:
                raise ValueError("direction must be one of {'abs', 'pos', 'neg'}")
            
            filtered_genes = grp_data[mask].copy()
            n_filtered = len(filtered_genes)
            
            
            r_all = r_vals[np.isfinite(r_vals)]
            r_filtered = filtered_genes["r"].dropna().values if n_filtered > 0 else np.array([])
            r_bg = np.setdiff1d(r_all, r_filtered) 
            
            bin_edges = np.linspace(-1.0, 1.0, bins + 1)
            cnt_bg, _ = np.histogram(r_bg, bins=bin_edges)
            cnt_set, _ = np.histogram(r_filtered, bins=bin_edges)
            
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            width = (bin_edges[1] - bin_edges[0]) * 0.9
            
            if log:
                y_bg = np.where(cnt_bg > 0, np.log10(cnt_bg), np.nan)
                y_set = np.where(cnt_set > 0, np.log10(cnt_set), np.nan)
                ax.set_ylabel("log10(Count)")
            else:
                y_bg, y_set = cnt_bg, cnt_set
                ax.set_ylabel("Count")
            
            ax.bar(bin_centers, y_bg, width=width, alpha=0.5, label="Not filtered",
                   color=bg_color, zorder=1, edgecolor=None)
            ax.bar(bin_centers, y_set, width=width, alpha=0.8, label="Filtered",
                   color=set_color, zorder=3, edgecolor="black", linewidth=1.0)
            

            if direction == "abs":

                ax.axvline(r_threshold, linestyle="--", color="darkred", 
                           label=f"Threshold: +{r_threshold}", linewidth=2.0, zorder=5)
                ax.axvline(-r_threshold, linestyle="--", color="darkred", 
                           label=f"Threshold: -{r_threshold}", linewidth=2.0, zorder=5)
            elif direction == "pos":
                ax.axvline(r_threshold, linestyle="--", color="darkred", 
                           label=f"Threshold: {r_threshold}", linewidth=2.0, zorder=5)
            elif direction == "neg":
                ax.axvline(-r_threshold, linestyle="--", color="darkred", 
                           label=f"Threshold: {-r_threshold}", linewidth=2.0, zorder=5)
            
            ax.set_xlabel("Pearson r")
            

            title_text = f"{grp}\nFiltered: {n_filtered} genes\n({direction_label}, FDR<{fdr_threshold})"
            ax.set_title(title_text, fontsize=10)
            ax.legend(fontsize=8, frameon=False, loc="upper right")
            ax.grid(False)
            

            if show_stats and n_filtered > 0:
                if direction == "abs":
                    mean_val = np.mean(np.abs(r_filtered))
                    median_val = np.median(np.abs(r_filtered))
                else:
                    mean_val = np.mean(r_filtered)
                    median_val = np.median(r_filtered)
                
                stats_text = (
                    f"n={n_filtered}\n"
                    f"mean={mean_val:.3f}\n"
                    f"median={median_val:.3f}"
                )
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                       fontsize=8, va="top", ha="left",
                       bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.7))
        

        for j in range(idx + 1, len(axes)):
            fig.delaxes(axes[j])
        
        fig.suptitle(
            f"Filtered Genes by Group (Target: {self._target_gene})",
            fontsize=13, y=0.995
        )
        fig.tight_layout()
        
        return fig, axes
