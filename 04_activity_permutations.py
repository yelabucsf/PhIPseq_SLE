import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit
from scipy.sparse import issparse
import scanpy as sc
import h5py
import time
import json
import psutil
from datetime import datetime
from joblib import Parallel, delayed, dump, load
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

H5AD_PATH = "data/adata_cohort1.h5ad"

ZSCORE_LAYER = "zscores_4andhalf"
DONOR_COL = "unique_patient_id"
SLEDAI_COL = "sledai_score"

SLEDAI_HIGH_THRESHOLD = 11
SLEDAI_LOW_VALUE = 0

OUTPUT_DIR = "results/permutation_full/high_vs_low_mean"

DEFAULT_N_PERMS = 100000
DEFAULT_N_JOBS = min(16, os.cpu_count() or 1)
DEFAULT_CHECKPOINT_FREQ = 2000
DEFAULT_BATCH_SIZE = 10

MAX_ITER = 25
TOL = 1e-8
PROB_CLIP = 1e-10

P_THRESHOLDS = [0.001, 0.005, 0.01, 0.05, 0.10]

KEY_GENES = ["SNRNP70", "YTHDF2", "SSB", "NPIPB15", "CD2", "RPLP0", "RPLP1", "RPLP2", "C1QA", "C1QB"]

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1e9

def print_memory_status(label=""):
    mem_gb = get_memory_usage()
    total_gb = psutil.virtual_memory().total / 1e9
    used_gb = psutil.virtual_memory().used / 1e9
    print(f"  Memory {label}: Process={mem_gb:.1f}GB, System={used_gb:.1f}/{total_gb:.1f}GB")

def vectorized_logistic_regression(X, Y, max_iter=MAX_ITER, tol=TOL):
    n_samples, n_features = X.shape

    p_mean = Y.mean()
    beta0 = np.full(n_features, np.log(p_mean / (1 - p_mean)), dtype=np.float32)
    beta1 = np.zeros(n_features, dtype=np.float32)
    Y_col = Y.reshape(-1, 1).astype(np.float32)

    for _ in range(max_iter):
        eta = beta0 + beta1 * X
        eta_clipped = np.clip(eta, -500, 500)
        p = expit(eta_clipped)
        p = np.clip(p, PROB_CLIP, 1 - PROB_CLIP)

        W = p * (1 - p)
        r = Y_col - p

        H00 = W.sum(axis=0)
        H01 = (W * X).sum(axis=0)
        H11 = (W * X * X).sum(axis=0)

        U0 = r.sum(axis=0)
        U1 = (r * X).sum(axis=0)

        det = H00 * H11 - H01 * H01
        det = np.where(np.abs(det) < 1e-10, 1e-10, det)

        delta_beta0 = np.clip((H11 * U0 - H01 * U1) / det, -10, 10)
        delta_beta1 = np.clip((H00 * U1 - H01 * U0) / det, -10, 10)

        beta0 = beta0 + delta_beta0
        beta1 = beta1 + delta_beta1

        if np.max(np.maximum(np.abs(delta_beta0), np.abs(delta_beta1))) < tol:
            break

    eta = beta0 + beta1 * X
    p = expit(np.clip(eta, -500, 500))
    p = np.clip(p, PROB_CLIP, 1 - PROB_CLIP)
    W = p * (1 - p)

    H00 = W.sum(axis=0)
    H01 = (W * X).sum(axis=0)
    H11 = (W * X * X).sum(axis=0)
    det = H00 * H11 - H01 * H01
    det = np.where(np.abs(det) < 1e-10, 1e-10, det)

    var_beta1 = H00 / det
    var_beta1 = np.where(var_beta1 < 0, np.nan, var_beta1)
    se_beta1 = np.sqrt(var_beta1)

    z_scores = (beta1 / se_beta1).astype(np.float32)

    return z_scores

def run_permutation_batch_mmap(X_path, Y, perm_indices, seed_base):
    X = load(X_path, mmap_mode='r')

    n_peptides = X.shape[1]
    batch_size = len(perm_indices)
    batch_zscores = np.zeros((n_peptides, batch_size), dtype=np.float32)

    for i, perm_idx in enumerate(perm_indices):
        np.random.seed(seed_base + perm_idx)
        Y_perm = np.random.permutation(Y)
        batch_zscores[:, i] = vectorized_logistic_regression(X, Y_perm)

    return batch_zscores

def compute_gene_statistics_from_zscores(z_scores, gene_to_peptide_idx, genes_list,
                                          thresholds=P_THRESHOLDS):
    if z_scores.ndim == 1:
        z_scores = z_scores.reshape(-1, 1)

    n_peptides, n_cols = z_scores.shape
    n_genes = len(genes_list)

    p_values = 2 * stats.norm.sf(np.abs(z_scores))

    stats_dict = {
        'min_p': np.zeros((n_genes, n_cols), dtype=np.float32),
        'top5_sum': np.zeros((n_genes, n_cols), dtype=np.float32),
        'fisher': np.zeros((n_genes, n_cols), dtype=np.float32),
        'cauchy': np.zeros((n_genes, n_cols), dtype=np.float32),
        'max_absZ': np.zeros((n_genes, n_cols), dtype=np.float32),
    }

    for thresh in thresholds:
        thresh_str = f"{thresh:.3f}".replace('.', '')
        stats_dict[f'truncsum_{thresh_str}'] = np.zeros((n_genes, n_cols), dtype=np.float32)
        stats_dict[f'count_{thresh_str}'] = np.zeros((n_genes, n_cols), dtype=np.int16)

    for i, gene in enumerate(genes_list):
        idx = gene_to_peptide_idx[gene]
        gene_pvals = p_values[idx, :]
        gene_zscores = z_scores[idx, :]

        valid_mask = ~np.isnan(gene_pvals)
        gene_pvals_masked = np.where(valid_mask, gene_pvals, 1.0)

        stats_dict['min_p'][i, :] = gene_pvals_masked.min(axis=0)

        gene_z_masked = np.where(valid_mask, np.abs(gene_zscores), 0.0)
        stats_dict['max_absZ'][i, :] = gene_z_masked.max(axis=0)

        for thresh in thresholds:
            thresh_str = f"{thresh:.3f}".replace('.', '')
            sig_mask = (gene_pvals < thresh) & valid_mask
            stats_dict[f'count_{thresh_str}'][i, :] = sig_mask.sum(axis=0)
            neglog_p = -np.log10(np.clip(gene_pvals, 1e-300, 1))
            truncsum = np.where(sig_mask, neglog_p, 0.0).sum(axis=0)
            stats_dict[f'truncsum_{thresh_str}'][i, :] = truncsum

        neglog_p_all = -np.log10(np.clip(gene_pvals_masked, 1e-300, 1))
        sorted_neglog = np.sort(neglog_p_all, axis=0)[::-1, :]
        top5 = sorted_neglog[:min(5, len(idx)), :].sum(axis=0)
        stats_dict['top5_sum'][i, :] = top5

        log_p = np.log(np.clip(gene_pvals_masked, 1e-300, 1))
        stats_dict['fisher'][i, :] = -2 * log_p.sum(axis=0)

        cauchy_vals = np.tan((0.5 - gene_pvals_masked) * np.pi)
        cauchy_vals = np.clip(cauchy_vals, -1e10, 1e10)
        stats_dict['cauchy'][i, :] = np.nanmean(cauchy_vals, axis=0)

    if n_cols == 1:
        stats_dict = {k: v.squeeze() for k, v in stats_dict.items()}

    return stats_dict

def load_peptide_mapping():
    adata = sc.read_h5ad(H5AD_PATH, backed="r")
    mapping = pd.DataFrame({"seq_id": adata.var_names.astype(str),
                            "gene": adata.var["gene"].astype(str).values})
    adata.file.close()
    return mapping

def load_data():
    print("=" * 70)
    print("LOADING DATA (HIGH vs LOW SLEDAI - MEAN AGGREGATION)")
    print("=" * 70)
    print_memory_status("before load")

    print("\n[1] Loading AnnData...")
    adata = sc.read_h5ad(H5AD_PATH)
    print(f"    Shape: {adata.shape}")
    print_memory_status("after AnnData")

    if SLEDAI_COL not in adata.obs.columns:
        raise ValueError(f"SLEDAI column '{SLEDAI_COL}' not found in adata.obs")

    obs_df = adata.obs.copy()

    print("\n[2] Filtering to HIGH and LOW SLEDAI...")
    high_mask = obs_df[SLEDAI_COL] >= SLEDAI_HIGH_THRESHOLD
    low_mask = obs_df[SLEDAI_COL] == SLEDAI_LOW_VALUE

    high_samples = obs_df[high_mask]
    low_samples = obs_df[low_mask]

    donors_high = set(high_samples[DONOR_COL].unique())
    donors_low = set(low_samples[DONOR_COL].unique())
    donors_both = donors_high & donors_low

    print(f"    HIGH SLEDAI (≥{SLEDAI_HIGH_THRESHOLD}): {len(high_samples)} samples, {len(donors_high)} donors")
    print(f"    LOW SLEDAI (={SLEDAI_LOW_VALUE}): {len(low_samples)} samples, {len(donors_low)} donors")
    print(f"    Overlapping donors (EXCLUDED): {len(donors_both)}")

    donors_high_only = donors_high - donors_both
    donors_low_only = donors_low - donors_both

    keep_mask = (
        (high_mask & obs_df[DONOR_COL].isin(donors_high_only)) |
        (low_mask & obs_df[DONOR_COL].isin(donors_low_only))
    )

    adata_filtered = adata[keep_mask].copy()
    adata_filtered.obs['is_high'] = (adata_filtered.obs[SLEDAI_COL] >= SLEDAI_HIGH_THRESHOLD).astype(int)

    n_high_samples = (adata_filtered.obs['is_high'] == 1).sum()
    n_low_samples = (adata_filtered.obs['is_high'] == 0).sum()
    n_high_donors = len(donors_high_only)
    n_low_donors = len(donors_low_only)

    print(f"\n    After excluding overlap:")
    print(f"      HIGH: {n_high_samples} samples from {n_high_donors} donors")
    print(f"      LOW:  {n_low_samples} samples from {n_low_donors} donors")

    print("\n[3] Extracting z-scores...")
    zscores = adata_filtered.layers[ZSCORE_LAYER]
    if issparse(zscores):
        zscores = zscores.toarray()

    peptide_ids = adata_filtered.var_names.tolist()

    print("\n[4] Aggregating to donor level (MEAN across samples)...")

    zscore_df = pd.DataFrame(
        zscores,
        index=adata_filtered.obs_names,
        columns=peptide_ids
    )
    zscore_df[DONOR_COL] = adata_filtered.obs[DONOR_COL].values
    zscore_df['is_high'] = adata_filtered.obs['is_high'].values

    del adata, adata_filtered
    print_memory_status("after zscore extraction")

    peptide_cols = peptide_ids

    donor_is_high = zscore_df.groupby(DONOR_COL)['is_high'].first()
    donor_zscores = zscore_df.groupby(DONOR_COL)[peptide_cols].mean()

    del zscore_df

    common_donors = donor_zscores.index.intersection(donor_is_high.index)
    donor_zscores = donor_zscores.loc[common_donors]
    is_case = donor_is_high.loc[common_donors].astype(int)

    X = donor_zscores.values.astype(np.float32)
    Y = is_case.values.astype(np.int32)

    del donor_zscores

    print(f"    X: {X.shape}, Y: {Y.shape}")
    print(f"    HIGH donors: {(Y==1).sum()}, LOW donors: {(Y==0).sum()}")
    print(f"    X memory: {X.nbytes / 1e9:.2f} GB")
    print_memory_status("after aggregation")

    print("\n[5] Loading peptide-gene mapping...")
    peptide_mapping = load_peptide_mapping()
    peptide_mapping = peptide_mapping[peptide_mapping['seq_id'].isin(peptide_ids)]

    peptide_to_idx = {p: i for i, p in enumerate(peptide_ids)}
    gene_to_peptide_idx = {}
    genes_list = []

    for gene, group in peptide_mapping.groupby('gene'):
        indices = [peptide_to_idx[p] for p in group['seq_id'] if p in peptide_to_idx]
        if len(indices) > 0:
            gene_to_peptide_idx[gene] = np.array(indices, dtype=np.int32)
            genes_list.append(gene)

    print(f"    Genes: {len(genes_list)}")
    print_memory_status("final")

    print(f"\n[6] Key genes status:")
    for gene in KEY_GENES:
        if gene in gene_to_peptide_idx:
            n_pep = len(gene_to_peptide_idx[gene])
            print(f"    {gene}: {n_pep} peptides")
        else:
            print(f"    {gene}: NOT FOUND")

    return {
        'X': X,
        'Y': Y,
        'peptide_ids': peptide_ids,
        'genes_list': genes_list,
        'gene_to_peptide_idx': gene_to_peptide_idx,
        'peptide_mapping': peptide_mapping,
        'n_high_donors': int((Y == 1).sum()),
        'n_low_donors': int((Y == 0).sum()),
    }

def get_checkpoint_path(output_dir):
    return os.path.join(output_dir, 'checkpoint.json')

def save_checkpoint(output_dir, completed_perms, total_perms, start_time):
    checkpoint = {
        'completed_perms': completed_perms,
        'total_perms': total_perms,
        'start_time': start_time,
        'last_update': datetime.now().isoformat()
    }
    with open(get_checkpoint_path(output_dir), 'w') as f:
        json.dump(checkpoint, f)

def load_checkpoint(output_dir):
    path = get_checkpoint_path(output_dir)
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None

def save_permutation_results(output_dir, perm_zscores, perm_gene_stats,
                              start_idx, end_idx, genes_list, peptide_ids):

    zscores_path = os.path.join(output_dir, 'peptide_zscores.h5')

    with h5py.File(zscores_path, 'a') as f:
        if 'zscores' not in f:
            n_peptides = len(peptide_ids)
            f.create_dataset('zscores',
                           shape=(n_peptides, 0),
                           maxshape=(n_peptides, None),
                           dtype='float16',
                           chunks=(min(10000, n_peptides), 100),
                           compression='gzip',
                           compression_opts=4)
            f.create_dataset('peptide_ids',
                           data=np.array(peptide_ids, dtype='S'))

        dset = f['zscores']
        current_size = dset.shape[1]
        new_size = current_size + (end_idx - start_idx)
        dset.resize((dset.shape[0], new_size))
        dset[:, current_size:new_size] = perm_zscores.astype(np.float16)

    gene_stats_path = os.path.join(output_dir, 'gene_statistics.h5')

    with h5py.File(gene_stats_path, 'a') as f:
        if 'genes' not in f:
            f.create_dataset('genes', data=np.array(genes_list, dtype='S'))

        n_genes = len(genes_list)
        batch_size = end_idx - start_idx

        for stat_name, stat_values in perm_gene_stats.items():
            if stat_name not in f:
                dtype = 'int16' if 'count' in stat_name else 'float32'
                f.create_dataset(stat_name,
                               shape=(n_genes, 0),
                               maxshape=(n_genes, None),
                               dtype=dtype,
                               chunks=(n_genes, 100),
                               compression='gzip',
                               compression_opts=4)

            dset = f[stat_name]
            current_size = dset.shape[1]
            new_size = current_size + batch_size
            dset.resize((dset.shape[0], new_size))
            dset[:, current_size:new_size] = stat_values

def run_permutation_parallel(data, n_perms, n_jobs, checkpoint_freq, batch_size,
                              output_dir, resume=False):
    X = data['X']
    Y = data['Y']
    genes_list = data['genes_list']
    peptide_ids = data['peptide_ids']
    gene_to_peptide_idx = data['gene_to_peptide_idx']

    n_donors, n_peptides = X.shape
    n_genes = len(genes_list)

    print("\n" + "=" * 70)
    print("PERMUTATION TEST CONFIGURATION")
    print("=" * 70)
    print(f"  Donors: {n_donors} (HIGH: {data['n_high_donors']}, LOW: {data['n_low_donors']})")
    print(f"  Peptides: {n_peptides:,}")
    print(f"  Genes: {n_genes:,}")
    print(f"  Permutations: {n_perms:,}")
    print(f"  Parallel jobs: {n_jobs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Checkpoint frequency: {checkpoint_freq}")
    print(f"  Output directory: {output_dir}")
    print_memory_status("before permutations")

    X_path = os.path.join(output_dir, 'X_mmap.joblib')
    if not os.path.exists(X_path):
        print(f"\n  Saving X for memory-mapped access...")
        dump(X, X_path)
        print(f"  Saved: {X_path} ({os.path.getsize(X_path) / 1e9:.2f} GB)")
    else:
        print(f"\n  Using existing mmap file: {X_path}")

    start_perm = 0
    if resume:
        checkpoint = load_checkpoint(output_dir)
        if checkpoint:
            start_perm = checkpoint['completed_perms']
            print(f"\n  Resuming from permutation {start_perm}")
        else:
            print("\n  No checkpoint found, starting fresh")

    if start_perm == 0:
        for fname in ['peptide_zscores.h5', 'gene_statistics.h5']:
            path = os.path.join(output_dir, fname)
            if os.path.exists(path):
                os.remove(path)

    print("\n" + "=" * 70)
    print("COMPUTING OBSERVED STATISTICS")
    print("=" * 70)

    print("\n[1] Running IRLS on observed data...")
    start_time = time.time()
    obs_zscores = vectorized_logistic_regression(X, Y)
    obs_time = time.time() - start_time
    print(f"    Time: {obs_time:.1f}s")

    print("[2] Computing gene-level statistics...")
    obs_gene_stats = compute_gene_statistics_from_zscores(
        obs_zscores, gene_to_peptide_idx, genes_list
    )

    obs_df = pd.DataFrame({
        'peptide': peptide_ids,
        'z_score': obs_zscores,
        'p_value': 2 * stats.norm.sf(np.abs(obs_zscores))
    })
    obs_df.to_csv(os.path.join(output_dir, 'observed_peptide_results.csv'), index=False)

    obs_gene_df = pd.DataFrame({'gene': genes_list})
    for stat_name, stat_values in obs_gene_stats.items():
        obs_gene_df[f'{stat_name}_obs'] = stat_values
    obs_gene_df.to_csv(os.path.join(output_dir, 'observed_gene_stats.csv'), index=False)

    print(f"    Saved observed results")
    print_memory_status("after observed stats")

    print("\n" + "=" * 70)
    print("RUNNING PERMUTATIONS")
    print("=" * 70)

    total_start_time = time.time()
    perm_completed = start_perm

    while perm_completed < n_perms:
        block_start = perm_completed
        block_end = min(perm_completed + checkpoint_freq, n_perms)
        block_size = block_end - block_start

        print(f"\n--- Block {block_start}-{block_end} ({block_size} permutations) ---")
        block_start_time = time.time()

        n_batches = (block_size + batch_size - 1) // batch_size
        batch_ranges = []
        for i in range(n_batches):
            batch_start_idx = block_start + i * batch_size
            batch_end_idx = min(batch_start_idx + batch_size, block_end)
            batch_ranges.append((batch_start_idx, batch_end_idx))

        print(f"    Running {n_batches} batches with {n_jobs} workers...")
        print_memory_status("before parallel")

        results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(run_permutation_batch_mmap)(
                X_path, Y, list(range(bs, be)), seed_base=42
            ) for bs, be in batch_ranges
        )

        print_memory_status("after parallel")

        block_zscores = np.concatenate(results, axis=1)

        print(f"    Computing gene-level statistics...")
        block_gene_stats = compute_gene_statistics_from_zscores(
            block_zscores, gene_to_peptide_idx, genes_list
        )

        print(f"    Saving results...")
        save_permutation_results(
            output_dir, block_zscores, block_gene_stats,
            block_start, block_end, genes_list, peptide_ids
        )

        perm_completed = block_end
        save_checkpoint(output_dir, perm_completed, n_perms, total_start_time)

        block_time = time.time() - block_start_time
        elapsed = time.time() - total_start_time
        remaining = (n_perms - perm_completed) / block_size * block_time if block_size > 0 else 0

        print(f"    Block time: {block_time/60:.1f} min")
        print(f"    Progress: {perm_completed}/{n_perms} ({100*perm_completed/n_perms:.1f}%)")
        print(f"    Elapsed: {elapsed/60:.1f} min, ETA: {remaining/60:.1f} min")
        print_memory_status("end of block")

    total_time = time.time() - total_start_time
    print(f"\n{'='*70}")
    print(f"PERMUTATIONS COMPLETE")
    print(f"{'='*70}")
    print(f"  Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"  Average per permutation: {total_time/n_perms:.2f}s")

    return obs_gene_stats

def compute_permutation_pvalues(output_dir, genes_list, obs_gene_stats):
    print("\n" + "=" * 70)
    print("COMPUTING PERMUTATION P-VALUES (ALL GENES)")
    print("=" * 70)

    n_genes = len(genes_list)
    gene_stats_path = os.path.join(output_dir, 'gene_statistics.h5')

    with h5py.File(gene_stats_path, 'r') as f:
        stat_names = [k for k in f.keys() if k != 'genes']
        n_perms = f[stat_names[0]].shape[1]
        print(f"  Found {n_perms} permutations")

        results = {'gene': genes_list}

        for stat_name in stat_names:
            print(f"  Processing {stat_name}...")

            perm_vals = f[stat_name][:]
            obs_vals = obs_gene_stats[stat_name]

            if stat_name == 'min_p':
                n_extreme = (perm_vals <= obs_vals.reshape(-1, 1)).sum(axis=1)
            else:
                n_extreme = (perm_vals >= obs_vals.reshape(-1, 1)).sum(axis=1)

            perm_pval = (n_extreme + 1) / (n_perms + 1)

            results[f'{stat_name}_obs'] = obs_vals
            results[f'{stat_name}_perm_p'] = perm_pval

    results_df = pd.DataFrame(results)

    results_df.to_csv(os.path.join(output_dir, 'gene_permutation_pvalues_raw.csv'), index=False)
    print(f"\n  Saved: gene_permutation_pvalues_raw.csv (NO K filtering, NO FDR)")

    return results_df

def apply_k_filtering_and_fdr(output_dir, K=5, zscore_threshold=4.5):
    print("\n" + "=" * 70)
    print(f"APPLYING K FILTERING (K≥{K}) AND FDR")
    print("=" * 70)

    raw_pvals_path = os.path.join(output_dir, 'gene_permutation_pvalues_raw.csv')
    if not os.path.exists(raw_pvals_path):
        print(f"  ERROR: {raw_pvals_path} not found. Run permutations first.")
        return None

    results_df = pd.read_csv(raw_pvals_path)
    print(f"  Loaded {len(results_df)} genes")

    obs_peptide_path = os.path.join(output_dir, 'observed_peptide_results.csv')
    obs_peptide = pd.read_csv(obs_peptide_path)

    peptide_mapping = load_peptide_mapping()

    obs_peptide = obs_peptide.merge(
        peptide_mapping[['seq_id', 'gene']],
        left_on='peptide', right_on='seq_id', how='left'
    )

    X_path = os.path.join(output_dir, 'X_mmap.joblib')

    print(f"\n  Loading data to compute K filtering...")
    data = load_data()
    X = data['X']
    Y = data['Y']
    peptide_ids = data['peptide_ids']
    gene_to_peptide_idx = data['gene_to_peptide_idx']
    genes_list = data['genes_list']

    print(f"  Computing high-reactivity donor counts per gene...")

    high_reactivity = (X >= zscore_threshold)

    gene_n_high_total = []
    gene_n_high_case = []
    gene_n_high_ctrl = []

    for gene in genes_list:
        idx = gene_to_peptide_idx[gene]

        gene_high = high_reactivity[:, idx].any(axis=1)

        n_high_case = (gene_high & (Y == 1)).sum()
        n_high_ctrl = (gene_high & (Y == 0)).sum()
        n_high_total = n_high_case + n_high_ctrl

        gene_n_high_total.append(n_high_total)
        gene_n_high_case.append(n_high_case)
        gene_n_high_ctrl.append(n_high_ctrl)

    results_df['n_high_total'] = gene_n_high_total
    results_df['n_high_case'] = gene_n_high_case
    results_df['n_high_ctrl'] = gene_n_high_ctrl

    print(f"\n  Genes before K filter: {len(results_df)}")
    filtered_df = results_df[results_df['n_high_total'] >= K].copy()
    print(f"  Genes after K≥{K} filter: {len(filtered_df)}")

    print(f"\n  Applying BH FDR correction...")

    stat_cols = [c.replace('_perm_p', '') for c in filtered_df.columns if '_perm_p' in c]

    for stat_name in stat_cols:
        pval_col = f'{stat_name}_perm_p'
        qval_col = f'{stat_name}_qval'

        pvals = filtered_df[pval_col].values
        mask = np.isfinite(pvals)

        if mask.sum() > 0:
            _, qvals, _, _ = multipletests(pvals[mask], method='fdr_bh')
            filtered_df[qval_col] = np.nan
            filtered_df.loc[filtered_df.index[mask], qval_col] = qvals

            n_sig_01 = (filtered_df[qval_col] <= 0.1).sum()
            n_sig_05 = (filtered_df[qval_col] <= 0.05).sum()
            print(f"    {stat_name}: q≤0.1: {n_sig_01}, q≤0.05: {n_sig_05}")

    output_path = os.path.join(output_dir, f'gene_permutation_pvalues_K{K}.csv')
    filtered_df.to_csv(output_path, index=False)
    print(f"\n  Saved: {output_path}")

    print(f"\n  Key genes (K≥{K}):")
    for gene in KEY_GENES:
        row = filtered_df[filtered_df['gene'] == gene]
        if len(row) > 0:
            row = row.iloc[0]
            truncsum_p = row.get('truncsum_0010_perm_p', np.nan)
            truncsum_q = row.get('truncsum_0010_qval', np.nan)
            n_high = row.get('n_high_total', 0)
            print(f"    {gene}: n_high={n_high}, truncsum_0010 p={truncsum_p:.4f}, q={truncsum_q:.4f}")
        else:

            raw_row = results_df[results_df['gene'] == gene]
            if len(raw_row) > 0:
                n_high = raw_row.iloc[0].get('n_high_total', 0)
                print(f"    {gene}: FILTERED OUT (n_high={n_high} < K={K})")
            else:
                print(f"    {gene}: NOT FOUND")

    return filtered_df

def main():
    parser = argparse.ArgumentParser(description='HIGH vs LOW SLEDAI Permutation (MEAN aggregation)')
    parser.add_argument('--n-perms', type=int, default=DEFAULT_N_PERMS,
                       help=f'Number of permutations (default: {DEFAULT_N_PERMS})')
    parser.add_argument('--n-jobs', type=int, default=DEFAULT_N_JOBS,
                       help=f'Number of parallel jobs (default: {DEFAULT_N_JOBS})')
    parser.add_argument('--checkpoint-freq', type=int, default=DEFAULT_CHECKPOINT_FREQ,
                       help=f'Checkpoint frequency (default: {DEFAULT_CHECKPOINT_FREQ})')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                       help=f'Batch size (default: {DEFAULT_BATCH_SIZE})')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR,
                       help=f'Output directory (default: {OUTPUT_DIR})')
    parser.add_argument('--k-filter', type=int, default=None,
                       help='Apply K filtering and FDR only (skip permutations)')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("HIGH vs LOW SLEDAI Permutation Analysis (MEAN Aggregation)")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_memory_status("startup")

    if args.k_filter is not None:
        apply_k_filtering_and_fdr(args.output_dir, K=args.k_filter)
        return 0

    data = load_data()

    obs_gene_stats = run_permutation_parallel(
        data,
        n_perms=args.n_perms,
        n_jobs=args.n_jobs,
        checkpoint_freq=args.checkpoint_freq,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        resume=args.resume,
    )

    results_df = compute_permutation_pvalues(
        args.output_dir,
        data['genes_list'],
        obs_gene_stats
    )

    for K in [3, 5]:
        print(f"\n--- Applying K≥{K} filtering ---")
        apply_k_filtering_and_fdr(args.output_dir, K=K)

    metadata = {
        'analysis': 'HIGH_vs_LOW_SLEDAI',
        'aggregation': 'MEAN',
        'n_perms': args.n_perms,
        'n_peptides': len(data['peptide_ids']),
        'n_genes': len(data['genes_list']),
        'n_donors': data['X'].shape[0],
        'n_high_donors': data['n_high_donors'],
        'n_low_donors': data['n_low_donors'],
        'sledai_high_threshold': SLEDAI_HIGH_THRESHOLD,
        'sledai_low_value': SLEDAI_LOW_VALUE,
        'completed': datetime.now().isoformat()
    }
    with open(os.path.join(args.output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to: {args.output_dir}")
    print("\nKey output files:")
    print("  - gene_permutation_pvalues_raw.csv   (all genes, no FDR)")
    print("  - gene_permutation_pvalues_K3.csv    (K≥3 filtered + FDR)")
    print("  - gene_permutation_pvalues_K5.csv    (K≥5 filtered + FDR)")
    print("  - peptide_zscores.h5                 (for custom analyses)")
    print("  - gene_statistics.h5                 (pre-computed stats)")
    print("\nTo re-run FDR with different K:")
    print(f"  python {sys.argv[0]} --output-dir {args.output_dir} --k-filter 4")

if __name__ == '__main__':
    main()
