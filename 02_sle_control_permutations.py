import os
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
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

H5AD_PATH = "data/adata_cohort1.h5ad"
ZSCORE_LAYER = "zscores_4andhalf"
GROUP_COL = "group"
CONTROL_LABEL = "healthy_control"
DONOR_COL = "unique_patient_id"

OUTPUT_DIR_NEW = "results/permutation_full/sle_vs_hc"

DEFAULT_N_PERMS = 100000
DEFAULT_N_JOBS = min(16, os.cpu_count() or 1)
DEFAULT_CHECKPOINT_FREQ = 1000
DEFAULT_SEED_BASE = 42

MAX_ITER = 25
TOL = 1e-8
PROB_CLIP = 1e-10

P_THRESHOLDS = [0.001, 0.005, 0.01, 0.05, 0.10]

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

def process_peptide_chunk(chunk_idx, peptide_indices, X_chunk, Y,
                          perm_start, perm_end, seed_base):
    n_perms = perm_end - perm_start
    n_peptides_chunk = X_chunk.shape[1]

    zscores = np.zeros((n_peptides_chunk, n_perms), dtype=np.float32)

    for i, perm_idx in enumerate(range(perm_start, perm_end)):

        np.random.seed(seed_base + perm_idx)
        Y_perm = np.random.permutation(Y)

        zscores[:, i] = vectorized_logistic_regression(X_chunk, Y_perm)

    return chunk_idx, peptide_indices, zscores

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
    print("LOADING DATA")
    print("=" * 70)
    print_memory_status("before load")

    print("\n[1] Loading AnnData...")
    adata = sc.read_h5ad(H5AD_PATH)
    print(f"    Shape: {adata.shape}")
    print_memory_status("after AnnData")

    print("[2] Extracting z-scores...")
    zscores = adata.layers[ZSCORE_LAYER]
    if issparse(zscores):
        zscores = zscores.toarray()

    zscore_df = pd.DataFrame(
        zscores,
        index=adata.obs_names,
        columns=adata.var_names
    )
    zscore_df[DONOR_COL] = adata.obs[DONOR_COL].values
    zscore_df[GROUP_COL] = adata.obs[GROUP_COL].values

    del adata
    print_memory_status("after zscore extraction")

    print("[3] Aggregating to donor level...")
    peptide_cols = list(zscore_df.columns[:-2])

    donor_has_case = zscore_df.groupby(DONOR_COL).apply(
        lambda x: (x[GROUP_COL] != CONTROL_LABEL).any()
    )
    donor_zscores = zscore_df.groupby(DONOR_COL)[peptide_cols].mean()

    del zscore_df

    common_donors = donor_zscores.index.intersection(donor_has_case.index)
    donor_zscores = donor_zscores.loc[common_donors]
    is_case = donor_has_case.loc[common_donors].astype(int)

    X = donor_zscores.values.astype(np.float32)
    Y = is_case.values.astype(np.int32)
    peptide_ids = donor_zscores.columns.tolist()

    del donor_zscores

    print(f"    X: {X.shape}, Y: {Y.shape}")
    print(f"    Cases: {(Y==1).sum()}, Controls: {(Y==0).sum()}")
    print(f"    X memory: {X.nbytes / 1e9:.2f} GB")
    print_memory_status("after aggregation")

    print("[4] Loading peptide-gene mapping...")
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

    return {
        'X': X,
        'Y': Y,
        'peptide_ids': peptide_ids,
        'genes_list': genes_list,
        'gene_to_peptide_idx': gene_to_peptide_idx,
        'peptide_mapping': peptide_mapping
    }

def get_checkpoint_path(output_dir):
    return os.path.join(output_dir, 'checkpoint.json')

def save_checkpoint(output_dir, completed_perms, total_perms, start_time):
    checkpoint = {
        'completed_perms': completed_perms,
        'total_perms': total_perms,
        'start_time': start_time,
        'last_update': datetime.now().isoformat(),
        'version': 'peptide_parallel_v2'
    }
    with open(get_checkpoint_path(output_dir), 'w') as f:
        json.dump(checkpoint, f, indent=2)

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

def run_permutation_peptide_parallel(data, n_perms, n_jobs, checkpoint_freq,
                                     output_dir, seed_base=DEFAULT_SEED_BASE,
                                     start_perm=0):
    X = data['X']
    Y = data['Y']
    genes_list = data['genes_list']
    peptide_ids = data['peptide_ids']
    gene_to_peptide_idx = data['gene_to_peptide_idx']

    n_donors, n_peptides = X.shape
    n_genes = len(genes_list)

    print("\n" + "=" * 70)
    print("PERMUTATION TEST CONFIGURATION (PEPTIDE-PARALLEL)")
    print("=" * 70)
    print(f"  Donors: {n_donors}")
    print(f"  Peptides: {n_peptides:,}")
    print(f"  Genes: {n_genes:,}")
    print(f"  Total Permutations: {n_perms:,}")
    print(f"  Starting from: {start_perm}")
    print(f"  Parallel jobs (peptide chunks): {n_jobs}")
    print(f"  Checkpoint frequency: {checkpoint_freq}")
    print(f"  Seed base: {seed_base}")
    print(f"  Output directory: {output_dir}")

    chunk_size = (n_peptides + n_jobs - 1) // n_jobs
    actual_chunks = (n_peptides + chunk_size - 1) // chunk_size
    chunk_memory = n_donors * chunk_size * 4 / 1e6

    print(f"\n  Peptide chunk optimization:")
    print(f"    Peptides per chunk: ~{chunk_size:,}")
    print(f"    Chunk memory: ~{chunk_memory:.1f} MB (target: <32 MB for L3)")
    print(f"    Total chunks: {actual_chunks}")

    print_memory_status("before permutations")

    peptide_chunks = []
    for i in range(n_jobs):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, n_peptides)
        if start_idx < n_peptides:
            peptide_chunks.append(np.arange(start_idx, end_idx, dtype=np.int32))

    print(f"    Actual chunks created: {len(peptide_chunks)}")

    if start_perm == 0:
        for fname in ['peptide_zscores.h5', 'gene_statistics.h5']:
            path = os.path.join(output_dir, fname)
            if os.path.exists(path):
                os.remove(path)

    print("\n" + "=" * 70)
    print("COMPUTING OBSERVED STATISTICS")
    print("=" * 70)

    print("\n[1] Running IRLS on observed data...")
    obs_start = time.time()
    obs_zscores = vectorized_logistic_regression(X, Y)
    obs_time = time.time() - obs_start
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
    print("RUNNING PERMUTATIONS (PEPTIDE-PARALLEL)")
    print("=" * 70)

    total_start_time = time.time()
    perm_completed = start_perm

    while perm_completed < n_perms:
        block_start = perm_completed
        block_end = min(perm_completed + checkpoint_freq, n_perms)
        block_size = block_end - block_start

        print(f"\n--- Block {block_start}-{block_end} ({block_size} permutations) ---")
        block_start_time = time.time()

        print(f"    Running {len(peptide_chunks)} peptide chunks in parallel...")
        print(f"    Each chunk processes {block_size} permutations")
        print_memory_status("before parallel")

        results = Parallel(n_jobs=len(peptide_chunks), verbose=10)(
            delayed(process_peptide_chunk)(
                chunk_idx=i,
                peptide_indices=chunk,
                X_chunk=X[:, chunk].copy(),
                Y=Y,
                perm_start=block_start,
                perm_end=block_end,
                seed_base=seed_base
            )
            for i, chunk in enumerate(peptide_chunks)
        )

        print_memory_status("after parallel")

        print(f"    Reassembling results...")
        block_zscores = np.zeros((n_peptides, block_size), dtype=np.float32)
        for chunk_idx, peptide_indices, chunk_zscores in results:
            block_zscores[peptide_indices, :] = chunk_zscores

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
        perms_per_sec = block_size / block_time
        remaining_perms = n_perms - perm_completed
        remaining_time = remaining_perms / perms_per_sec if perms_per_sec > 0 else 0

        print(f"    Block time: {block_time/60:.1f} min ({perms_per_sec:.2f} perms/sec)")
        print(f"    Progress: {perm_completed}/{n_perms} ({100*perm_completed/n_perms:.1f}%)")
        print(f"    Elapsed: {elapsed/60:.1f} min, ETA: {remaining_time/60:.1f} min")
        print_memory_status("end of block")

    total_time = time.time() - total_start_time
    print(f"\n{'='*70}")
    print(f"PERMUTATIONS COMPLETE")
    print(f"{'='*70}")
    print(f"  Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"  Average per permutation: {total_time/(n_perms - start_perm):.3f}s")

    return obs_gene_stats

def compute_final_pvalues(output_dir, genes_list, obs_gene_stats):
    print("\n" + "=" * 70)
    print("COMPUTING PERMUTATION P-VALUES")
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

            n_sig_005 = (perm_pval < 0.05).sum()
            n_sig_001 = (perm_pval < 0.01).sum()
            n_sig_0001 = (perm_pval < 0.001).sum()
            print(f"    Significant: {n_sig_0001} (p<0.001), {n_sig_001} (p<0.01), {n_sig_005} (p<0.05)")

    results_df = pd.DataFrame(results)

    try:
        from scipy.stats import false_discovery_control
        for stat_name in ['min_p', 'truncsum_0050', 'truncsum_0010']:
            pval_col = f'{stat_name}_perm_p'
            if pval_col in results_df.columns:
                pvals = np.clip(results_df[pval_col].values, 1e-10, 1)
                qvals = false_discovery_control(pvals, method='bh')
                results_df[f'{stat_name}_qval'] = qvals
    except ImportError:
        print("  Note: scipy.stats.false_discovery_control not available, skipping FDR")

    results_df.to_csv(os.path.join(output_dir, 'gene_permutation_pvalues.csv'), index=False)
    print(f"\n  Saved: gene_permutation_pvalues.csv")

    return results_df

def main():
    parser = argparse.ArgumentParser(
        description='PhIP-seq SLE Permutation Analysis (Peptide-Parallel Optimization)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--n-perms', type=int, default=DEFAULT_N_PERMS,
                       help=f'Total number of permutations (default: {DEFAULT_N_PERMS})')
    parser.add_argument('--n-jobs', type=int, default=DEFAULT_N_JOBS,
                       help=f'Number of parallel jobs/peptide chunks (default: {DEFAULT_N_JOBS})')
    parser.add_argument('--checkpoint-freq', type=int, default=DEFAULT_CHECKPOINT_FREQ,
                       help=f'Checkpoint frequency (default: {DEFAULT_CHECKPOINT_FREQ})')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR_NEW,
                       help=f'Output directory (default: {OUTPUT_DIR_NEW})')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from checkpoint in output directory')
    parser.add_argument('--resume-from', type=int, default=None,
                       help='Start from specific permutation index (e.g., 10000 to continue from 10K)')
    parser.add_argument('--seed-base', type=int, default=DEFAULT_SEED_BASE,
                       help=f'Base seed for reproducibility (default: {DEFAULT_SEED_BASE})')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("PhIP-seq SLE Permutation Analysis (Peptide-Parallel Optimization)")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {args.output_dir}")
    print_memory_status("startup")

    data = load_data()

    start_perm = 0

    if args.resume_from is not None:
        start_perm = args.resume_from
        print(f"\nResuming from permutation {start_perm} (user-specified)")

    elif args.resume:
        checkpoint = load_checkpoint(args.output_dir)
        if checkpoint:
            start_perm = checkpoint['completed_perms']
            print(f"\nResuming from permutation {start_perm} (from checkpoint)")
        else:
            print("\nNo checkpoint found, starting fresh")

    obs_gene_stats = run_permutation_peptide_parallel(
        data,
        n_perms=args.n_perms,
        n_jobs=args.n_jobs,
        checkpoint_freq=args.checkpoint_freq,
        output_dir=args.output_dir,
        seed_base=args.seed_base,
        start_perm=start_perm
    )

    results_df = compute_final_pvalues(
        args.output_dir,
        data['genes_list'],
        obs_gene_stats
    )

    metadata = {
        'n_perms': int(args.n_perms),
        'n_jobs': int(args.n_jobs),
        'n_peptides': int(len(data['peptide_ids'])),
        'n_genes': int(len(data['genes_list'])),
        'n_donors': int(data['X'].shape[0]),
        'n_cases': int((data['Y'] == 1).sum()),
        'n_controls': int((data['Y'] == 0).sum()),
        'seed_base': int(args.seed_base),
        'algorithm': 'peptide_parallel_v2',
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
    print("  - gene_permutation_pvalues.csv  (main results)")
    print("  - peptide_zscores.h5            (for custom analyses)")
    print("  - gene_statistics.h5            (pre-computed stats)")

if __name__ == '__main__':
    main()
