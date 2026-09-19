import os

import json

import time

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

sle_dir = "results"

perm_dir = f"{sle_dir}/permutation_full"

slehc_path = f"{perm_dir}/sle_vs_hc/slehc_gene_level_clean_summary.csv"

highlow_path = f"{perm_dir}/high_vs_low_mean/highlow_mean_gene_level_clean_summary.csv"

picnic_path = "data/PICNIC-9606-data.csv"

out_dir = f"{sle_dir}/picnic_permutation"

fig_dir = f"{sle_dir}/fig"

os.makedirs(out_dir, exist_ok=True)

os.makedirs(fig_dir, exist_ok=True)

QVAL_THRESHOLD = 0.1

PICNIC_SCORE_THRESHOLD = 0.5

RANDOM_STATE = 42

N_PERM_PRIMARY = 100_000

COL_BACKGROUND = '#888888'

COL_ALL_HITS = '#D55E00'

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def load_picnic():
    log("Loading PICNIC...")
    picnic = pd.read_csv(picnic_path)
    picnic = picnic.rename(columns={
        'PICNIC score': 'picnic_score', 'PICNIC GO score': 'picnic_go_score',
        'Uniprot ID': 'uniprot_id', 'Genes': 'genes_raw',
        'CD-Code included': 'cdcode_included',
    })
    picnic['picnic_go_score'] = pd.to_numeric(picnic['picnic_go_score'], errors='coerce')

    n_dup_acc_rows = picnic['uniprot_id'].duplicated(keep=False).sum()
    if n_dup_acc_rows:
        is_self = picnic['ID'] == picnic['uniprot_id']
        picnic = pd.concat([
            picnic[~picnic['uniprot_id'].duplicated(keep=False)],
            picnic[picnic['uniprot_id'].duplicated(keep=False) & is_self],
        ]).reset_index(drop=True)
        log(f"  Dropped {n_dup_acc_rows - picnic['uniprot_id'].duplicated(keep=False).sum()} "
            f"duplicate-accession rows from raw PICNIC file (kept ID==uniprot_id row, not max score)")

    picnic_accs = set(picnic['uniprot_id'])
    n_cond = (picnic['picnic_score'] >= PICNIC_SCORE_THRESHOLD).sum()
    log(f"  {len(picnic):,} reviewed human proteins in PICNIC, "
        f"{n_cond} ({100*n_cond/len(picnic):.1f}%) score >= {PICNIC_SCORE_THRESHOLD} (base rate)")

    log("  Building official Gene_Name map (idmapping.dat.gz, restricted to PICNIC accessions)...")
    import gzip
    rows = []
    with gzip.open('data/HUMAN_9606_idmapping.dat.gz', 'rt') as f:
        for line in f:
            acc, key, val = line.rstrip('\n').split('\t')
            if key == 'Gene_Name' and acc in picnic_accs:
                rows.append((acc, val.upper()))
    idmap = pd.DataFrame(rows, columns=['uniprot_id', 'gene_upper']).drop_duplicates()

    n_no_gene_name = len(picnic_accs) - idmap['uniprot_id'].nunique()

    per_gene_acc_count = idmap.groupby('gene_upper')['uniprot_id'].nunique()
    ambiguous_genes = set(per_gene_acc_count[per_gene_acc_count > 1].index)
    idmap_unambig = idmap[~idmap['gene_upper'].isin(ambiguous_genes)]

    gene_to_picnic = idmap_unambig.merge(picnic, on='uniprot_id', how='left')

    log(f"  {idmap['uniprot_id'].nunique():,}/{len(picnic_accs):,} PICNIC accessions have an official "
        f"Gene_Name ({n_no_gene_name} do not, likely uncharacterized ORFs)")
    log(f"  {len(ambiguous_genes)} gene symbols are the official name of >1 reviewed accession "
        f"(real ambiguity, excluded from match rather than tie-broken by score)")

    picnic_meta = dict(n_reviewed_proteins=len(picnic), base_rate_ge_0p5=float(n_cond / len(picnic)),
                        n_accs_no_gene_name=int(n_no_gene_name),
                        n_ambiguous_gene_symbols=len(ambiguous_genes),
                        ambiguous_genes=sorted(ambiguous_genes))
    return picnic, gene_to_picnic, picnic_meta

def build_gene_table(slehc_df, highlow_df):
    log("Building gene-level analysis table...")

    picnic, gene_to_picnic, picnic_meta = load_picnic()
    universe = sorted(set(slehc_df['gene']) & set(highlow_df['gene']))
    gt = pd.DataFrame({'gene': universe})
    gt['gene_upper'] = gt['gene'].str.upper()

    gt = gt.merge(gene_to_picnic[['gene_upper', 'picnic_score', 'picnic_go_score',
                                   'cdcode_included', 'uniprot_id']],
                   on='gene_upper', how='left')
    n_matched = gt['picnic_score'].notna().sum()
    log(f"  PICNIC merge: {n_matched:,}/{len(gt):,} ({100*n_matched/len(gt):.1f}%) matched")

    sle_cols = slehc_df[['gene', 'z_score', 'fisher_qval', 'fisher_perm_p', 'log_or', 'odds_ratio']].rename(
        columns={c: f'{c}_slehc' for c in ['z_score', 'fisher_qval', 'fisher_perm_p', 'log_or', 'odds_ratio']})
    hl_cols = highlow_df[['gene', 'z_score', 'fisher_qval', 'fisher_perm_p', 'log_or', 'odds_ratio',
                          'best_peptide']].rename(
        columns={**{c: f'{c}_highlow' for c in ['z_score', 'fisher_qval', 'fisher_perm_p', 'log_or', 'odds_ratio']},
                 'best_peptide': 'best_peptide_highlow'})
    gt = gt.merge(sle_cols, on='gene', how='left').merge(hl_cols, on='gene', how='left')

    return gt

def phase7_permutation_null(gene_table, n_perm=N_PERM_PRIMARY, random_state=RANDOM_STATE):
    log("=" * 70)
    log(f"Uniform permutation null, n_perm={n_perm}")
    log("=" * 70)
    matched = gene_table[gene_table['picnic_score'].notna()].copy()
    sig = matched[(matched['fisher_qval_highlow'] <= QVAL_THRESHOLD) & (matched['z_score_highlow'] > 0)]
    N = len(sig)
    observed_median = sig['picnic_score'].median()
    log(f"  N hits = {N}, observed median PICNIC = {observed_median:.4f}")

    rng = np.random.RandomState(random_state)
    scores = matched['picnic_score'].values

    null_medians_uniform = np.empty(n_perm)
    for i in range(n_perm):
        draw = rng.choice(scores, size=N, replace=False)
        null_medians_uniform[i] = np.median(draw)
    p_uniform = (null_medians_uniform >= observed_median).mean()

    summary = dict(N_hits=N, observed_median=observed_median,
                    null_uniform_median=float(np.median(null_medians_uniform)),
                    null_uniform_p=float(p_uniform))
    return summary, null_medians_uniform

def _style_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#444444')
    ax.spines['bottom'].set_color('#444444')
    ax.tick_params(colors='#444444')

def _save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(f"{fig_dir}/picnic_{name}.{ext}", dpi=200, bbox_inches='tight')
    plt.close(fig)
    log(f"  Saved fig/picnic_{name}.pdf/png")

def fig_permutation_null_uniform_only(perm_summary, null_uniform):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.hist(null_uniform, bins=60, color=COL_BACKGROUND, alpha=0.8)
    ax.axvline(perm_summary['observed_median'], color=COL_ALL_HITS, lw=2)
    ax.text(perm_summary['observed_median'], ax.get_ylim()[1] * 0.9,
            f'  observed\n  p={perm_summary["null_uniform_p"]:.2g}',
            color=COL_ALL_HITS, fontsize=9)
    ax.set_xlabel('median PICNIC score')
    ax.set_ylabel('permutation count')
    ax.set_title(f'Uniform random draws (N={perm_summary["N_hits"]} High-vs-No hits, '
                  f'{N_PERM_PRIMARY:,} draws)', fontsize=10)
    _style_axes(ax)
    _save(fig, 'permutation_null_uniform')

slehc_df = pd.read_csv(slehc_path)

highlow_df = pd.read_csv(highlow_path)

gene_table = build_gene_table(slehc_df, highlow_df)

perm_summary, null_uniform = phase7_permutation_null(gene_table)

with open(f"{out_dir}/phase7_permutation_summary.json", 'w') as f:
        json.dump(perm_summary, f, indent=2)

np.save(f"{out_dir}/phase7_null_uniform.npy", null_uniform)

fig_permutation_null_uniform_only(perm_summary, null_uniform)
