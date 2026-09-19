# SLE PhIP-seq manuscript analyses

Analysis code for the computational figure panels of the SLE PhIP-seq manuscript, including upstream calculations and external-cohort evaluation.

| Figure | Analysis scripts |
|---|---|
| Figure 1b | [06_figure1b_volcano.ipynb](06_figure1b_volcano.ipynb) |
| Figure 1c | [07_figure1c_ucsf_classification.ipynb](07_figure1c_ucsf_classification.ipynb), [09_evaluate_nyu_sle_control.ipynb](09_evaluate_nyu_sle_control.ipynb), [10_evaluate_itn_sle_control.ipynb](10_evaluate_itn_sle_control.ipynb), [11_figure1c_external_roc.ipynb](11_figure1c_external_roc.ipynb) |
| Figure 2a | [14_figure2a_heatmap.ipynb](14_figure2a_heatmap.ipynb) |
| Figure 2b/c | [15_figure2b_peptide_correlation.ipynb](15_figure2b_peptide_correlation.ipynb), [16_figure2c_rbp_enrichment.ipynb](16_figure2c_rbp_enrichment.ipynb) |
| Figure 3a | [17_figure3a_ucsf_activity.ipynb](17_figure3a_ucsf_activity.ipynb), [18_figure3a_external_activity.ipynb](18_figure3a_external_activity.ipynb) |
| Figure 3b | [19_figure3b_heatmap.ipynb](19_figure3b_heatmap.ipynb) |
| Figure 3c | [20_figure3c_activity_volcano.ipynb](20_figure3c_activity_volcano.ipynb) |
| Supplementary Figure 1a–c | [12_supplementary1a_autoreactivity.ipynb](12_supplementary1a_autoreactivity.ipynb), [13_supplementary1bc_feature_count.ipynb](13_supplementary1bc_feature_count.ipynb) |
| Supplementary Figure 3 | [21_supplementary3_picnic_permutation.py](21_supplementary3_picnic_permutation.py) |

Upstream calculations: [01_donor_peptide_odds_ratios.ipynb](01_donor_peptide_odds_ratios.ipynb), [02_sle_control_permutations.py](02_sle_control_permutations.py), [03_sle_control_gene_summary.ipynb](03_sle_control_gene_summary.ipynb), [04_activity_permutations.py](04_activity_permutations.py), [05_activity_gene_summary.ipynb](05_activity_gene_summary.ipynb), [08_train_external_validation_model.ipynb](08_train_external_validation_model.ipynb).

## Input data

The three annotation files are publicly available and not redistributed here:

- `mmc2.xlsx`: Table S2 (RNA-binding proteome) of Trendel J. *et al.* The human RNA-binding proteome and its dynamics during translational arrest. *Cell* **176**, 391–403.e19 (2019), https://doi.org/10.1016/j.cell.2018.11.004, as downloaded from the article's supplemental information. 
- `PICNIC-9606-data.csv`: PICNIC condensate scores for the human proteome (*Homo sapiens*, NCBI taxonomy 9606), downloaded from the PICNIC web server, https://picnic.cd-code.org (Browse, species *Homo sapiens*, Download). PICNIC is described in Hadarovich A. *et al.* PICNIC accurately predicts condensate-forming proteins regardless of their structural disorder across organisms. *Nat. Commun.* **15**, 10668 (2024), https://doi.org/10.1038/s41467-024-55089-x.
- `HUMAN_9606_idmapping.dat.gz`: UniProt ID mapping for human, https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/idmapping/by_organism/HUMAN_9606_idmapping.dat.gz. The analysis used UniProt release 2025_03 (18 June 2025). 

## Environment

Run notebooks and scripts from this directory using Python 3.9.20 and the packages in [requirements.txt](requirements.txt).
