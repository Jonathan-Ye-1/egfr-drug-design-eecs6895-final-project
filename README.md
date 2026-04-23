# Target-Conditioned De Novo 3D Drug Design for EGFR Inhibitors

**EECS 6895 Final Project** | Spring 2026

An end-to-end AI-assisted drug discovery pipeline using the pretrained TargetDiff diffusion model
to generate novel 3D molecules conditioned on EGFR binding pockets (wild-type 1M17 and T790M
mutant 4I22), followed by multi-criteria evaluation and comparison with clinical drugs.

## Pipeline Overview

```
PDB Input (1M17 / 4I22)
  → Pocket Extraction (10 Å)
  → TargetDiff Conditional Generation (1000 mol/target)
  → RDKit Validity Filter
  → AutoDock Vina Scoring
  → QED / SA / Lipinski Evaluation
  → Multi-criteria Ranking → Top-10 Candidates
  → PyMOL / py3Dmol Visualization
```

## Targets

| Target | PDB ID | Description |
|--------|--------|-------------|
| EGFR Wild-Type | 1M17 | EGFR + Erlotinib co-crystal, 2.6 Å |
| EGFR T790M | 4I22 | T790M resistance mutant |

## Quick Start

```bash
# 1. Clone this repo
git clone https://github.com/<YOUR_USERNAME>/egfr_diffusion.git
cd egfr_diffusion

# 2. Create conda environment
conda create -n egfr python=3.10
conda activate egfr

# 3. Install dependencies
pip install -r requirements.txt
conda install -c conda-forge openbabel  # for PDBQT conversion

# 4. Clone TargetDiff
git clone https://github.com/guanjq/targetdiff.git external/targetdiff

# 5. Download pretrained checkpoint (manual step)
# From: https://drive.google.com/file/d/1_BUWcHMQLbvOPbU4aYiDYcvF_0VEPjPZ
# Save to: external/targetdiff/checkpoints/pretrained_diffusion.pt

# 6. Start notebooks
jupyter lab notebooks/
```

## Notebook Guide

| Notebook | Day | Task |
|----------|-----|------|
| 01_setup_and_test | Day 1 | Environment check, TargetDiff clone & checkpoint |
| 02_prepare_targets | Day 2 | Download PDB, extract pockets, baseline conformers |
| 03_vina_baseline | Day 3 | Receptor prep, baseline drug Vina scoring |
| 04_generate_wildtype | Day 4 | TargetDiff generation on 1M17 |
| 05_evaluate_wildtype | Day 5–6 | Validity/QED/SA/Vina evaluation for WT |
| 06_generate_mutant | Day 8 | TargetDiff generation on 4I22 (T790M) |
| 07_evaluate_mutant | Day 9 | Evaluation + UMAP chemical space |
| 08_final_analysis | Day 10–11 | Top-10 refinement, all publication figures |

## Evaluation Metrics

| Metric | Tool | Target |
|--------|------|--------|
| Validity | RDKit | > 80% |
| Uniqueness | RDKit SMILES | > 90% |
| Diversity | Tanimoto distance | > 0.7 |
| Vina Score | AutoDock Vina | ≤ −8 kcal/mol |
| QED | RDKit | ≥ 0.5 |
| SA Score | RDKit (Ertl 2009) | ≤ 4.0 |
| Lipinski Ro5 | Custom | Pass all 4 |

## References

1. Guan et al., *3D Equivariant Diffusion for Target-Aware Molecule Generation*, ICLR 2023.
2. Trott & Olson, *AutoDock Vina*, J. Comput. Chem. 2010.
3. Bickerton et al., *Quantifying the chemical beauty of drugs*, Nature Chemistry 2012.
4. Ertl & Schuffenhauer, *Estimation of synthetic accessibility score*, J. Cheminformatics 2009.
