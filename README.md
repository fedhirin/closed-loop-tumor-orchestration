# An autonomous multimodal agent for closed-loop tumor burden–efficacy–toxicity assessment and adaptive treatment orchestration in real-world lung cancer management

This repository contains the training and evaluation implementation for longitudinal non-small cell lung cancer treatment orchestration. Four modality encoders transform CT volumes, genomic profiles, structured clinical events, and pharmacological graphs into a shared token space. Treatment-conditioned hierarchical cross-attention fuses these representations across treatment cycles, while a causal temporal Transformer and recurrent memory model the evolving patient state. A constrained policy balances response and toxicity and defers high-uncertainty decisions for clinical review.

## Repository map

The Python package is under `code/tumor_orchestration`. `configurations/main.yaml` records the primary experiment and `configurations/validation.yaml` provides a reduced configuration for local pipeline validation. `datasets.txt` is the verified access and license index. Runtime outputs belong under `runs` and aggregate evaluation artifacts under `artifacts`.

## Environment

The primary environment uses Python 3.10, PyTorch 2.1.2, CUDA 12.1, cuDNN 8, NumPy 1.26.3, SciPy 1.11.4, scikit-learn 1.4.0, and pandas 2.1.4.

Install with pip:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

Install with conda:

```bash
conda env create -f environment.yml
conda activate tumor-orchestration
python -m pip install --no-deps -e .
```

Build the container:

```bash
docker build -t tumor-orchestration:1.0 .
```

## Data

The study uses NSCLC-Radiomics, NSCLC-Radiogenomics, LIDC-IDRI, TCGA-LUAD, TCGA-LUSC, GENIE BPC NSCLC 2.0, MIMIC-IV 3.1, DrugBank 5.1.16, and the SEER November 2024 submission. Canonical links and access conditions are listed in `datasets.txt`.

MIMIC-IV requires credentialing, the data use agreement, and the required human-subjects training. GDC open data remain subject to the NIH Genomic Data Sharing Policy; controlled files require dbGaP authorization. DrugBank files require an account and a license appropriate to the intended use. SEER data require acceptance of the SEER Research Data Agreement. The preparation pipeline never redistributes source records.

CT volumes are resampled to 1 mm isotropic spacing, clipped to the lung window from −1024 to 400 HU, scaled to [0, 1], and cropped to 96 × 96 × 96 around the largest lesion. RNA-seq features are log2 transformed, quantile normalized, and reduced to 1,536 variance-ranked genes; 468 mutation indicators are retained and mapped to pathway tokens. Clinical events are filtered for ICD-10 C34.x, aligned by treatment cycle, imputed with five chained-equation estimates, and represented using 87 structured variables. Drug graphs use 2,048-dimensional molecular fingerprints and interactions with confidence above 0.7.

Each local dataset manifest must contain relative filenames, byte sizes, and SHA-256 digests. The data preparation functions in `tumor_orchestration.data` build and verify these manifests without exposing patient identifiers outside the authorized storage boundary.

## Training

The primary run uses four NVIDIA A100 80 GB GPUs:

```bash
torchrun --standalone --nproc_per_node=4 -m tumor_orchestration.commands --config configurations/main.yaml --data datasets --output runs/main
```

The configuration uses a per-device batch size of 16, no gradient accumulation, and an effective batch size of 64. Phase one optimizes the supervised multi-task objective for 100 epochs with AdamW, learning rate 1e-4, weight decay 1e-3, five warmup epochs, cosine decay, FP16, and early-stopping patience of 15. Phase two refines the policy for 50 PPO episodes with learning rate 3e-5 and clip ratio 0.2. The reported configuration was run over 20 seeds.

The loss is the response binary cross-entropy plus weighted Cox survival, multi-label toxicity, Dice segmentation, PPO, and cross-modal alignment terms. Their weights are 0.5, 0.8, 0.3, 0.1, and 0.2. The safety reward uses efficacy weight 1.0, toxicity weight 0.8, violation penalty 100, and toxicity threshold 0.7. Policy uncertainty is estimated with 30 dropout passes, with escalation calibrated to the 90th percentile of validation entropy.

The main configuration uses a shared dimension of 512, consistent with the selected setting in the hyperparameter sensitivity analysis and the Methods definition. The supplementary implementation paragraph mentions 256 dimensions and the complexity paragraph also evaluates that smaller configuration; these values are retained as an explicitly selectable research variant rather than silently replacing the primary setting.

## Evaluation

Evaluation uses stratified five-fold patient-level splits, reserves 10% of each training fold for validation, repeats the experiment over 20 seeds, and computes 95% intervals with 1,000 bootstrap resamples. Cross-database assessment includes fixed-split external validation and leave-one-database-out evaluation. AUC comparisons use DeLong tests, concordance comparisons use McNemar tests, other paired metrics use paired t-tests, and comparisons across 28 baselines use Bonferroni correction.

The evaluation package includes treatment-response and toxicity AUC, survival concordance, Dice similarity, treatment concordance, time-dependent survival AUC, Brier score, calibration slope, expected calibration error, decision curves, net reclassification improvement, integrated discrimination improvement, Cochran Q, I², random-effects pooling, subgroup interaction analysis, missing-modality analysis, noise sweeps, label-noise analysis, and CT adversarial sensitivity.

Aggregate saved predictions with:

```bash
tumor-evaluate --config configurations/main.yaml --weights runs/main/model.pt --predictions artifacts/predictions.json --output artifacts/evaluation.json
```

Reference targets for the primary analysis are response AUC 0.856 ± 0.013, survival C-index 0.762 ± 0.015, toxicity AUC 0.871 ± 0.012, tumor Dice 0.912 ± 0.010, and treatment concordance 87.3% ± 2.8 over 20 seeds. Differences outside the reported seed variability should trigger checks of cohort versions, patient-level partitioning, modality alignment, and preprocessing manifests.

## Compute profile

The reported hardware is four NVIDIA A100 80 GB GPUs with mixed precision. The model contains approximately 127 million parameters in the reported analysis. Default training took about 14.2 aggregate GPU-hours. A patient assessment took 4.2 ± 0.3 seconds on one A100 and a full treatment-cycle assessment took 18.7 ± 1.4 seconds. Storage depends on the authorized source selections and is not reported in the study; record actual raw, intermediate, and processed sizes in each local manifest.

## Safety scope

This software is a research implementation for retrospective analysis. It is not a medical device and must not issue unsupervised clinical orders. Drug interactions, allergy records, comorbidities, predicted toxicity, and uncertainty escalation are mandatory inputs to the treatment policy. High-uncertainty outputs are deferred, and the clinical team remains responsible for treatment decisions.
