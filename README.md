# A Dual-Process Agentic Architecture for Political Hate Speech Detection in a Low-Resource Language

Code, datasets, prompts, model predictions, and evaluation results accompanying the research paper:

> A. Susuri, E. Fetahi, M. Saracevic, and E. Canhasi, **“A Dual-Process Agentic Architecture for Political Hate Speech Detection in a Low-Resource Language,”** *Electronics* (MDPI), under review, 2026.

This repository investigates a dual-process architecture for detecting **political hate speech in Albanian**, combining fast classifiers with a tool-using large language model (LLM) agent. It includes the released, de-identified evaluation data, annotation materials, experimental configurations, recorded model outputs, and scripts for reproducing the reported analyses.

## Approach

The architecture consists of two systems:

- **System 1 — fast classification:** a fine-tuned multilingual BERT (mBERT) classifier and **Jev**, a zero-shot System One model from TypeSafe AI.
- **System 2 — deliberative classification:** an LLM agent that may consult three tools: System 1 classifier scores, a knowledge base of political actors, and the comment's discussion-thread context.

The principal routing policy, **P1**, escalates examples where the two System 1 classifiers disagree. The experiments also assess standalone LLMs, single-prompt alternatives, agent ablations, alternative routing policies, calibration, hard negatives, counterfactuals, and grouped cross-validation over posts.

LLM requests use OpenRouter; Jev requests use the TypeSafe AI API. The agent implementation and tool definitions are in [`agent.py`](agent.py), model and experiment settings in [`config.py`](config.py), and the readable research prompts and schemas in [`prompts/`](prompts/).

## Repository structure

| Location | Description |
| --- | --- |
| `data/rev2/` | Main labelled data, hard negatives, natural counterfactuals, annotation judgments, and grouped CV assignments |
| `data/` | Split manifests, template counterfactuals, and additional evaluation sets |
| `annotation_guideline/` | Annotation guidelines and decision rules |
| `prompts/` | Prompts, tool schemas, Jev questions, and experimental message examples |
| `kb/political_entities.json` | Political-entity knowledge base used by the agent |
| `agent.py`, `entity_kb.py` | Agent logic and tool implementations |
| `llm_client.py`, `jev_client.py` | Provider clients and response-cache handling |
| `config.py` | Model identifiers, experiment plans, inference parameters, and budget settings |
| `rev2_*.py` | Post-split experiments, prediction import, and statistical analysis |
| `rev2_gpu/` | mBERT training scripts and recorded training/prediction outputs |
| `gpu_baselines/` | Transformer baselines for the additional experimental split |
| `results/classifier_preds/` | Saved classifier predictions |
| `results/llm_runs/` | Saved LLM decisions, rationales, tool use, latency, and usage statistics |
| `results/tables/`, `results/figures/` | Evaluation tables and visualizations |
| `results/stability/` | Repeated-inference stability results |
| `cache/llm_cache.zip`, `cache/jev_cache.zip` | Archived LLM and Jev API responses |
| `LICENSE`, `DATA_LICENSE` | Software and dataset licensing information |

## Datasets

The primary dataset is [`data/rev2/phs_post.csv`](data/rev2/phs_post.csv), containing **1,725 Albanian-language comments**. The binary target is **1 = political hate speech** and **0 = not political hate speech**. Data are grouped by their source posts for train, validation, and test partitioning.

| Dataset | Description |
| --- | --- |
| `data/rev2/phs_post.csv` | Main labelled comments, split assignments, and thread context |
| `data/post_split.csv` | Post-level train/validation/test split manifest |
| `data/rev2/post_cv_folds.csv` | Four-fold cross-validation assignment of 36 posts |
| `data/rev2/hard_negatives.csv` | 205 non-hateful comments mentioning political actors |
| `data/rev2/natural_cf2.csv` | 187 original test comments and two political-name substitutions each (561 rows) |
| `data/rev2/annotation_results.csv` | Two-annotator judgments for counterfactual pairs and hard-negative examples |
| `data/counterfactual.csv` | 457 political-entity counterfactual templates |
| `data/rev2/generic_templates.csv` | 60 templates targeting non-political entities |
| `data/phs_folds.csv`, `data/natural_cf.csv` | Additional split and counterfactual evaluation data |

In the main dataset, `text` stores the comment, `label` stores the target class, `split` stores its assignment, and `post_id` identifies the released post grouping. `parent_text` and `related_texts` provide available discussion context. Released comment/post identifiers are internal to the dataset and do not contain platform account identifiers.

The annotation conventions and instructions are documented in [`annotation_guideline/`](annotation_guideline/). Judgments from the two annotators are provided in `data/rev2/annotation_results.csv`.

## Installation

Use **Python 3.10–3.12** and install the packages listed in [`requirements.txt`](requirements.txt). Run the commands below from the repository root.

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For GPU training, use a PyTorch build compatible with your CUDA environment. CPU execution is sufficient for the evaluation scripts, but transformer training is substantially faster on a suitable GPU.

## Reproducing the results

### 1. Generate the main evaluation tables

The quickest way to reproduce the paper's statistical analyses is to use the distributed data, predictions, and recorded inference outputs:

```bash
python rev2_analysis.py
python rev2_extra.py
```

These commands regenerate the evaluation artifacts in `results/tables/rev2/`, including performance summaries, calibration results, policy comparisons, counterfactual sensitivity, and cross-validation statistics. **No API credentials, model retraining, or GPU are required** for this workflow.

### 2. Recompute classical baselines and import mBERT predictions

```bash
python rev2_classical.py
python rev2_import_mbert.py
python rev2_analysis.py
python rev2_extra.py
```

`rev2_classical.py` fits TF-IDF-based classifiers using the released post-split data. `rev2_import_mbert.py` processes saved seed-level mBERT predictions, selects the model using validation performance, and prepares the corresponding classifier prediction files. It does not retrain the transformer.

These scripts write output files within `results/`; run them in a separate working copy if you want to retain an unchanged snapshot of the distributed outputs.

### 3. Reproduce model inference

The repository includes compressed response caches. **Leave the archives in `cache/`; manual extraction is not required.** The API clients read recorded responses from these ZIP files when a matching request is available.

To run inference for requests that are not in the cache, set the relevant environment variables:

```bash
# Linux / macOS
export OPENROUTER_API_KEY="YOUR_OPENROUTER_KEY"
export TYPESAFE_API_KEY="YOUR_TYPESAFE_KEY"

# Windows PowerShell
# $env:OPENROUTER_API_KEY = "YOUR_OPENROUTER_KEY"
# $env:TYPESAFE_API_KEY = "YOUR_TYPESAFE_KEY"
```

Run the Jev predictions before the LLM-agent plans:

```bash
python rev2_jev.py
python rev2_llm.py --plan rev2
python rev2_llm.py --plan rev2_free
python rev2_llm.py --plan rev2_nemo_paid
python rev2_llm.py --plan rev2_bias
```

**Note:** Model plans may involve paid services. Missing cache entries can trigger live requests, and external model availability, provider versions, latency, and pricing may differ over time. The scripts skip completed per-example outputs where applicable. Set `PHS_NO_CACHE=1` only when fresh requests are specifically intended. The stability experiment (`python rev2_stability.py`) deliberately bypasses caching and requires live API access.

### 4. Retrain multilingual BERT (optional)

The mBERT scripts download the pretrained `bert-base-multilingual-cased` model and train on the released post-split dataset:

```bash
python rev2_gpu/train_mbert_post.py
python rev2_gpu/train_mbert_cv.py
python rev2_import_mbert.py
```

The post-split training script covers five random seeds (`11`, `22`, `33`, `44`, `55`) and both standard and hard-negative-augmented variants. The cross-validation script evaluates across four grouped post folds. Its training configuration is defined in `rev2_gpu/train_mbert_post.py` and `rev2_gpu/train_mbert_cv.py`.

The training scripts skip runs with existing `done.json` completion markers. For a full retraining experiment, use a separate working copy with a fresh `rev2_gpu/outputs/` directory, and run the post-split training script before cross-validation. Results may vary with hardware, library versions, and numerical settings.

## Paper results and output files

| Reported analysis | Output files |
| --- | --- |
| Classical and mBERT baselines | `results/tables/rev2_classical.csv`, `results/tables/rev2_mbert_seeds.csv` |
| System 1 performance and calibration | `results/tables/rev2/single_systems_ptest.csv`, `calibration_ptest.csv` |
| LLM and agent performance | `results/tables/rev2/single_systems_ptest.csv`, `llm_latency_ptest.csv` |
| Agent versus single-prompt comparison | `results/tables/rev2/agent_vs_single.csv` |
| Routing and validation policy analysis | `results/tables/rev2/policies_test.csv`, `validation_grid_*.csv`, `decomposition_ptest.csv` |
| Complementarity of classifiers | `results/tables/rev2/complementarity_ptest.csv` |
| Counterfactuals and hard negatives | `results/tables/rev2/natural_cf2.csv`, `hard_negatives_fpr.csv`, `templates.csv`, `templates_full.csv` |
| Post-level evaluation and sensitivity | `results/tables/rev2/per_post_ptest.csv`, `cv_over_posts.csv`, `sensitivity_ptest.csv` |
| CPU measurements | `results/tables/T13_edge.csv`, `T13_edge_vm.csv`, `T13_edge_system.json` |
| Overall summaries | `results/tables/rev2/summary.txt`, `extra.txt` (generated by the analysis scripts) |

Additional evaluation artifacts, including the alternative split analyses, are stored under `results/tables/`. Scripts for those experiments include `classical_baselines.py`, `run_jev.py`, `run_llm_experiments.py`, `evaluate.py`, and `formal_analysis.py`.

## Experiment settings and reproducibility

Core experimental settings and model identifiers are defined in [`config.py`](config.py). The default LLM configuration uses temperature `0`, a maximum of `1,000` output tokens per call, and up to six agent steps. The prompts, tool specifications, and message examples are available in [`prompts/`](prompts/).

Saved prediction tables and API responses support analysis of the recorded experiments without dependence on the current state of external inference services. Reproducing live responses or full training runs can yield different outputs because of provider changes, hardware differences, and software versions. Collection of the source social-media data is not part of the public reproduction workflow; the released de-identified datasets are the starting point for independent evaluation.

## Data ethics and permitted use

The data originate from publicly accessible Facebook posts and comments, principally from news pages and public profiles of political figures. Platform metadata, direct user identifiers, links, and private-person names have been removed or masked. Public political actors and organizations remain identifiable where required for the entity-bias experiments.

**The dataset contains offensive and hateful language.** It is provided for research into harmful-content detection and moderation. Do not attempt to re-identify the individuals who posted the comments or use the data to target individuals.

## Licenses

- **Source code:** MIT License, [`LICENSE`](LICENSE).
- **Datasets, annotations, and knowledge base:** Creative Commons Attribution–NonCommercial 4.0 International (CC BY-NC 4.0), [`DATA_LICENSE`](DATA_LICENSE).
- **Cached model responses:** subject to applicable model and API provider terms in addition to the repository's licensing information.

## Citation

If you use this repository, please cite the associated paper:

> A. Susuri, E. Fetahi, M. Saracevic, and E. Canhasi, “A Dual-Process Agentic Architecture for Political Hate Speech Detection in a Low-Resource Language,” *Electronics* (MDPI), under review, 2026.

Citation metadata for the repository is also available in [`CITATION.cff`](CITATION.cff).

## Contact

**Endrit Fetahi** (corresponding author)  
University of Prizren “Ukshin Hoti”
