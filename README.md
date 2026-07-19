# Ave: Answer-Space and Path-Evidence Calibration for MMKGC

Ave is a research implementation for **multi-modal knowledge graph completion
(MMKGC)**. It combines a fine-grained multi-modal backbone with two
complementary calibration components:

- **Answer-Space Prior (ASP):** builds direction-aware relation-role priors
  from training triples and uses them to calibrate candidate entity scores.
- **Reliable Path Evidence (RPE):** mines reliable two-hop relation rules from
  the training graph and adds sparse candidate-level path evidence.

All calibration coefficients are selected on the validation split. After model
and coefficient selection, the test split is evaluated once.

## Overview

Given an incomplete triple query, Ave follows this workflow:

1. encode structural, visual, and textual entity information with a
   fine-grained multi-modal backbone;
2. train the backbone and ASP branch with the knowledge graph completion and
   fine-grained contrastive objectives;
3. select the forward and inverse ASP scales using validation MRR;
4. mine two-hop relation rules using training triples only;
5. select the forward and inverse RPE scales on the validation split;
6. freeze all settings and report filtered test metrics.

The main entry point is [`train_ave.py`](train_ave.py). Dataset-specific
commands are provided in [`run.sh`](run.sh).

## Repository structure

```text
.
├── data/                       # Train/validation/test KG splits
│   ├── DB15K/
│   ├── MKG-W/
│   └── MKG-Y/
├── tokens/                     # Visual/text token files and embedding tables
├── dataset.py                  # Dataset loading and reciprocal queries
├── merge_tokens.py             # Entity-to-token alignment
├── model_ave_backbone.py       # Fine-grained multi-modal backbone
├── model_ave.py                # Ave model and ASP components
├── train_ave.py                # Complete training and evaluation pipeline
├── train_ave_fgc.py            # Backbone/FGC training entry point
├── save_token_embeddings.py    # Text embedding extraction utility
├── run.sh                      # Reproduction commands
└── requirements.txt            # Reference dependency versions
```

## Datasets

The archive contains the following structural splits:

| Dataset | Entities | Relations | Train | Validation | Test |
|:--|--:|--:|--:|--:|--:|
| DB15K | 12,842 | 279 | 79,222 | 9,904 | 9,902 |
| MKG-W | 15,000 | 169 | 34,196 | 4,276 | 4,274 |
| MKG-Y | 15,000 | 28 | 21,310 | 2,665 | 2,663 |

> **DB15K split note:** the DB15K files distributed in this archive contain
> 9,904 validation triples and 9,902 test triples. Some published benchmark
> tables report these two counts in the opposite order. Report the split
> actually used by an experiment and do not tune calibration parameters on the
> test set.

Each dataset directory must contain:

```text
data/<DATASET>/
├── entities.txt
├── relations.txt
├── train.txt
├── valid.txt
└── test.txt
```

`MKG-W` and `MKG-Y` additionally use `entity2id.txt` to align entity names with
token dictionaries.

## Environment

The reference environment is:

- Python 3.9
- PyTorch 2.0.0
- NumPy 1.24.2
- scikit-learn 1.2.2
- tqdm 4.64.1
- Transformers 4.28.0 (needed when regenerating text embeddings)

A CUDA-capable GPU is required by `train_ave.py`.

```bash
conda create -n ave python=3.9 -y
conda activate ave

pip install numpy==1.24.2 scikit-learn==1.2.2 tqdm==4.64.1
pip install torch==2.0.0
pip install transformers==4.28.0
```

Install the PyTorch build appropriate for the local CUDA version when the
command above is not suitable for the target machine.

## Token preparation

Visual and textual token dictionaries are stored under `tokens/`. The default
configuration expects BEiT visual tokens and BERT or LLaMA textual tokens.

Representative files are:

```text
tokens/
├── DB15K-visual.json.zip
├── DB15K-textual.json
├── MKG-W-visual.json.zip
├── MKG-W-textual.json
├── MKG-Y-visual.json.zip
├── MKG-Y-textual-llama.json
├── visual.pth
├── textual.pth
└── textual_llama.pth
```

Compressed visual JSON files do not need to be manually extracted:
`train_ave.py` can read either `<name>.json` or `<name>.json.zip`.

MKG-Y uses the LLaMA tokenizer configuration in `run.sh` and therefore requires:

```text
tokens/MKG-Y-textual-llama.json
tokens/textual_llama.pth
```

Because `tokens/textual_llama.pth` is too large to include in this repository,
download the Hugging Face-format LLaMA-7B checkpoint from Hugging Face, set its
local path as `model_path` in `save_token_embeddings.py`, and run:

```bash
python save_token_embeddings.py
```

The generated embedding file will be saved as
`tokens/textual_llama.pth`.

## Running experiments

Run commands from the repository root. Copy the block for the desired dataset
from `run.sh`; do not launch every block simultaneously unless enough GPU
memory is available.

For example, the MKG-W configuration is:

```bash
CUDA_VISIBLE_DEVICES=0 python train_ave.py \
  --data MKG-W --exp ave \
  --text_tokenizer bert --visual_tokenizer beit \
  --num_epoch 1500 --valid_epoch 50 --early_stop 0 \
  --seed 2024 --dim 256 --hidden_dim 1024 --num_head 4 \
  --num_layer_enc_ent 1 --num_layer_enc_rel 1 --num_layer_dec 2 \
  --dropout 0.01 --emb_dropout 0.9 \
  --vis_dropout 0.4 --txt_dropout 0.1 \
  --max_vis_token 8 --max_txt_token 8 \
  --batch_size 2048 --eval_batch_size 256 \
  --lr 5e-4 --step_size 50 --mu 0.001 \
  --lambda_role_ce 1.0 --lambda_role_reg 0.0001 \
  --role_direct_weight 0.5 --similar_roles 4 \
  --role_scales "0,0.25,0.5,0.75,1,1.5" \
  --min_rule_support 2 \
  --path_alphas "0,0.25,0.5,1,2,4,8" \
  --non_deterministic
```

The supplied MKG-Y command uses `batch_size=512`, `eval_batch_size=64`, and
`max_txt_token=12` as a 24 GiB GPU configuration. If memory remains
insufficient, reduce the batch sizes before changing model dimensions or token
lengths.

## Outputs

For an experiment named with `--exp <EXP>`, files are written to:

```text
logs/<EXP>/<DATASET>/<RUN>.log
ckpt/<EXP>/<DATASET>/<RUN>_best.ckpt
result/<EXP>/<DATASET>/<RUN>.json
```

The JSON summary records:

- the selected checkpoint epoch;
- validation-selected ASP scales;
- validation-selected RPE scales;
- rule-mining statistics;
- validation and final filtered-test metrics;
- relation-category and head/tail breakdowns.

The principal metrics are MRR and Hits@1/3/10. Evaluation is performed for
both head and tail prediction under the filtered protocol.

## Reproducibility checklist

- Keep dataset splits and token files unchanged across compared methods.
- Select checkpoints and all ASP/RPE coefficients using validation MRR only.
- Evaluate the test split only after all settings are frozen.
- Record the random seed, tokenizer, token limits, and batch sizes.
- Remove `--non_deterministic` when stronger determinism is preferred.
- Report mean and standard deviation over multiple seeds for final claims.

## Attribution

The fine-grained tokenized multi-modal backbone, pretrained token resources,
and fine-grained contrastive objective build on the public implementation
associated with **MyGO**. Ave adds the Answer-Space Prior, Reliable Path
Evidence, direction-specific validation selection, and the integrated
calibration workflow. Publications and redistributed code based on this
repository should preserve the upstream attribution.


MKG-W and MKG-Y were introduced with the MMRNS benchmark:

> Derong Xu, Tong Xu, Shiwei Wu, Jingbo Zhou, and Enhong Chen. Relation-enhanced
> Negative Sampling for Multimodal Knowledge Graph Completion. ACM Multimedia,
> 2022. <https://doi.org/10.1145/3503161.3548388>
