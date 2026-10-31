# Full training on Kaggle

## Before starting

- Use a GPU accelerator in Kaggle (T4 x2 or P100). Turn **Internet on** in the notebook settings so the repo and `efficient-kan` can be installed.
- Attach the ASVspoof2019 LA dataset (the `awsaf49/asvpoof-2019-dataset` input has worked before); check the exact `LA/LA` path under `/kaggle/input`.
- **Start a fresh run.** The model now includes the paper's residual encoder, pre-emphasis and positional embeddings, so checkpoints from earlier runs (including the 12-epoch run whose best dev EER was 14.57%) cannot be loaded with `INIT_FROM`, `RESUME_FROM` or `EVAL.py`. They stop with a clear error. Keep the old `best.pt` if you want it: it only loads on commit `21d1c66`.

## Setup (first notebook cell)

```bash
!git clone https://github.com/gladiator456/aasist3.git /kaggle/working/aasist3
%cd /kaggle/working/aasist3
!pip install -q -r requirements.txt
!ls /kaggle/input   # find the dataset folder name
```

## Short trial first (a few minutes)

```bash
!DATASET_ROOT="/kaggle/input/datasets/awsaf49/asvpoof-2019-dataset/LA/LA" \
CHECKPOINT_DIR="/kaggle/working/checkpoints_trial" \
MAX_TRAIN_SAMPLES=2000 MAX_DEV_SAMPLES=1000 NUM_EPOCHS=2 BATCH_SIZE=4 LOG_EVERY=50 WARMUP_STEPS=50 \
python -u TRAIN.py
```

Check that it runs and confirm `last.pt` / `best.pt` appear. Note the seconds per 100 steps in the log: a full epoch is about `25380 / BATCH_SIZE / 100` times that, plus a few minutes for dev evaluation. Choose `NUM_EPOCHS` so the whole run stays under about 9 hours (Kaggle's background limit is roughly 12).

For memory, the new model needs very roughly 0.5 GB per sample on CPU in fp32 (a rough proxy, not a GPU measurement), so `BATCH_SIZE=16` should fit on a 15 GB T4. Run `!nvidia-smi` while the trial is running and, if it runs out of memory, drop to 8.

## Full run

```bash
%cd /kaggle/working/aasist3

!DATASET_ROOT="/kaggle/input/datasets/awsaf49/asvpoof-2019-dataset/LA/LA" \
CHECKPOINT_DIR="/kaggle/working/checkpoints_full" \
NUM_EPOCHS=20 \
BATCH_SIZE=16 \
BASE_LR=1e-4 \
LOG_EVERY=100 \
NUM_WORKERS=2 \
python -u TRAIN.py
```

Do not set `MAX_TRAIN_SAMPLES` or `MAX_DEV_SAMPLES` for full-dataset training. Their default is 0, meaning use all examples. If `BATCH_SIZE=16` runs out of memory, use 8 or 4. Adjust `NUM_EPOCHS` using the timing from the trial.

## What the trainer reports

Each epoch prints training loss, development loss, accuracy, EER, ROC-AUC, a confusion matrix, and recall for both spoof and bonafide classes. Do not use accuracy alone: the dataset is imbalanced. The first epochs use a linear LR warm-up (`WARMUP_STEPS`, default 500) before cosine decay.

- `checkpoints_full/best.pt`: checkpoint with the lowest development EER seen in this run.
- `checkpoints_full/last.pt`: most recent completed epoch.

Kaggle sessions have time limits. Download or "Save Version" the checkpoint files from `/kaggle/working` before the session ends.

## Resuming an interrupted full run

Set `RESUME_FROM` to the full run's `last.pt` (or `auto`) and keep `NUM_EPOCHS`, dataset paths, batch size, and subset settings consistent with the original run. Resume restores the optimizer, scheduler, and scaler state. Do not use `INIT_FROM` and `RESUME_FROM` together.

```bash
%cd /kaggle/working/aasist3

!DATASET_ROOT="/kaggle/input/datasets/awsaf49/asvpoof-2019-dataset/LA/LA" \
RESUME_FROM=auto \
CHECKPOINT_DIR="/kaggle/working/checkpoints_full" \
NUM_EPOCHS=20 \
BATCH_SIZE=16 \
BASE_LR=1e-4 \
LOG_EVERY=100 \
NUM_WORKERS=2 \
python -u TRAIN.py
```

If resuming in a fresh Kaggle session, first restore the saved checkpoint files into the same `CHECKPOINT_DIR` (for example by attaching your saved output as a dataset and copying `last.pt` over).

## Scoring on the eval split

`best.pt` is chosen by dev EER, so report the held-out eval number instead:

```bash
%cd /kaggle/working/aasist3

!DATASET_ROOT="/kaggle/input/datasets/awsaf49/asvpoof-2019-dataset/LA/LA" \
CHECKPOINT="/kaggle/working/checkpoints_full/best.pt" \
SCORES_OUT="/kaggle/working/eval_scores.txt" \
python -u EVAL.py
```

It prints the eval EER and ROC-AUC (first ~4 s of each clip; no min t-DCF). You can add this as a third cell in the background notebook so it runs automatically after training.
