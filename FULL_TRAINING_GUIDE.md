# Full training on Kaggle

## Before starting

- Use a GPU accelerator in Kaggle.
- Attach the ASVspoof2019 LA dataset.
- Keep the short-trial checkpoints untouched.
- The command below initializes model weights from the successful 2-epoch trial, but creates a fresh optimizer and learning-rate schedule. This is intentional: the trial's scheduler was configured for only 2 epochs and must not be reused for a 25-epoch run.

## Run

From a Kaggle notebook cell:

```bash
%cd /kaggle/working/aasist3

!DATASET_ROOT="/kaggle/input/datasets/awsaf49/asvpoof-2019-dataset/LA/LA" \
INIT_FROM="/kaggle/working/aasist3/checkpoints/best.pt" \
CHECKPOINT_DIR="/kaggle/working/aasist3/checkpoints_full" \
NUM_EPOCHS=25 \
BATCH_SIZE=2 \
BASE_LR=1e-4 \
LOG_EVERY=100 \
NUM_WORKERS=2 \
python -u TRAIN.py
```

Do not set `MAX_TRAIN_SAMPLES` or `MAX_DEV_SAMPLES` for full-dataset training. Their default is 0, meaning use all examples.

## What the updated trainer reports

Each epoch prints training loss, development loss, accuracy, EER, ROC-AUC, confusion matrix, and recall for both spoof and bonafide classes. Do not use accuracy alone: the dataset is imbalanced.

- `checkpoints_full/best.pt`: checkpoint with the lowest development EER seen in this run.
- `checkpoints_full/last.pt`: most recent completed epoch.

Download or copy the checkpoint files out of `/kaggle/working` before the Kaggle session ends.

## Resuming an interrupted full run

Set `RESUME_FROM` to the full run's `last.pt` and keep `NUM_EPOCHS`, dataset paths, batch size, and subset settings consistent with the original run. Resume restores the optimizer, scheduler, and scaler state. Do not use `INIT_FROM` and `RESUME_FROM` together.

For example, if the original 25-epoch run stopped after epoch 8:

```bash
%cd /kaggle/working/aasist3

!DATASET_ROOT="/kaggle/input/datasets/awsaf49/asvpoof-2019-dataset/LA/LA" \
RESUME_FROM="/kaggle/working/aasist3/checkpoints_full/last.pt" \
CHECKPOINT_DIR="/kaggle/working/aasist3/checkpoints_full" \
NUM_EPOCHS=25 \
BATCH_SIZE=2 \
BASE_LR=1e-4 \
LOG_EVERY=100 \
NUM_WORKERS=2 \
python -u TRAIN.py
```

If resuming in a fresh Kaggle session, first restore the checkpoint ZIP into the same checkpoint directory.
