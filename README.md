# Cats vs. Dogs: ResNet18 Transfer Learning with Grid Search

Classifies cat and dog images with an ImageNet-pretrained ResNet18 and selects hyperparameters
(learning rate, batch size, fine-tuned layers, dropout, optimizer) via grid search, choosing the
configuration with the lowest **validation loss**. The test set is used only once, at the end.

## Pipeline
1. Download the Kaggle dataset and build a fixed, stratified 70/15/15 split
2. Preload images into memory and define transforms (augmentation on train only)
3. Build ResNet18 with a new `Dropout -> Linear(512, 2)` head and selective freezing
4. Grid search on train/validation with early stopping (resumable via CSV)
5. Retrain the best configuration and plot learning curves
6. Evaluate on train / validation / test, confusion matrix, error examples

## Run
```bash
pip install -r requirements.txt
jupyter notebook cats_dogs_resnet18_gridsearch.ipynb
```
A GPU is strongly recommended (full grid: ~1-2 h on a T4). To run faster, lower
`GRID_TRAIN_FRACTION` or shrink `PARAM_GRID` in the parameters cell.
The dataset is downloaded through `kagglehub` (may require Kaggle credentials).

## Outputs
Saved to `results_transfer_learning/`: grid results, split, best model, history, plots and metrics.
