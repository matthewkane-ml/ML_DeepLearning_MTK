# CNN Image Classifier — Dogs vs. Cats

> A VGG16-inspired convolutional neural network that classifies images as dogs or cats, built from scratch with TensorFlow/Keras and trained with early stopping and model checkpointing.

---

## Problem

Image classification is one of the most foundational tasks in computer vision. This project implements a deep convolutional neural network modeled on the VGG16 architecture to distinguish between dog and cat images — demonstrating the full pipeline from raw image loading and preprocessing to model training, evaluation, and inference.

## Dataset

- **Source:** Local image dataset organized into `dog/` and `cat/` subfolders under `data/raw/`
- **Format:** RGB images, resized to 224 × 224 pixels for VGG16-compatible input
- **Split:** 80% training / 20% validation via `ImageDataGenerator(validation_split=0.2)`
- **Classes:** `dog` (index 0), `cat` (index 1)

## Approach

1. **Data loading:** Used Keras `ImageDataGenerator` with `rescale=1./255` normalization and an 80/20 train/validation split. Images were loaded in batches of 8 at 224×224 resolution.
2. **Architecture:** Built a VGG16-style CNN from scratch using Keras `Sequential`:
   - 5 convolutional blocks with increasing filter depth (64 → 128 → 256 → 512 → 512)
   - Each block uses 3×3 `same` padding convolutions with ReLU activations, followed by 2×2 max pooling
   - Flattened output feeds into two 4,096-unit dense layers, then a 2-unit softmax output layer
3. **Training:** Compiled with `Adam (lr=0.001)` and `categorical_crossentropy` loss. Used `EarlyStopping` (patience=3) on validation accuracy and `ModelCheckpoint` to save the best weights.
4. **Inference:** Loaded the saved model to run predictions on individual images, outputting the class with the higher softmax probability.

## Results

The model was trained for 3 epochs on a small local dataset of **18 images** (16 training, 2 validation). Validation accuracy plateaued at **50%** across all epochs — effectively random chance — which is expected given the dataset size. Early stopping triggered after epoch 3 and restored the best weights from epoch 1. The training loss did decrease steadily (0.76 → 0.69), confirming the model was learning, but the validation set was too small to show meaningful improvement.

The primary value of this project is in the architectural implementation: building and training a full VGG16-style CNN end-to-end, not the benchmark performance. Re-running with the full Kaggle Dogs vs. Cats dataset (25,000 images) would be the natural next step.

## Tech stack

`Python` · `TensorFlow` · `Keras` · `NumPy` · `Matplotlib`

## Run it locally

```bash
git clone https://github.com/matthewkane-ml/ML_DeepLearning_MTK.git
cd ML_DeepLearning_MTK
pip install -r requirements.txt

# Organize your images into:
# data/raw/dog/   (dog images)
# data/raw/cat/   (cat images)

# Then open and run the notebook
jupyter notebook src/DeepLearning-VGG16.ipynb
```

## What I'd do next

- Use **transfer learning** with pre-trained VGG16 ImageNet weights instead of training from scratch — this would dramatically improve accuracy with a small dataset
- Add **data augmentation** (horizontal flips, rotations, zoom) to reduce overfitting
- Train for more epochs with a lower learning rate schedule to allow the model to converge more fully
- Extend to a multi-class classifier beyond binary dog/cat

---

**Author:** Matthew Kane — [LinkedIn](https://www.linkedin.com/in/thomas-k-392094410/) · [GitHub portfolio](https://github.com/matthewkane-ml)
