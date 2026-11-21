# Chest X-Ray Pneumonia Classification

Interactive Streamlit demo comparing two convolutional networks, ResNet18 and DenseNet121,
fine-tuned to classify chest X-ray images as NORMAL or PNEUMONIA.

Built for AIML-501 Model Development, residency project, MS in Artificial Intelligence and
Data Analytics.

> **Educational use only.** This is a coursework demonstration, not a clinical decision-support
> system. It is not intended for diagnosis, triage, or treatment, and it has not been clinically
> validated. Do not upload identifiable patient images.

## What it does

Upload a chest X-ray image and both models run inference on it side by side, so you can compare
their predicted class and their confidence on the same input. A second panel shows the held-out
evaluation results from Day 1.

## Day 1 evaluation results

| Model | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC |
|-------|---------:|----------:|-------:|------------:|---:|--------:|
| ResNet18 | 81.7% | 78.3% | 97.9% | 54.7% | 0.870 | 0.949 |
| DenseNet121 | 81.7% | 78.6% | 97.2% | 56.0% | 0.869 | 0.943 |

Both models recall pneumonia well, above 97%, but specificity is much weaker, 54.7% and 56.0%.
In other words they rarely miss a pneumonia case, and they frequently flag a normal X-ray as
pneumonia. After retraining, the two models land on the same accuracy, 81.7%, and their ROC-AUC
values stay close at 0.949 and 0.943, so neither one clearly sees more than the other. The
remaining difference is mostly about where the decision threshold sits.

## Running it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501 and runs entirely on CPU.

## Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit interface, model loading, and inference |
| `requirements.txt` | Python dependencies |
| `resnet18_pneumonia.pth` | Trained ResNet18 weights |
| `densenet121_pneumonia.pth` | Trained DenseNet121 weights |
| `model_comparison_results.csv` | Day 1 held-out evaluation metrics |

Weights load with `weights_only=True` and are mapped to CPU, so the app runs without a GPU.

## Portfolio

Part of my [Applied ML Portfolio](https://omsairamkvs99.github.io/ml-portfolio/).
