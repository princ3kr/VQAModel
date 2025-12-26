# VQA Model Notebooks

This directory contains two Jupyter notebooks demonstrating Visual Question Answering (VQA) models with different architectures and complexity levels.

---

## 📓 NOTEBOOK-1: Basic VQA Model

**File:** [Basic-VQA-Model.ipynb](./NOTEBOOK-1/Basic-VQA-Model.ipynb)

### Architecture
- **Image Encoder:** ResNet18 (pretrained, partial freeze)
- **Text Encoder:** LSTM (2-layer, hidden_size=768)
- **Tokenization:** Custom word-level tokenizer (vocabulary built from training data)
- **Fusion:** Concatenation + FC layers
- **Dataset:** Small VQA dataset with ~7,525 samples (filtered to top 100 answers)

### Dataset
📁 **Source:** Local `dataset/` folder with:
- `data_train.csv` - 9,974 samples (filtered to 7,525)
- `data_eval.csv` - 2,494 samples (filtered to 1,864)
- `images/` folder containing image files

### Training Configuration

|   Parameter   |                  Value                 |
|---------------|----------------------------------------|
|     Epochs    |         50 (early stopped at 22)       |
|   Batch Size  |                   32                   |
| Learning Rate |                 0.0005                 |
|   Optimizer   |        AdamW (weight_decay=0.01)       |
|      Loss     | CrossEntropyLoss (label_smoothing=0.1) |

### Results

|   Split   | Loss |  Accuracy  |
|-----------|------|------------|
| **Train** | 2.23 |   52.88%   |
|  **Val**  | 3.03 |   31.53%   |
| **Test**  | 3.02 | **28.97%** |

---

## 📓 NOTEBOOK-2: Upgraded VQA Model (BERT + ResNet)

**File:** [Upgraded_model.ipynb](./NOTEBOOK-2/Upgraded_model.ipynb)

### Architecture
- **Image Encoder:** ResNet18 (pretrained, freeze except layer4)
- **Text Encoder:** BERT (bert-base-uncased, frozen)
- **Tokenization:** HuggingFace BertTokenizer
- **Fusion:** Concatenation + FC layers with dropout
- **Dataset:** VQA 2.0 subset (~20k train, ~19k val samples, top 1000 answers)

### Dataset
📁 **Source:** VQA 2.0 dataset in `data/` folder:
- `questions/train.json`, `questions/test.json`, `questions/val.json`
- `annotations/train.json`, `annotations/test.json`, `annotations/val.json`
- `images/train/`, `images/val/`, `images/test/` (COCO 2014 format)

### Training Configuration
|   Parameter   |                  Value                 |
|---------------|----------------------------------------|
|     Epochs    |          50 (interrupted at 6)         |
|   Batch Size  |                   32                   |
| Learning Rate |                 0.0001                 |
|   Optimizer   |        AdamW (weight_decay=0.01)       |
|      Loss     | CrossEntropyLoss (label_smoothing=0.1) |
|   Hidden Dim  |                  512                   |

### Results (at Epoch 5, before interrupt)

|   Split   | Loss |  Accuracy  |
|-----------|------|------------|
| **Train** | 4.62 |   21.52%   |
|  **Val**  | 4.37 | **21.96%** |
| **Test**  | N/A  |    N/A     |

> ⚠️ Training was interrupted at epoch 6. Model was still converging.

---

## Comparison

|    Feature     | Notebook 1 (Basic) | Notebook 2 (Upgraded) |
|----------------|--------------------|-----------------------|
|  Text Encoder  |        LSTM        |     BERT (frozen)     |
|  Tokenization  |       Custom       |     BertTokenizer     |
|  Dataset Size  |       ~7.5k        |         ~20k          |
| Answer Classes |       100          |        1000           |
|  Best Val Acc  |    **31.53%**      |         21.96%        |
|  Best Test Acc |    **28.97%**      |         —             |

> 💡 **Note:** Notebook 1 achieves higher accuracy due to fewer answer classes (100 vs 1000) and a simpler classification task.
