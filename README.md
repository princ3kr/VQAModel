# 🖼️ Visual Question Answering (VQA) Model

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A deep learning model for Visual Question Answering that combines visual and textual understanding to answer questions about images. Built with PyTorch, leveraging ResNet50 for image encoding and BERT for text encoding with a novel gated fusion mechanism.

---

## 📊 Model Performance

| Dataset | Hard Accuracy | Soft Accuracy (VQA Standard) |
|---------|--------------|------------------------------|
| **Validation** | **49.90%** | **58.56%** |

> **Note**: Soft Accuracy uses the official VQA evaluation metric: `min(#humans_who_gave_answer / 3, 1)`

---

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐
│   Input Image   │     │  Input Question │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│   ResNet50      │     │  BERT Encoder   │
│ (Image Encoder) │     │ (Text Encoder)  │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     │
              ┌──────▼──────┐
              │  Attention  │
              │   Module    │
              └──────┬──────┘
                     │
              ┌──────▼──────┐
              │   Gated     │
              │   Fusion    │
              └──────┬──────┘
                     │
              ┌──────▼──────┐
              │ Classifier  │
              │  (FC Layer) │
              └──────┬──────┘
                     │
                     ▼
              ┌─────────────┐
              │   Answer    │
              └─────────────┘
```

### Key Components

|     Component     |                 Description                 |
|-------------------|---------------------------------------------|
| **Image Encoder** |      ResNet50 (pretrained on ImageNet)      |
|  **Text Encoder** |              BERT-base-uncased              |
|   **Attention**   |   Visual attention guided by text features  |
|   **Fusion**      |     Gated fusion mechanism with dropout     |
|   **Classifier**  | Fully connected layer (1000 answer classes) |

---

## 📁 Project Structure

```
VQAModel/
├── configs/
│   └── default_config.yaml    # Hyperparameters & paths
├── data/
│   ├── vqa_dataset.py         # Dataset loader
│   └── answer_to_idx.json     # Answer vocabulary
├── models/
│   ├── vqa_model.py           # Main VQA model
│   ├── encoders.py            # Image & Text encoders
│   ├── fusion.py              # Gated fusion module
│   └── attention.py           # Attention mechanism
├── scripts/
│   ├── train.py               # Training script
│   ├── evaluate.py            # Evaluation script
│   ├── generate_submission.py # EvalAI submission generator
│   └── build_vocab.py         # Answer vocabulary builder
├── utils/
│   └── metrics.py             # VQA soft accuracy metric
├── checkpoints/               # Saved model weights
├── notebooks/                 # Jupyter notebooks
└── dataset/                   # VQA v2.0 dataset
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/princ3kr/VQAModel.git
   cd VQAModel
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download VQA v2.0 Dataset**
   
   Download the following from [VQA v2.0 website](https://visualqa.org/download.html):
   - Training images (COCO 2014)
   - Validation images (COCO 2014)
   - Training questions
   - Validation questions
   - Training annotations
   - Validation annotations

   Place them in the `dataset/coco2014/` directory following this structure:
   ```
   dataset/coco2014/
   ├── images/
   │   ├── train2014/
   │   ├── val2014/
   │   └── test2014/
   ├── questions/
   │   ├── OpenEnded_mscoco_train2014_questions.json
   │   ├── OpenEnded_mscoco_val2014_questions.json
   │   └── OpenEnded_mscoco_test2015_questions.json
   └── annotations/
       ├── mscoco_train2014_annotations.json
       └── mscoco_val2014_annotations.json
   ```

---

## 🏋️ Training

### Build Answer Vocabulary (First Time Only)
```bash
python scripts/build_vocab.py
```

### Train the Model
```bash
python scripts/train.py
```

### Configuration

Edit `configs/default_config.yaml` to customize training:

```yaml
model:
  image_encoder:
    model_name: "resnet50"
    frozen: true
  text_encoder:
    model_name: "bert-base-uncased"
    frozen: false
  fusion:
    hidden_size: 1024
    dropout: 0.5
  output_size: 1000

training:
  batch_size: 16
  epochs: 10
  learning_rate: 0.0001
  save_dir: "checkpoints/"
```

---

## 📈 Evaluation

### Evaluate on Validation Set
```bash
python scripts/evaluate.py
```

### Generate EvalAI Submission (Test Set)
```bash
python scripts/generate_submission.py
```

The submission file will be saved to `checkpoints/vqa_submission.json`.

---

## 📋 Results Interpretation

The model uses two accuracy metrics:

- **Hard Accuracy**: Exact match with the most common ground truth answer
- **Soft Accuracy (VQA Standard)**: `min(#annotators_who_gave_answer / 3, 1)`
  - If 0 annotators gave the predicted answer: 0%
  - If 1 annotator gave the predicted answer: 33.3%
  - If 2 annotators gave the predicted answer: 66.7%
  - If 3+ annotators gave the predicted answer: 100%

---

## 🔧 Technical Details

### Hardware Requirements
- **GPU**: NVIDIA GPU with 8GB+ VRAM (training)
- **RAM**: 16GB+ recommended
- **Storage**: ~25GB for dataset

### Training Details
- **Optimizer**: Adam
- **Learning Rate**: 1e-4
- **Batch Size**: 16-32
- **Image Size**: 224×224
- **Max Question Length**: 30 tokens

---

## 📚 References

- [VQA: Visual Question Answering](https://arxiv.org/abs/1505.00468) - Agrawal et al.
- [Making the V in VQA Matter](https://arxiv.org/abs/1612.00837) - Goyal et al.
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) - Devlin et al.
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) - He et al.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

<p align="center">
  Made with ❤️ using PyTorch
</p>
