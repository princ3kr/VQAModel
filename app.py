import torch
import pandas as pd
import numpy as np
from torchvision import transforms
from transformers import BertTokenizer
import matplotlib.pyplot as plt
import streamlit as st
import yaml
import json
import PIL.Image as Image
from huggingface_hub import hf_hub_download

from models.vqa_model import VQAModel

# Hugging Face model repository - update this with your repo
HF_REPO_ID = "princ3kr/vqa-model"
MODEL_FILENAME = "best_model.pth"

@st.cache_data
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

@st.cache_data
def load_vocab(vocab_path):
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    idx_to_ans = {v: k for k , v in vocab.items()}
    return idx_to_ans

@st.cache_resource
def download_model_from_hf():
    """Download model from Hugging Face Hub"""
    model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_FILENAME)
    return model_path

@st.cache_resource
def load_model(config):
    model_path = download_model_from_hf()
    model = VQAModel(config['model'])
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    return model, device

def img_preprocessing(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)
    return image

@st.cache_resource
def load_tokenizer(tokenizer_name):
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    return tokenizer

def ques_preprocessing(question, tokenizer):
    ques = tokenizer(question, return_tensors='pt', padding='max_length', truncation=True, max_length=32)
    input_ids = ques['input_ids']
    attention_mask = ques['attention_mask']
    return input_ids, attention_mask

def main():
    st.title("VQA Model")
    st.write("This is a VQA model that can answer questions about images.")

    config = load_config('configs/default_config.yaml')
    model, device = load_model(config)
    idx_to_ans = load_vocab(config['data']['answer_vocab_path'])
    tokenizer = load_tokenizer(config['model']['text_encoder']['model_name'])

    col1, col2 = st.columns(2)

    with col1:
        image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if image_file:
            img = Image.open(image_file)
            st.image(img, caption="Uploaded Image", use_container_width=True)
            img_tensor = img_preprocessing(img)
            img_tensor = img_tensor.to(device)

    with col2:
        question = st.text_input("Enter a question")
        if image_file and question:
            input_ids, attention_mask = ques_preprocessing(question, tokenizer)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            with torch.no_grad():
                outputs = model(img_tensor, input_ids, attention_mask)
                _, pred = torch.max(outputs.data, 1)
                answer = idx_to_ans[pred.item()]
                st.write(f"### Answer: {answer}")
        else:
            st.info("Please upload an image and enter a question to get an answer.")


if __name__ == "__main__":
    main()