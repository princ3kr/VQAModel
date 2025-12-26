import sys
import os
sys.path.append(os.getcwd())

import yaml
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import BertTokenizer
from data.vqa_dataset import VQADataset

def test_loading():
    print("Testing VQA Data Loading...")
    
    # Load Config
    with open('configs/default_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    print("Config loaded.")
    
    # Setup Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Setup Tokenizer
    tokenizer = BertTokenizer.from_pretrained(config['model']['text_encoder']['model_name'])
    
    # Init Dataset
    train_config = config['data']['train']
    dataset = VQADataset(
        image_dir=train_config['image_dir'],
        questions_path=train_config['questions_path'],
        annotations_path=train_config['annotations_path'],
        answer_vocab_path=config['data']['answer_vocab_path'],
        transform=transform,
        tokenizer=tokenizer,
        split='train'
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Init DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Fetch one batch
    batch = next(iter(dataloader))
    
    print("\nBatch Keys:", batch.keys())
    print("Image Shape:", batch['image'].shape)
    print("Input IDs Shape:", batch['input_ids'].shape)
    print("Attention Mask Shape:", batch['attention_mask'].shape)
    print("Label Shape:", batch['label'].shape)
    
    print("\nSample Label (first item):", batch['label'][0].item())
    
    assert batch['image'].shape == (4, 3, 224, 224)
    assert batch['input_ids'].shape == (4, 32)
    assert batch['label'].shape == (4,)
    
    print("\nSUCCESS: Data loading works!")

if __name__ == '__main__':
    test_loading()
