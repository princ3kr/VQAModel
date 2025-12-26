import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import yaml
import sys
import os
from transformers import BertTokenizer
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vqa_model import VQAModel
from data.vqa_dataset import VQADataset
from utils.metrics import soft_accuracy

def custom_collate(batch):
    """Custom collate to handle variable-length all_answers lists."""
    images = torch.stack([item['image'] for item in batch])
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    all_answers = [item['all_answers'] for item in batch]
    
    return {
        'image': images,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'label': labels,
        'all_answers': all_answers
    }

def evaluate():
    # Load Config
    with open('configs/default_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    # Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize Tokenizer
    tokenizer_name = config['model']['text_encoder']['model_name']
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

    # Initialize Model
    model = VQAModel(config['model']).to(device)
    
    # Load Checkpoint
    checkpoint_path = os.path.join(config['training']['save_dir'], 'best_model.pth')
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print("No checkpoint found! Testing with random weights (sanity check only).")

    model.eval()
    
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Using validation set for evaluation (test set has no annotations)
    val_dataset = VQADataset(
        config['data']['val']['image_dir'],
        config['data']['val']['questions_path'],
        config['data']['val']['annotations_path'],
        config['data']['answer_vocab_path'],
        transform=transform,
        tokenizer=tokenizer,
        split='val'
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=False, 
        pin_memory=config['training']['pin_memory'], 
        num_workers=config['training']['no_of_workers'],
        collate_fn=custom_collate
    )
    
    # Evaluation Loop
    hard_correct = 0
    soft_score = 0.0
    total = 0
    
    print("Starting evaluation...")
    print("Using VQA Soft Accuracy: min(#humans_who_gave_answer / 3, 1)")
    
    with torch.no_grad():
        for batch in tqdm(val_loader):
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            all_answers = batch['all_answers']  # List of lists
            
            outputs = model(images, input_ids, attention_mask)
            _, predicted = torch.max(outputs.data, 1)
            
            # Hard accuracy
            valid_mask = labels != -1
            hard_correct += (predicted[valid_mask] == labels[valid_mask]).sum().item()
            
            # Soft accuracy
            soft_score += soft_accuracy(predicted, all_answers)
            total += valid_mask.sum().item()
    
    hard_acc = 100 * hard_correct / total if total > 0 else 0
    soft_acc = 100 * soft_score / total if total > 0 else 0
    
    print(f"\n{'='*50}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"Total samples: {total}")
    print(f"Hard Accuracy: {hard_acc:.2f}%")
    print(f"Soft Accuracy: {soft_acc:.2f}%  (VQA Standard)")
    print(f"{'='*50}")

if __name__ == '__main__':
    evaluate()
