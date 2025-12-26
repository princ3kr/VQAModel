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

def train():
    # Load Config
    with open('configs/default_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize Tokenizer
    tokenizer_name = config['model']['text_encoder']['model_name']
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Initialize Model
    model = VQAModel(config['model']).to(device)
    
    # Initialize Datasets
    train_dataset = VQADataset(
        config['data']['train']['image_dir'],
        config['data']['train']['questions_path'],
        config['data']['train']['annotations_path'],
        config['data']['answer_vocab_path'],
        tokenizer=tokenizer,
        transform=transform,
        split='train'
    )
    val_dataset = VQADataset(
        config['data']['val']['image_dir'],
        config['data']['val']['questions_path'],
        config['data']['val']['annotations_path'],
        config['data']['answer_vocab_path'],
        tokenizer=tokenizer,
        transform=transform,
        split='val'
    )

    # Initialize Dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'],
        shuffle=True, 
        pin_memory=config['training']['pin_memory'], 
        num_workers=config['training']['no_of_workers'],
        collate_fn=custom_collate
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['training']['batch_size'],
        shuffle=False, 
        pin_memory=config['training']['pin_memory'], 
        num_workers=config['training']['no_of_workers'],
        collate_fn=custom_collate
    )
    
    # Optimizer and Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    
    # Training Loop
    print("Starting training...")
    best_acc = 0.0
    
    for epoch in range(config['training']['epochs']):
        model.train()
        train_loss = 0.0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(images, input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}")
        
        # Validation Loop with Soft Accuracy
        model.eval()
        soft_score = 0.0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                all_answers = batch['all_answers']
                
                outputs = model(images, input_ids, attention_mask)
                _, predicted = torch.max(outputs.data, 1)
                
                valid_mask = labels != -1
                soft_score += soft_accuracy(predicted, all_answers)
                total += valid_mask.sum().item()
        
        val_acc = 100 * soft_score / total if total > 0 else 0
        print(f"Epoch {epoch+1} Val Soft Accuracy: {val_acc:.2f}%")
        
        # Save Best Model   
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(config['training']['save_dir'], 'best_model.pth')
            os.makedirs(config['training']['save_dir'], exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"Saved new best model with acc: {val_acc:.2f}%")

if __name__ == '__main__':
    train()