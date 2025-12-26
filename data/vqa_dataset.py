import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import json
from tqdm import tqdm

class VQADataset(Dataset):
    def __init__(self, image_dir, questions_path, annotations_path, answer_vocab_path, transform=None, tokenizer=None, split='train'):
        self.image_dir = image_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.split = split
        
        # Load Questions
        print(f"Loading questions from {questions_path}...")
        with open(questions_path, 'r') as f:
            all_questions = json.load(f)['questions']
            
        # Load Annotations
        self.annotations = None
        if split != 'test':
            print(f"Loading annotations from {annotations_path}...")
            with open(annotations_path, 'r') as f:
                self.annotations = json.load(f)['annotations']

            # Load Answer Vocab
            print(f"Loading answer vocab from {answer_vocab_path}...")
            with open(answer_vocab_path, 'r') as f:
                self.answer_to_idx = json.load(f)

        # Create map from question_id to annotation
        if self.annotations:
            self.qid_to_ann = {ann['question_id']: ann for ann in self.annotations}
        
        # Filter out questions with missing images
        print(f"Validating image paths for {len(all_questions)} questions...")
        self.questions = []
        missing_count = 0
        for q in tqdm(all_questions, desc=f"Filtering {split} set"):
            image_id = q['image_id']
            filename = f"COCO_{os.path.basename(self.image_dir)}_{int(image_id):012d}.jpg"
            image_path = os.path.join(self.image_dir, filename)
            if os.path.exists(image_path):
                self.questions.append(q)
            else:
                missing_count += 1
        
        if missing_count > 0:
            print(f"Warning: Skipped {missing_count} questions due to missing images.")
        print(f"Final {split} dataset size: {len(self.questions)} questions")

    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        question_item = self.questions[idx]
        image_id = question_item['image_id']
        question_text = question_item['question']
        question_id = question_item['question_id']
        
        # 1. Load Image
        filename = f"COCO_{os.path.basename(self.image_dir)}_{int(image_id):012d}.jpg"
        image_path = os.path.join(self.image_dir, filename)
        
        try:
            image = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            raise FileNotFoundError(f"Image not found: {image_path}")

        if self.transform:
            image = self.transform(image)

        # 2. Tokenize Question
        if self.tokenizer:
            encoding = self.tokenizer(
                question_text,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=32
            )
            input_ids = encoding['input_ids'].squeeze(0)
            attention_mask = encoding['attention_mask'].squeeze(0)
        else:
            input_ids = torch.zeros(32, dtype=torch.long)
            attention_mask = torch.zeros(32, dtype=torch.long)

        # 3. Get Label
        label = torch.tensor(-1, dtype=torch.long)
        all_answers = []  # For soft accuracy
        
        if self.split != 'test' and self.annotations:
            ann = self.qid_to_ann.get(question_id)
            if ann:
                # Get main answer for hard accuracy
                answer = ann['multiple_choice_answer']
                if answer in self.answer_to_idx:
                    label = torch.tensor(self.answer_to_idx[answer], dtype=torch.long)
                else:
                    label = torch.tensor(-1, dtype=torch.long)
                
                # Get all 10 human answers for soft accuracy
                for ans_item in ann.get('answers', []):
                    ans_text = ans_item['answer']
                    if ans_text in self.answer_to_idx:
                        all_answers.append(self.answer_to_idx[ans_text])
                    else:
                        all_answers.append(-1)

        return {
            'image': image,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': label,
            'all_answers': all_answers
        }
