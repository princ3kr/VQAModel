import json
import os
import argparse
from collections import Counter

def build_vocab(annotations_path, output_path, top_k=1000):
    print(f"Loading annotations from {annotations_path}...")
    with open(annotations_path, 'r') as f:
        data = json.load(f)
    
    annotations = data['annotations']
    answers = [ann['multiple_choice_answer'] for ann in annotations]
    
    counter = Counter(answers)
    
    # Get top k answers
    most_common = counter.most_common(top_k)
    vocab = {ans: idx for idx, (ans, count) in enumerate(most_common)}
    
    print(f"Vocab size: {len(vocab)}")
    print(f"Top 5 answers: {most_common[:5]}")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(vocab, f, indent=4)
    print(f"Saved vocabulary to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations_path', type=str, required=True, help='Path to train annotations JSON')
    parser.add_argument('--output_path', type=str, default='data/answer_to_idx.json', help='Output path for vocab')
    parser.add_argument('--top_k', type=int, default=1000, help='Number of answers to keep')
    args = parser.parse_args()
    
    build_vocab(args.annotations_path, args.output_path, top_k=args.top_k)
