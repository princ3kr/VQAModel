import torch

def accuracy(outputs, labels):
    """Hard accuracy: exact match between prediction and label."""
    _, preds = torch.max(outputs, 1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def soft_accuracy(predictions, all_answers_batch):
    """
    VQA Soft Accuracy: min(#humans_who_gave_that_answer / 3, 1)
    
    Args:
        predictions: tensor of shape (batch_size,) with predicted answer indices
        all_answers_batch: list of lists, each inner list contains 10 answer indices
    
    Returns:
        total soft accuracy score for the batch
    """
    total_score = 0.0
    for pred, answers in zip(predictions.cpu().tolist(), all_answers_batch):
        if len(answers) == 0:
            continue
        count = sum(1 for ans in answers if ans == pred)
        score = min(count / 3.0, 1.0)
        total_score += score
    return total_score
