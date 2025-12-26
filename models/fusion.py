import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedFusion(nn.Module):
    def __init__(self, input_size=1024, hidden_size=1024, dropout_p=0.3):
        super().__init__()
        self.linear_v = nn.Linear(input_size, hidden_size)
        self.linear_t = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        
    def forward(self, img_features, text_features):
        v = self.linear_v(img_features)
        t = self.linear_t(text_features)
        
        x = v * t
        x = self.dropout(x)
        return x
