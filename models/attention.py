import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_size=1024, attention_dim=512):
        super(Attention, self).__init__()
        self.w_image = nn.Linear(hidden_size, attention_dim)
        self.w_text = nn.Linear(hidden_size, attention_dim)
        self.w_attn = nn.Linear(attention_dim, 1)
        
    def forward(self, image_features, text_features):
        """
        image_features: (batch, num_regions, hidden_size) e.g., (32, 49, 1024)
        text_features: (batch, hidden_size) e.g., (32, 1024)
        """
        # Project features
        img_proj = self.w_image(image_features)
        text_proj = self.w_text(text_features).unsqueeze(1)

        # Combine (broadcast text over regions)
        combined = torch.tanh(img_proj + text_proj)
        
        # Calculate attention weights
        scores = self.w_attn(combined).squeeze(2)
        alpha = F.softmax(scores, dim=1)
        
        # Weighted sum of image regions
        attended_features = (image_features * alpha.unsqueeze(2)).sum(dim=1)
        
        return attended_features, alpha
