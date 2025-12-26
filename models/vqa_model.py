import torch
import torch.nn as nn
from .encoders import ImageEncoder, TextEncoder
from .fusion import GatedFusion
from .attention import Attention

class VQAModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_size = config['fusion']['hidden_size']
        
        self.image_encoder = ImageEncoder(embed_size=embed_size, frozen=config['image_encoder']['frozen'])
        self.text_encoder = TextEncoder(model_name=config['text_encoder']['model_name'], embed_size=embed_size, frozen=config['text_encoder']['frozen'])
        
        self.attention = Attention(hidden_size=embed_size, attention_dim=512)
        self.fusion = GatedFusion(input_size=embed_size, hidden_size=embed_size, dropout_p=config['fusion']['dropout'])
        
        self.classifier = nn.Linear(embed_size, config['output_size'])
        
    def forward(self, images, questions, attention_mask):
        img_features = self.image_encoder(images)
        text_features = self.text_encoder(questions, attention_mask)
        
        attended_img_features, attn_weights = self.attention(img_features, text_features)
        
        fused = self.fusion(attended_img_features, text_features)
        output = self.classifier(fused)
        
        return output
