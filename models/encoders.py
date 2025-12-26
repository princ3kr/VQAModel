import torch
import torch.nn as nn
import torchvision.models as models
from transformers import BertModel

class ImageEncoder(nn.Module):
    def __init__(self, embed_size=1024, frozen=True):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        # Remove avgpool and fc layers to keep spatial dimensions (7x7)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        
        # 1x1 convolution to project ResNet features (2048) to embed_size
        self.conv = nn.Conv2d(2048, embed_size, kernel_size=1)
        
        if frozen:
            for param in self.resnet.parameters():
                param.requires_grad = False
                
    def forward(self, images):
        features = self.resnet(images)
        features = self.conv(features)
        features = features.view(features.size(0), features.size(1), -1).permute(0, 2, 1)
        return features

class TextEncoder(nn.Module):
    def __init__(self, model_name='bert-base-uncased', embed_size=1024, frozen=False):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.bert_fc = nn.Linear(self.bert.config.hidden_size, embed_size)
        
        if frozen:
            for param in self.bert.parameters():
                param.requires_grad = False
                
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return self.bert_fc(cls_embedding)
