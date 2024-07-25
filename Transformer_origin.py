import torch
from torch import nn
import torch.nn.functional as F

class Transformer(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, pad_idx, nhead, num_layers, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim, padding_idx=pad_idx
        )
        # d_model: the input dimension
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.feature1 = nn.Linear(33, 128)
        self.feature2 = nn.Linear(128, 32)
        self.fc = nn.Linear(embedding_dim + 32, num_classes)

    def forward(self, x, features):

        embeddings = self.embedding(x)
        src_mask = x != 1
        #print(embeddings.shape, src_mask.shape )
        x = self.transformer_encoder(embeddings, src_key_padding_mask=None)
        x = x.mean(dim=1)

        features = features.to(torch.float32)
        features = self.feature1(features)
        features = self.feature2(features)
        x = torch.cat((x, features),dim=1)
        x = F.normalize(x, dim=-1)

        return self.fc(x)