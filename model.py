import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(config.hidden_size, 4 * config.hidden_size),
                                 nn.GELU(),
                                 nn.Linear(4 * config.hidden_size, config.hidden_size),
                                 nn.Dropout(config.mlp_pdrop))

    def forward(self, x):
        return self.mlp(x)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadSelfAttention, self).__init__()
        # query, key, value
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)

        # dropouts
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)

        # linear projection after attention
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)

        assert config.hidden_size % config.n_heads == 0, "Hidden size should be multiple of n_heads"
        self.n_heads = config.n_heads
        self.head_size = config.hidden_size // self.n_heads

    def forward(self, x):
        batch_size, seq_length, hidden_size = x.size()
        q = self.query(x).view(batch_size, seq_length, self.n_heads, self.head_size).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_length, self.head_size, self.n_heads).transpose(1, 3)
        v = self.value(x).view(batch_size, seq_length, self.n_heads, self.head_size).transpose(1, 2)

        attention_mask = torch.full((seq_length, seq_length), -float('inf'), device=x.device, dtype=x.dtype)
        attention_mask = torch.triu(attention_mask, diagonal=1)
        attention_score = torch.matmul(q, k) / math.sqrt(self.head_size) + attention_mask
        attention_score = F.softmax(attention_score, dim=-1)
        attention_score = self.attn_drop(attention_score)

        score = torch.matmul(attention_score, v)
        score = score.transpose(1, 2).contiguous().view(batch_size, seq_length, hidden_size)
        score = self.proj(score)
        score = self.resid_drop(score)
        return score


class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.attention = MultiHeadSelfAttention(config)
        self.mlp = MLP(config)

        # layer normalization
        self.ln_1 = nn.LayerNorm(config.hidden_size)
        self.ln_2 = nn.LayerNorm(config.hidden_size)

    def forward(self, x):
        x = self.ln_1(x)
        x = self.attention(x) + x
        x = self.ln_2(x)
        x = self.mlp(x) + x
        return x


class Embeddings(nn.Module):
    def __init__(self, config):
        super(Embeddings, self).__init__()
        # embedding layers
        self.pix_embed = nn.Embedding(config.n_clusters, config.hidden_size)
        self.pos_embed = nn.Embedding(config.n_pixels ** 2, config.hidden_size)
        self.embed_drop = nn.Dropout(config.embed_pdrop)

    def forward(self, x):
        seq_length = x.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand_as(x)

        pix_embeddings = self.pix_embed(x)
        pos_embeddings = self.pos_embed(position_ids)
        embeddings = self.embed_drop(pix_embeddings + pos_embeddings)
        return embeddings


class ImageGPT(nn.Module):
    def __init__(self, config):
        super(ImageGPT, self).__init__()
        # start token
        self.start_of_image = torch.nn.Parameter(torch.zeros(config.hidden_size))
        nn.init.normal_(self.start_of_image)

        # embedding layers
        self.embedding = Embeddings(config)

        # transformer blocks
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layers)])

        # decoder head
        self.ln_f = nn.LayerNorm(config.hidden_size)
        self.head = nn.Linear(config.hidden_size, config.n_clusters, bias=False)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        # prepend sos token
        start = self.start_of_image.repeat(x.size(0), 1).unsqueeze(1)
        h = self.embedding(x)
        h = torch.cat((start, h[:, :-1, :]), dim=1)

        x = self.blocks(h)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits


class ImageGPTConfig:
    # name
    name = 'xs'

    # training
    total_steps = 100000
    warmup_steps = 500
    batch_size = 2
    learning_rate = 0.0001
    weight_decay = 0.1
    betas = [0.9, 0.95]

    # architecture
    hidden_size = 8
    n_heads = 2
    n_layers = 4
    n_pixels = 32
    n_clusters = 6

    # dropout probabilities
    mlp_pdrop = 0.5
    attn_pdrop = 0.5
    resid_pdrop = 0.5
    embed_pdrop = 0.5

    def __init__(self, config):
        for k, v in config.items():
            setattr(self, k, v)

