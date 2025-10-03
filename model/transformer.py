import torch
from torch import nn
from torchtyping import TensorType

from model.util import count_parameters


class MultiHeadAttention(nn.Module):
    def __init__(self, *, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # get QKV
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

    def forward(self, x: TensorType["b", "context_len", "embed_dim"]):
        b, context_len, embed_dim = x.shape

        qkv = self.qkv(x)  # (b, context_len, 3 * embed_dim)

        # todo: double check shapes
        all_q = (
            qkv[:, :, :embed_dim].view(b, context_len, self.num_heads, self.head_dim).transpose(1, 2)
        )  # (b, num_heads, context_len, head_dim)
        all_k = qkv[:, :, embed_dim : 2 * embed_dim].view(b, context_len, self.num_heads, self.head_dim).transpose(1, 2)
        all_v = qkv[:, :, 2 * embed_dim :].view(b, context_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (all_q @ all_k.transpose(-2, -1)) / (self.head_dim**0.5)  # (b, num_heads, context_len, context_len)

        # mask
        mask = torch.tril(torch.ones(context_len, context_len, device=x.device)).unsqueeze(0).unsqueeze(0)
        attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = torch.softmax(attn, dim=-1)

        out = attn @ all_v  # (b, num_heads, context_len, head_dim)
        out = out.transpose(1, 2).contiguous().view(b, context_len, embed_dim)  # (b, context_len, embed_dim)

        return out


class TransformerBlock(nn.Module):
    """
    Transformer block with (masked) multi-head self-attn.
    """

    def __init__(self, *, embed_dim, num_heads, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

        self.attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.fc = nn.Linear(embed_dim, embed_dim * 4, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(embed_dim * 4, embed_dim, bias=True)

        self.ff = nn.Sequential(
            self.fc,
            self.act,
            self.fc2,
        )

    def forward(self, x: TensorType["b", "context_len", "embed_dim"]):
        x = self.ln1(x + self.dropout(self.attn(x)))
        x = self.ln2(x + self.dropout(self.ff(x)))

        return x


class Transformer(nn.Module):
    def __init__(
        self,
        *,
        embed_dim,
        context_len,
        vocab_size,
        num_heads,
        num_layers,
        dropout=0.1,
    ):
        super(Transformer, self).__init__()

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(context_len, embed_dim)

        self.pe_dropout = nn.Dropout(p=dropout)

        self.blocks = nn.ModuleList()

        for _ in range(num_layers):
            self.blocks.append(
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
            )

        self.num_layers = num_layers

    def forward(self, tokens: TensorType["b", "context_len"]):
        x = self.token_embedding(tokens) + self.pos_embedding(torch.arange(tokens.size(1), device=tokens.device))[None, ...]
        x = self.pe_dropout(x)

        for i in range(self.num_layers):
            x = self.blocks[i](x)

        return x


class GPT(nn.Module):
    def __init__(
        self,
        embed_dim,
        context_len,
        vocab_size,
        num_heads,
        num_layers,
        dropout,
    ):
        super(GPT, self).__init__()
        self.transformer = Transformer(
            embed_dim=embed_dim,
            context_len=context_len,
            vocab_size=vocab_size,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.linear = nn.Linear(embed_dim, vocab_size)

    def forward(self, tokens):
        x = self.transformer(tokens)  # b, context_len, embed_dim
        x = self.linear(x)
        # x = torch.nn.functional.softmax(x, dim=-1)

        return x


if __name__ == "__main__":
    gpt = GPT(
        embed_dim=512,
        context_len=128,
        vocab_size=50257,
        num_heads=8,
        num_layers=6,
        dropout=0.1,
    )

    fake_t = torch.randint(0, 50257, (4, 128))
    out = gpt(fake_t)
    print(f"Params: {count_parameters(gpt, True) / 1e6:.2f} M")
