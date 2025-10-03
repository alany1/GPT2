from contextlib import nullcontext

import torch

from datasets.shakespeare import Shakespeare
from params_proto import ParamsProto
from model.transformer import GPT
from model.util import count_parameters
from datetime import datetime

class Args(ParamsProto):
    data_dir = "/Users/alanyu/fortyfive/nanoGPT/data/shakespeare_char/"

    batch_size = 12
    grad_accumulation_steps = 40

    learning_rate = 6e-4

    vocab_size = 50304
    context_len = 1024
    num_layers = 8
    num_heads = 8
    embed_dim = 512
    dropout = 0.1
    n_epoch = 2_000
    
    device = "cpu"

def main(**deps):
    Args._update(**deps)
    trainset = Shakespeare(
        data_dir=Args.data_dir,
        target_size=Args.batch_size * Args.grad_accumulation_steps,
        train=True,
        context_len=Args.context_len,
    )

    valset = Shakespeare(
        data_dir=Args.data_dir,
        target_size=Args.batch_size * Args.grad_accumulation_steps // 2,
        train=False,
        context_len=Args.context_len,
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=Args.batch_size,
        shuffle=True,
    )

    valloader = torch.utils.data.DataLoader(
        valset,
        batch_size=Args.batch_size,
        shuffle=False,
    )

    gpt = GPT(
        embed_dim=Args.embed_dim,
        context_len=Args.context_len,
        vocab_size=Args.vocab_size,
        num_heads=Args.num_heads,
        num_layers=Args.num_layers,
        dropout=Args.dropout,
    )
    
    print(f"Params: {count_parameters(gpt, True) / 1e6:.2f} M")

    dtype = (
        "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
    )
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    ctx = nullcontext() if Args.device == "cpu" else torch.amp.autocast(device_type=Args.device, dtype=ptdtype)

    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

    optimizer = torch.optim.Adam(gpt.parameters(), lr=Args.learning_rate)

    for epoch in range(Args.n_epoch):
        gpt.eval()
        with torch.inference_mode():
            for x, y in valloader:
                with ctx:
                    logits = gpt(x)
                loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            print(f"epoch {epoch} val loss: {loss.item():.4f}")

        gpt.train()
        optimizer.zero_grad()

        for x, y in trainloader:
            with ctx:
                logits = gpt(x)
            
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss = loss / Args.grad_accumulation_steps

            scaler.scale(loss).backward()
            
        scaler.step(optimizer)
        scaler.update()

        optimizer.step()
        print(f"epoch {epoch} train loss: {loss.item():.4f}")

    log_dir = f"../outputs/gpt_shakespeare/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    import os
    os.makedirs(log_dir, exist_ok=True)
    torch.save(gpt.state_dict(), f"{log_dir}/last.pt")
    
    with open(f"{log_dir}/args.pkl", 'wb') as f:
        import pickle
        pickle.dump(dict(**vars(Args)), f)

if __name__ == "__main__":
    main(
        context_len=64,
        batch_size=12,
        num_layers=4,
        num_heads=4,
        vocab_size=65,
        embed_dim=128,
        n_epoch=2_000,
        dropout=0.0,
        device="mps",
        grad_accumulation_steps=1,
    )
