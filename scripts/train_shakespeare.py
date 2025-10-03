from contextlib import nullcontext

import torch
import matplotlib.pyplot as plt

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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Args.n_epoch, eta_min=Args.learning_rate * 0.1
    )

    train_losses = []
    val_losses = []

    for epoch in range(Args.n_epoch):
        gpt.eval()
        with torch.inference_mode():
            for x, y in valloader:
                with ctx:
                    logits = gpt(x)
                val_loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            print(f"epoch {epoch} val loss: {val_loss.item():.4f}")
            val_losses.append(val_loss.item())

        gpt.train()
        optimizer.zero_grad()

        for x, y in trainloader:
            with ctx:
                logits = gpt(x)

            train_loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss = train_loss / Args.grad_accumulation_steps

            scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()

        optimizer.step()
        print(f"epoch {epoch} train loss: {train_loss.item():.4f}")
        train_losses.append(train_loss.item())

        scheduler.step()

    log_dir = f"../outputs/gpt_shakespeare/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    import os
    os.makedirs(log_dir, exist_ok=True)
    torch.save(gpt.state_dict(), f"{log_dir}/last.pt")

    with open(f"{log_dir}/args.pkl", 'wb') as f:
        import pickle
        pickle.dump(dict(**vars(Args)), f)

    # Plot loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', alpha=0.8)
    plt.plot(val_losses, label='Val Loss', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{log_dir}/loss_curves.png", dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main(
        context_len=64,
        batch_size=12,
        num_layers=4,
        num_heads=4,
        vocab_size=65,
        embed_dim=128,
        n_epoch=2_000,
        # n_epoch=10,
        dropout=0.0,
        device="mps",
        grad_accumulation_steps=1,
    )
