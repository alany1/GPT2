import pickle
import torch
from model.util import pick

def get_encode_decode(data_dir):
    meta_path = f"{data_dir}/meta.pkl"
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    stoi, itos = meta["stoi"], meta["itos"]
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])

    return encode, decode


def load_model(log_dir, device):
    from model.transformer import GPT

    ckpt = f"{log_dir}/last.pt"
    sd = torch.load(ckpt, map_location=device)

    with open(f"{log_dir}/args.pkl", "rb") as f:
        args = pickle.load(f)

    model = GPT(
        **pick(
            args,
            "embed_dim",
            "context_len",
            "vocab_size",
            "num_heads",
            "num_layers",
            "dropout",
        )
    )
    
    model.load_state_dict(sd)
    model.to(device)
    
    return model, args

def main(*, log_dir, device, num_tokens, num_samples):
    model, args = load_model(log_dir, device)
    encode, decode = get_encode_decode(args["data_dir"])
    
    sos = "\n"
    context = torch.tensor(encode(sos)).long().to(device)[None, ...]

    for _ in range(num_samples):
        result = model.sample(context, num_tokens)
        print(decode(result[0].tolist()))
        print("--------------------------------------------")

if __name__ == '__main__':
    main(
        log_dir="../outputs/gpt_shakespeare/20251004-065016",
        device="cpu",
        num_tokens=500,
        num_samples=10,
    )
