import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from ifcoder.models import ConvVAE, vae_loss
from ifcoder.io import export_anndata


class NPZPatches(Dataset):
    def __init__(self, npz_path: str, normalize: str = "per_channel"):
        d = np.load(npz_path, allow_pickle=True)
        x = d["patches"]  # (N,C,H,W)
        if x.ndim != 4:
            raise ValueError(f"Expected patches with shape (N,C,H,W). Got {x.shape}")

        self.image_numbers = d["image_numbers"] if "image_numbers" in d else None
        self.centers = d["centers"] if "centers" in d else None

        x = x.astype(np.float32)

        # Normalization choices:
        # - per_channel: scale each channel to [0,1] based on global max
        # - per_patch: scale each patch to [0,1] based on patch max
        if normalize == "per_channel":
            # robust-ish: use per-channel max over all pixels
            ch_max = x.reshape(x.shape[0], x.shape[1], -1).max(axis=(0, 2))
            ch_max = np.clip(ch_max, 1e-6, None)
            x = x / ch_max[None, :, None, None]
        elif normalize == "per_patch":
            pmax = x.reshape(x.shape[0], -1).max(axis=1)
            pmax = np.clip(pmax, 1e-6, None)
            x = x / pmax[:, None, None, None]
        elif normalize == "none":
            pass
        else:
            raise ValueError("normalize must be one of: per_channel, per_patch, none")

        self.x = torch.from_numpy(x)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Input .npz produced by ifcoder.extract")
    ap.add_argument("--out-emb", default="embeddings.h5ad", help="Output embeddings .h5ad")
    ap.add_argument("--out-ckpt", default=None, help="Optional checkpoint .pt file")
    ap.add_argument("--patch-size", type=int, default=None, help="Patch size (infer if omitted)")
    ap.add_argument("--latent-dim", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--normalize", default="per_channel", choices=["per_channel", "per_patch", "none"])
    args = ap.parse_args()

    ds = NPZPatches(args.data, normalize=args.normalize)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False)

    # Infer shape
    x0 = ds[0]
    C, H, W = x0.shape
    if H != W:
        raise ValueError(f"Expected square patches. Got {H}x{W}")
    patch_size = args.patch_size or H

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvVAE(in_channels=C, patch_size=patch_size, latent_dim=args.latent_dim, hidden=args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(1, args.epochs + 1):
        loss_sum = 0.0
        recon_sum = 0.0
        kl_sum = 0.0
        n = 0

        for x in dl:
            x = x.to(device)
            x_hat, mu, logvar = model(x)
            loss, recon, kl = vae_loss(x, x_hat, mu, logvar, beta=args.beta)

            opt.zero_grad()
            loss.backward()
            opt.step()

            bs = x.size(0)
            loss_sum += loss.detach().item() * bs
            recon_sum += recon.detach().item() * bs
            kl_sum += kl.detach().item() * bs
            n += bs

        print(f"epoch {epoch:03d}  loss={loss_sum/n:.5f}  recon={recon_sum/n:.5f}  kl={kl_sum/n:.5f}")

    # Compute embeddings (mu) for all patches
    model.eval()
    all_mu = []
    with torch.no_grad():
        for i in range(0, len(ds), args.batch_size):
            x = ds.x[i : i + args.batch_size].to(device)
            mu, logvar = model.encode(x)
            all_mu.append(mu.cpu().numpy())
    emb = np.concatenate(all_mu, axis=0)
    # np.savez_compressed(args.emb_out, mu=emb)
    # print(f"Saved embeddings -> {args.emb_out}  shape={emb.shape}")

    # Build obs
    obs = {}
    if ds.image_numbers is not None:
        obs["image_numbers"] = ds.image_numbers
    if ds.centers is not None:
        obs["center_x"] = ds.centers[:, 0]
        obs["center_y"] = ds.centers[:, 1]

    obs = pd.DataFrame(obs) if obs else None

    # Export embedding in anndata format
    export_anndata(
        embedding=emb,
        out=args.out_emb,
        obs=obs,
        obsm_key="X_ifcoder",
        uns={
            "model": "ConvVAE",
            "latent_dim": args.latent_dim,
            "normalize": args.normalize,
        },
    )

    # Optional: save checkpoint
    if args.out_ckpt:
        ckpt = {
            "state_dict": model.state_dict(),
            "in_channels": C,
            "patch_size": patch_size,
            "latent_dim": args.latent_dim,
            "hidden": args.hidden,
            "normalize": args.normalize,
        }
        torch.save(ckpt, args.out_ckpt)
        print(f"Saved model -> {args.out_ckpt}")


if __name__ == "__main__":
    main()
