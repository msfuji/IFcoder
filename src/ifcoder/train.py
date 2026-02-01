import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import anndata as ad

from ifcoder.models import ConvVAE, vae_loss


# -------------------------
# Dataset
# -------------------------

class AnnDataPatches(Dataset):
    def __init__(self, h5ad_path: str, normalize: str = "per_channel"):
        adata = ad.read_h5ad(h5ad_path)

        if "patches" not in adata.obsm:
            raise ValueError("AnnData must contain obsm['patches']")

        x = adata.obsm["patches"]  # (N, C, H, W)
        if x.ndim != 4:
            raise ValueError(f"Expected patches with shape (N,C,H,W). Got {x.shape}")

        x = x.astype(np.float32)

        # Normalization
        if normalize == "per_channel":
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
        self.adata = adata  # keep reference for later

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx]


# -------------------------
# Training
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Input .h5ad produced by ifcoder.extract")
    ap.add_argument("--out", default="embeddings.h5ad", help="Output .h5ad with embeddings")
    ap.add_argument("--out-ckpt", default=None, help="Optional checkpoint .pt file")
    ap.add_argument("--patch-size", type=int, default=None, help="Patch size (infer if omitted)")
    ap.add_argument("--latent-dim", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument(
        "--normalize",
        default="per_channel",
        choices=["per_channel", "per_patch", "none"],
    )
    args = ap.parse_args()

    # -------------------------
    # load dataset
    # -------------------------
    ds = AnnDataPatches(args.data, normalize=args.normalize)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )

    # infer shape
    x0 = ds[0]
    C, H, W = x0.shape
    if H != W:
        raise ValueError(f"Expected square patches. Got {H}x{W}")
    patch_size = args.patch_size or H

    # -------------------------
    # model
    # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvVAE(
        in_channels=C,
        patch_size=patch_size,
        latent_dim=args.latent_dim,
        hidden=args.hidden,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # -------------------------
    # training loop
    # -------------------------
    model.train()
    for epoch in range(1, args.epochs + 1):
        loss_sum = recon_sum = kl_sum = 0.0
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

        print(
            f"epoch {epoch:03d}  "
            f"loss={loss_sum/n:.5f}  "
            f"recon={recon_sum/n:.5f}  "
            f"kl={kl_sum/n:.5f}"
        )

    # -------------------------
    # compute embeddings
    # -------------------------
    model.eval()
    all_mu = []

    with torch.no_grad():
        for i in range(0, len(ds), args.batch_size):
            x = ds.x[i : i + args.batch_size].to(device)
            mu, _ = model.encode(x)
            all_mu.append(mu.cpu().numpy())

    emb = np.concatenate(all_mu, axis=0)

    # -------------------------
    # write embeddings back to AnnData
    # -------------------------
    adata = ds.adata
    adata.obsm["X_ifcoder"] = emb

    adata.uns.setdefault("ifcoder", {})
    adata.uns["ifcoder"].update(
        {
            "model": "ConvVAE",
            "latent_dim": args.latent_dim,
            "hidden": args.hidden,
            "beta": args.beta,
            "normalize": args.normalize,
        }
    )

    adata.write_h5ad(args.out)
    print(f"Saved embeddings -> {args.out}  shape={emb.shape}")

    # -------------------------
    # optional checkpoint
    # -------------------------
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
