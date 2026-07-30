"""
Visual sanity check for a trained checkpoint from train.py.

PSNR alone can hide problems (e.g. a blurry "average color" reconstruction
can still score decently on a simple scene without geometry actually being
right). This script renders held-out views (plus one training view, for
reference) from a checkpoint and saves them next to their ground-truth
images so you can actually look at them side by side.

Reuses load_split()/render() from train.py directly rather than
reimplementing the camera/rasterization logic a second time — same code
path as training, so what you're looking at is exactly what the loss was
computed on.
"""

import argparse
import os

import numpy as np
import torch
from PIL import Image

from train import load_split, render, psnr


def parse_args():
    p = argparse.ArgumentParser(description="Render + compare a checkpoint against ground truth.")
    p.add_argument("--data_dir", type=str, default="checkered_suzanne")
    p.add_argument("--light", type=int, default=1, choices=[1, 2, 3, 4])
    p.add_argument("--held_out", type=int, default=15,
                    help="Must match what the checkpoint was trained with, so the held-out split "
                         "lines up with views the model never saw.")
    p.add_argument("--downsample", type=int, default=2,
                    help="Must match the training run's --downsample — camera intrinsics are "
                         "resolution-dependent, so a mismatch here would render at a size the "
                         "checkpoint wasn't optimized for.")
    p.add_argument("--ckpt", type=str, required=True, help="Path to a specific step_XXXXXX.pt file.")
    p.add_argument("--num_held_out", type=int, default=5,
                    help="How many held-out views to render (out of --held_out total). Default 5 "
                         "is enough to eyeball, doesn't need to be all 15.")
    p.add_argument("--out_dir", type=str, default="render_compare_out")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def to_uint8(img_tensor):
    # img_tensor is (H, W, 3) float in [0, 1] (clamped in case training over/undershot slightly
    # at some pixels — rasterization doesn't guarantee outputs stay perfectly in-range).
    arr = img_tensor.clamp(0, 1).detach().cpu().numpy()
    return (arr * 255).astype(np.uint8)


def save_side_by_side(pred, target, path):
    pred_img = Image.fromarray(to_uint8(pred))
    target_img = Image.fromarray(to_uint8(target))
    w, h = pred_img.size
    combined = Image.new("RGB", (w * 2, h))
    combined.paste(target_img, (0, 0))   # left = ground truth
    combined.paste(pred_img, (w, 0))     # right = prediction
    combined.save(path)


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[load] checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device)
    params = {k: v.to(device) for k, v in ckpt["params"].items()}
    print(f"[load] checkpoint was saved at step {ckpt['step']}")

    all_views = load_split(args.data_dir, args.light, args.downsample)
    held_out_views = all_views[-args.held_out:]
    train_views = all_views[:-args.held_out]

    with torch.no_grad():
        # One training view, for reference alongside the held-out renders below.
        train_view = train_views[0]
        pred = render(params, train_view, device)
        target = train_view["image"].to(device)
        train_psnr = psnr(pred, target).item()
        save_side_by_side(pred, target, os.path.join(args.out_dir, "train_view_0000.png"))
        print(f"[render] train view 0: PSNR={train_psnr:.2f} dB "
              f"-> {args.out_dir}/train_view_0000.png (left=ground truth, right=prediction)")

        held_out_psnrs = []
        for i, view in enumerate(held_out_views[:args.num_held_out]):
            pred = render(params, view, device)
            target = view["image"].to(device)
            view_psnr = psnr(pred, target).item()
            held_out_psnrs.append(view_psnr)
            out_path = os.path.join(args.out_dir, f"held_out_view_{i:04d}.png")
            save_side_by_side(pred, target, out_path)
            print(f"[render] held-out view {i}: PSNR={view_psnr:.2f} dB -> {out_path}")

    print(f"\n[summary] train view PSNR: {train_psnr:.2f} dB")
    print(f"[summary] held-out mean PSNR (this subset): {sum(held_out_psnrs) / len(held_out_psnrs):.2f} dB")


if __name__ == "__main__":
    main()
