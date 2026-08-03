import argparse
import os

import numpy as np
import torch
from PIL import Image
import OpenEXR

from train import load_split, render, psnr, render_normals


def load_gt_normal(data_dir, light, img_name, downsample):
    exr_name = f"suzanne{os.path.splitext(img_name)[0]}.exr"
    exr_path = os.path.join(data_dir, "train", f"light{light}", "images_exr", exr_name)
    exr_file = OpenEXR.File(exr_path)
    normal_part = next(p for p in exr_file.parts if p.name() == "normal")
    normal = np.stack([
        normal_part.channels["normal.X"].pixels,
        normal_part.channels["normal.Y"].pixels,
        normal_part.channels["normal.Z"].pixels,
    ], axis=-1)  # (H, W, 3), roughly unit-length, world-space

    if downsample > 1:
        h, w = normal.shape[:2]
        channels = []
        for c in range(3):
            img_c = Image.fromarray(normal[..., c], mode="F")
            img_c = img_c.resize((w // downsample, h // downsample), Image.Resampling.BOX)
            channels.append(np.array(img_c))
        normal = np.stack(channels, axis=-1)

    normal_01 = (normal+1)/2
    return torch.from_numpy(normal_01)


def load_target(args, view, device):
    if args.render_type == "beauty":
        return view["image"].to(device)
    elif args.render_type == "normals":
        return load_gt_normal(args.data_dir, args.light, view["img_name"], args.downsample).to(device)

def parse_args():
    p = argparse.ArgumentParser(description="Render + compare a checkpoint against ground truth.")
    p.add_argument("--data_dir", type=str, default="checkered_suzanne")
    p.add_argument("--light", type=int, default=1, choices=[1, 2, 3, 4])
    p.add_argument("--held_out", type=int, default=15,)
    p.add_argument("--downsample", type=int, default=2)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--num_held_out", type=int, default=5)
    p.add_argument("--out_dir", type=str, default="render_compare_out")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--render_type", type=str, default="beauty")
    return p.parse_args()


def to_uint8(img_tensor):
    # img_tensor is (H, W, 3) float in [0, 1] (clamped in case training over/undershot slightly
    # at some pixels -- rasterization doesn't guarantee outputs stay perfectly in-range).
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
    out_dir = os.path.join(args.out_dir, args.render_type)
    os.makedirs(out_dir, exist_ok=True)
    render_type_to_fcn = {"beauty": render, "normals": render_normals}
    render_fcn = render_type_to_fcn[args.render_type]

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
        pred = render_fcn(params, train_view, device)
        target = load_target(args, train_view, device)
        train_psnr = psnr(pred, target).item()
        save_side_by_side(pred, target, os.path.join(out_dir, "train_view_0000.png"))
        print(f"[render] train view 0: PSNR={train_psnr:.2f} dB "
              f"-> {out_dir}/train_view_0000.png")

        held_out_psnrs = []
        for i, view in enumerate(held_out_views[:args.num_held_out]):
            pred = render_fcn(params, view, device)
            target = load_target(args, view, device)
            view_psnr = psnr(pred, target).item()
            held_out_psnrs.append(view_psnr)
            out_path = os.path.join(out_dir, f"held_out_view_{i:04d}.png")
            save_side_by_side(pred, target, out_path)
            print(f"[render] held-out view {i}: PSNR={view_psnr:.2f} dB -> {out_path}")

    print(f"\n[summary] train view PSNR: {train_psnr:.2f} dB")
    print(f"[summary] held-out mean PSNR (this subset): {sum(held_out_psnrs) / len(held_out_psnrs):.2f} dB")


if __name__ == "__main__":
    main()
