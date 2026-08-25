import argparse
import os
import random

import numpy as np
import torch
from PIL import Image

from train import load_split, load_exr_aovs, render, render_normals, render_albedo, psnr, evaluate_aovs


def gt_exr_path(data_dir, view):
    frame_idx = os.path.splitext(view["img_name"])[0]  # "0001.png" -> "0001"
    return os.path.join(data_dir, view["subset_dir"], view["light_id"], "images_exr", f"suzanne{frame_idx}.exr")


def load_target(args, view, device):
    if args.render_type == "beauty":
        return view["image"].to(device), None
    gt_normal, gt_albedo = load_exr_aovs(gt_exr_path(args.data_dir, view), args.downsample)
    mask = (gt_albedo.sum(-1) > 1e-4).to(device)
    if args.render_type == "normals":
        target = ((gt_normal + 1) / 2).to(device)
    elif args.render_type == "albedo":
        target = gt_albedo.to(device)
    return target, mask


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="checkered_suzanne")
    p.add_argument("--test_light", action="store_true")    
    p.add_argument("--held_out", type=int, default=15)
    p.add_argument("--downsample", type=int, default=2)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--num_held_out", type=int, default=5)
    p.add_argument("--out_dir", type=str, default="render_compare_out")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--render_type", type=str, default="beauty", choices=["beauty", "normals", "albedo"])
    return p.parse_args()


def to_uint8(img_tensor):
    # img_tensor is (H, W, 3) float in [0, 1] (clamped in case training over/undershot slightly
    # at some pixels -- rasterization doesn't guarantee outputs stay perfectly in-range).
    arr = img_tensor.clamp(0, 1).detach().cpu().numpy()
    return (arr * 255).astype(np.uint8)


def save_side_by_side(pred, target, path, mask=None):
    if mask is not None:
        # zero out background on both sides so the saved comparison doesn't have a
        # mismatched background encoding between the ground truth and the predicted render
        pred = pred * mask.unsqueeze(-1)
        target = target * mask.unsqueeze(-1)
    pred_img = Image.fromarray(to_uint8(pred))
    target_img = Image.fromarray(to_uint8(target))
    w, h = pred_img.size
    combined = Image.new("RGB", (w * 2, h))
    combined.paste(target_img, (0, 0))   # left = ground truth
    combined.paste(pred_img, (w, 0))     # right = prediction
    combined.save(path)


def main():
    args = parse_args()
    random.seed(args.seed)
    device = torch.device(args.device)
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    render_type_to_fcn = {"beauty": render, "normals": render_normals, "albedo": render_albedo}
    render_fcn = render_type_to_fcn[args.render_type]

    print(f"[load] checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device)
    params = {k: v.to(device) for k, v in ckpt["params"].items()}
    print(f"[load] checkpoint was saved at step {ckpt['step']}")

    all_views = load_split(args.data_dir, args.downsample, test_light=args.test_light)
    random.shuffle(all_views)
    held_out_views = all_views[-args.held_out:]
    train_views = all_views[:-args.held_out]

    def run_render(view):
        # beauty depends on light position, whereas normals/albedo do not
        if args.render_type == "beauty":
            image, _info = render(params, view, view["lightcoords"], device)
            return image
        return render_fcn(params, view, device)

    with torch.no_grad():
        # one training view, for reference alongside the held-out renders below
        train_view = train_views[0]
        pred = run_render(train_view)
        target, mask = load_target(args, train_view, device)
        train_psnr = psnr(pred, target, mask).item()
        save_side_by_side(pred, target, os.path.join(out_dir, "train_view_0000.png"), mask)
        print(f"[render] train view 0 ({train_view['light_id']}): PSNR={train_psnr:.2f} dB "
              f"-> {out_dir}/train_view_0000.png")

        held_out_psnrs = []
        for i, view in enumerate(held_out_views[:args.num_held_out]):
            pred = run_render(view)
            target, mask = load_target(args, view, device)
            view_psnr = psnr(pred, target, mask).item()
            held_out_psnrs.append(view_psnr)
            out_path = os.path.join(out_dir, f"held_out_view_{i:04d}.png")
            save_side_by_side(pred, target, out_path, mask)
            print(f"[render] held-out view {i} ({view['light_id']}): PSNR={view_psnr:.2f} dB -> {out_path}")

    print(f"\n[summary] train view PSNR: {train_psnr:.2f} dB")
    print(f"[summary] held-out mean PSNR (sample of {len(held_out_psnrs)}): {sum(held_out_psnrs) / len(held_out_psnrs):.2f} dB")

    # Averages over every view in this split (all 30 test-light views when --test_light is passed)
    with torch.no_grad():
        full_psnrs = []
        for view in all_views:
            pred, _info = render(params, view, view["lightcoords"], device)
            target = view["image"].to(device)
            full_psnrs.append(psnr(pred, target).item())  # unmasked for beauty PSNR

        exr_views = load_split(args.data_dir, args.downsample, test_light=args.test_light, load_exr=True)
        albedo_l1, normal_err_deg = evaluate_aovs(params, exr_views, device)  # masked internally by evaluate_aovs

    print(f"\n[average metrics over {len(all_views)} views]")
    print(f"  avg beauty PSNR (full frame): {sum(full_psnrs) / len(full_psnrs):.2f} dB")
    print(f"  avg albedo L1 error (background masked): {albedo_l1:.4f}")
    print(f"  avg normal angular error (background masked): {normal_err_deg:.2f} deg")


if __name__ == "__main__":
    main()
