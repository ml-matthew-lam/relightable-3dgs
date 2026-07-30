import argparse
import json
import math
import os
import random

import numpy as np
import torch
from PIL import Image

from gsplat import rasterization


# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Baseline vanilla 3DGS training on one light group.")
    p.add_argument("--data_dir", type=str, default="checkered_suzanne",
                    help="Path to the dataset root (the folder containing train/ and test/).")
    p.add_argument("--light", type=int, default=1, choices=[1, 2, 3, 4],
                    help="Which train/lightN/ group to train on. Per CLAUDE.md, do NOT mix "
                         "multiple lights into one vanilla-SH run.")
    p.add_argument("--held_out", type=int, default=15,
                    help="Number of views held out from the END of this light's 100-view list "
                         "for same-light novel-view PSNR eval. Frame order is not spatially "
                         "sorted (random camera seed per light group, verified against the "
                         "actual transforms.json), so a contiguous tail block gives the same "
                         "view-coverage spread as a random subset would.")
    p.add_argument("--downsample", type=int, default=2,
                    help="Integer factor to shrink images by (e.g. 2 = half resolution). "
                         "EXR native res is 1400x1400 / PNG matches; full res is unnecessarily "
                         "slow for a plumbing sanity check. Camera intrinsics are rebuilt from "
                         "the downsampled resolution directly (FOV is resolution-independent), "
                         "so no separate intrinsics-scaling math is needed.")
    p.add_argument("--num_init_points", type=int, default=20000,
                    help="How many Gaussians to randomly initialize. No adaptive densification "
                         "yet (see module docstring), so this is the Gaussian count for the "
                         "entire run — set high enough to have a chance at resolving detail.")
    p.add_argument("--scene_extent", type=float, default=1.5,
                    help="Half-width (in world units) of the cube that random init points are "
                         "sampled from, centered at the origin. ASSUMPTION, not measured: "
                         "default Blender Suzanne spans roughly +/-1.2 units unscaled. Sanity "
                         "check this against your actual Blender scene scale before trusting it.")
    p.add_argument("--iters", type=int, default=7000,
                    help="Total optimization steps.")
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--ckpt_every", type=int, default=500,
                    help="Colab sessions can disconnect any time (idle timeout / hard runtime "
                         "cap) — checkpoint frequently enough that a disconnect doesn't cost "
                         "much progress.")
    p.add_argument("--ckpt_dir", type=str, default="checkpoints/light1_baseline")
    p.add_argument("--resume", action="store_true",
                    help="If set, load the latest checkpoint in --ckpt_dir and continue instead "
                         "of starting fresh.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def set_seed(seed):
    # fixing python/numpy/torch RNGs so we get deterministic behaviour
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_split(data_dir, light, downsample):
    """Loads one light group's transforms.json + PNGs, returns a list of
    per-view dicts ready for the render/loss loop."""
    light_dir = os.path.join(data_dir, "train", f"light{light}")
    with open(os.path.join(light_dir, "transforms.json")) as f:
        meta = json.load(f)

    camera_angle_x = meta["camera_angle_x"]  # horizontal FOV in radians

    views = []
    for frame in meta["frames"]:
        img_path = os.path.join(light_dir, frame["file_path"])
        # BlenderNeRF's file_path is relative and already includes the
        # extension/subfolder convention it wrote; PNGs live in images_png/
        # but file_path as stored points at e.g. "train/0001.png" -- join
        # against light_dir and swap in the images_png subfolder.
        img_name = os.path.basename(frame["file_path"])
        img_path = os.path.join(light_dir, "images_png", img_name)

        img = Image.open(img_path).convert("RGB")
        if downsample > 1:
            new_size = (img.width // downsample, img.height // downsample)
            img = img.resize(new_size, Image.Resampling.BOX)  # box filter = correct average-downsample
        width, height = img.size
        img_t = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0)  # HWC, [0,1]

        # --- Camera intrinsics ---
        # Pinhole model: focal length in pixels derived from horizontal FOV
        # and the (downsampled) image width.
        focal = 0.5 * width / math.tan(0.5 * camera_angle_x)
        Ks = torch.tensor([
            [focal, 0.0, width / 2.0],
            [0.0, focal, height / 2.0],
            [0.0, 0.0, 1.0],
        ], dtype=torch.float32)

        # --- Camera pose ---
        # convert matrices from transforms.json to gsplat's conventions
        c2w_gl = torch.tensor(frame["transform_matrix"], dtype=torch.float32)
        flip = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0]))
        c2w_cv = c2w_gl @ flip
        viewmat = torch.inverse(c2w_cv)

        views.append({
            "image": img_t,
            "viewmat": viewmat,
            "Ks": Ks,
            "width": width,
            "height": height,
        })
    return views


# ---------------------------------------------------------------------------
# Gaussian model
# ---------------------------------------------------------------------------
def init_gaussians(num_points, scene_extent, device):
    # means: uniform random points in a cube centered at the origin
    means = (torch.rand(num_points, 3, device=device) * 2 - 1) * scene_extent

    # scales: stored as log-scale so exp(log_scales) is always positive;
    # initialized from the average nearest-neighbor spacing of the random
    # points, so Gaussians start out roughly touching
    # their neighbors rather than either invisibly tiny or overlapping into
    # a solid blob
    avg_spacing = (2 * scene_extent) / (num_points ** (1 / 3))
    log_scales = torch.full((num_points, 3), math.log(avg_spacing), device=device)

    # quats: identity rotation (w, x, y, z) = (1, 0, 0, 0) for every Gaussian;
    # not normalized to unit length here because normalization happens
    # at render time so the optimizer can move freely in raw parameter space
    quats = torch.zeros(num_points, 4, device=device)
    quats[:, 0] = 1.0

    # opacities: stored as logits so sigmoid(opacity_logits) is in (0, 1);
    # initialized to a relatively low starting opacity (~0.1) -- otherwise,
    # a few big Gaussians may dominate the loss early and get stuck
    init_opacity = 0.1
    opacity_logits = torch.full((num_points,), math.log(init_opacity / (1 - init_opacity)), device=device)

    # colors: SH degree 0 only, stored as raw logits;
    # sigmoid applied at render time to clamp values to [0, 1] RGB; 
    # random init so that different Gaussians start out visually distinguishable
    color_logits = torch.randn(num_points, 3, device=device) * 0.1

    params = {
        "means": means,
        "log_scales": log_scales,
        "quats": quats,
        "opacity_logits": opacity_logits,
        "color_logits": color_logits,
    }
    for v in params.values():
        v.requires_grad_(True)
    return params


def make_optimizer(params):
    # Per-parameter learning rates following the defaults from the 
    # original 3DGS paper by Kerbl et al.
    return torch.optim.Adam([
        {"params": [params["means"]], "lr": 1.6e-4, "name": "means"},
        {"params": [params["log_scales"]], "lr": 5e-3, "name": "log_scales"},
        {"params": [params["quats"]], "lr": 1e-3, "name": "quats"},
        {"params": [params["opacity_logits"]], "lr": 5e-2, "name": "opacity_logits"},
        {"params": [params["color_logits"]], "lr": 2.5e-3, "name": "color_logits"},
    ])


def render(params, view, device):
    """Runs one forward pass through gsplat's rasterizer for a single view."""
    means = params["means"]
    scales = torch.exp(params["log_scales"])
    quats = params["quats"]
    opacities = torch.sigmoid(params["opacity_logits"])
    colors = torch.sigmoid(params["color_logits"])

    # rasterization() supports batched cameras via a leading dim; we only
    # have one camera per call, so add a size-1 batch dim and squeeze after
    render_colors, render_alphas, _meta = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=view["viewmat"].unsqueeze(0).to(device),
        Ks=view["Ks"].unsqueeze(0).to(device),
        width=view["width"],
        height=view["height"],
        sh_degree=None,  # no SH evaluation -- colors are used directly as final RGB
    )
    return render_colors[0]


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def save_checkpoint(ckpt_dir, step, params, optimizer):
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"step_{step:06d}.pt")
    torch.save({
        "step": step,
        "params": {k: v.detach().cpu() for k, v in params.items()},
        "optimizer": optimizer.state_dict(),
    }, path)
    print(f"[checkpoint] saved {path}")


def load_latest_checkpoint(ckpt_dir, device):
    if not os.path.isdir(ckpt_dir):
        return None
    ckpts = sorted(f for f in os.listdir(ckpt_dir) if f.startswith("step_") and f.endswith(".pt"))
    if not ckpts:
        return None
    path = os.path.join(ckpt_dir, ckpts[-1])
    print(f"[checkpoint] resuming from {path}")
    ckpt = torch.load(path, map_location=device)
    for v in ckpt["params"].values():
        v.requires_grad_(True)
    return ckpt


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------
def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    return -10.0 * torch.log10(mse)


def evaluate(params, held_out_views, device):
    with torch.no_grad():
        total_psnr = 0.0
        for view in held_out_views:
            pred = render(params, view, device)
            target = view["image"].to(device)
            total_psnr += psnr(pred, target).item()
        return total_psnr / len(held_out_views)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    print(f"[setup] device={device}, light=light{args.light}, iters={args.iters}")

    all_views = load_split(args.data_dir, args.light, args.downsample)
    held_out_views = all_views[-args.held_out:]
    train_views = all_views[:-args.held_out]
    print(f"[data] {len(train_views)} train views, {len(held_out_views)} held-out views")

    ckpt = load_latest_checkpoint(args.ckpt_dir, device) if args.resume else None
    if ckpt is not None:
        params = {k: v.to(device) for k, v in ckpt["params"].items()}
        start_step = ckpt["step"] + 1
        optimizer = make_optimizer(params)
        optimizer.load_state_dict(ckpt["optimizer"])
    else:
        params = init_gaussians(args.num_init_points, args.scene_extent, device)
        start_step = 0
        optimizer = make_optimizer(params)

    for step in range(start_step, args.iters):
        view = train_views[random.randrange(len(train_views))]
        pred = render(params, view, device)
        target = view["image"].to(device)

        loss = torch.abs(pred - target).mean()  # plain L1 -- to be updated later

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"[train] step {step}/{args.iters}  loss={loss.item():.4f}")

        if step > 0 and step % args.eval_every == 0:
            mean_psnr = evaluate(params, held_out_views, device)
            print(f"[eval] step {step}  held-out PSNR={mean_psnr:.2f} dB")

        if step > 0 and step % args.ckpt_every == 0:
            save_checkpoint(args.ckpt_dir, step, params, optimizer)

    save_checkpoint(args.ckpt_dir, args.iters, params, optimizer)
    final_psnr = evaluate(params, held_out_views, device)
    print(f"[done] final held-out PSNR={final_psnr:.2f} dB")


if __name__ == "__main__":
    main()
