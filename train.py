import argparse
import json
import math
import os
import random

import numpy as np
import OpenEXR
import torch
from PIL import Image
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure
from gsplat.strategy import DefaultStrategy

from gsplat import rasterization


GS_KEYS = ("means", "scales", "quats", "opacities", "albedo_logits", "roughness_logits")


# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="checkered_suzanne")
    p.add_argument("--held_out", type=int, default=15)
    p.add_argument("--downsample", type=int, default=2)
    p.add_argument("--num_init_points", type=int, default=100000)
    p.add_argument("--scene_extent", type=float, default=1.5)
    p.add_argument("--iters", type=int, default=15000)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--ckpt_every", type=int, default=500)
    p.add_argument("--ckpt_dir", type=str, default="checkpoints")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--lambda_normal_smooth", type=float, default=0.01)
    p.add_argument("--lambda_albedo_smooth", type=float, default=0.01)
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
def load_exr_aovs(path, downsample):
    """Load normal and albedo AOVs from an EXR and return (normal, albedo)"""
    exr = OpenEXR.File(path)
    normal = albedo = None
    for part in exr.parts:
        if part.name() == "normal":
            c = part.channels
            normal = np.stack([c["normal.X"].pixels, c["normal.Y"].pixels, c["normal.Z"].pixels], axis=-1)
        elif part.name() == "albedo":
            albedo = part.channels["albedo"].pixels[..., :3]
    normal_t = torch.from_numpy(normal.copy())
    albedo_t = torch.from_numpy(albedo.copy())

    if downsample > 1:
        normal_t = F.avg_pool2d(normal_t.permute(2, 0, 1).unsqueeze(0), downsample).squeeze(0).permute(1, 2, 0)
        normal_t = F.normalize(normal_t, p=2, dim=-1)
        albedo_t = F.avg_pool2d(albedo_t.permute(2, 0, 1).unsqueeze(0), downsample).squeeze(0).permute(1, 2, 0)

    return normal_t, albedo_t


def load_split(data_dir, downsample, test_light=False, load_exr=False):
    subset_dir = "test" if test_light else "train"
    if test_light:
        with open(os.path.join(data_dir, subset_dir, "light5", "transforms.json")) as f:
            meta = json.load(f)
        for frame in meta["frames"]:
            frame["light_pos"] = "light5"
        camera_angle_x = meta["camera_angle_x"]  # horizontal FOV in radians
    else:
        meta = {"frames": []}
        camera_angle_x = None
        for i in range(1, 5):
            with open(os.path.join(data_dir, subset_dir, f"light{i}", "transforms.json")) as f:
                light_meta = json.load(f)
            for frame in light_meta["frames"]:
                frame["light_pos"] = f"light{i}"
            meta["frames"] += light_meta["frames"]
            camera_angle_x = light_meta["camera_angle_x"]  # assumed to be the same across all lights

    with open(os.path.join(data_dir, "light_metadata.json")) as f:
        light_metadata = json.load(f)

    views = []
    for frame in meta["frames"]:
        # BlenderNeRF's file_path is relative and already includes the
        # extension/subfolder convention it wrote; PNGs live in images_png/
        # but file_path as stored points at e.g. "train/0001.png" -- join
        # against data_dir/subset_dir/light_dir and swap in the images_png subfolder.
        img_name = os.path.basename(frame["file_path"])
        img_path = os.path.join(data_dir, subset_dir, frame["light_pos"], "images_png", img_name)

        img = Image.open(img_path).convert("RGB")
        if downsample > 1:
            new_size = (img.width // downsample, img.height // downsample)
            img = img.resize(new_size, Image.Resampling.BOX)
        width, height = img.size
        img_t = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0)

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

        lightcoords = light_metadata[frame["light_pos"]]["position"]

        view = {
            "image": img_t,
            "viewmat": viewmat,
            "lightcoords": lightcoords,
            "Ks": Ks,
            "width": width,
            "height": height,
            "img_name": img_name,  # e.g. "0001.png" -- used to locate the matching EXR for GT AOVs
            "light_id": frame["light_pos"],  # e.g. "light1"
            "subset_dir": subset_dir,  # "train" or "test"
        }

        if load_exr:
            frame_idx = os.path.splitext(img_name)[0]  # "0001.png" -> "0001"
            exr_path = os.path.join(data_dir, subset_dir, frame["light_pos"], "images_exr", f"suzanne{frame_idx}.exr")
            gt_normal, gt_albedo = load_exr_aovs(exr_path, downsample)
            view["gt_normal"] = gt_normal
            view["gt_albedo"] = gt_albedo

        views.append(view)
    return views


# ---------------------------------------------------------------------------
# Gaussian model
# ---------------------------------------------------------------------------
def init_gaussians(num_points, scene_extent, device):
    # means: uniform random points in a cube centered at the origin
    means = (torch.rand(num_points, 3, device=device) * 2 - 1) * scene_extent

    avg_spacing = (2 * scene_extent) / (num_points ** (1 / 3))
    scales = torch.full((num_points, 3), math.log(avg_spacing), device=device)  # log-space

    quats = torch.randn(num_points, 4, device=device)
    quats = quats / quats.norm(dim=-1, keepdim=True)

    init_opacity = 0.1
    opacities = torch.full((num_points,), math.log(init_opacity / (1 - init_opacity)), device=device)  # logit-space

    albedo_logits = torch.randn(num_points, 3, device=device) * 0.1
    roughness_logits = torch.zeros(num_points, 1, device=device)

    light_log_intensity = torch.tensor(math.log(4.6 ** 2), device=device)

    params = {
        "means": means,
        "scales": scales,
        "quats": quats,
        "opacities": opacities,
        "albedo_logits": albedo_logits,
        "roughness_logits": roughness_logits,
        "light_log_intensity": light_log_intensity,
    }
    for v in params.values():
        v.requires_grad_(True)
    return params

# ---------------------------------------------------------------------------
# Computation of Normals
# ---------------------------------------------------------------------------

def quat_to_matrix(quats):
    quats = quats / quats.norm(dim=-1, keepdim=True)
    w, x, y, z = quats.unbind(-1)
    res = torch.stack([
        torch.stack([1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],   dim=-1),
        torch.stack([2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],   dim=-1),
        torch.stack([2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)], dim=-1),
    ], dim=-2)
    return res

def compute_normals(means, log_scales, quats, camera_pos):
    R = quat_to_matrix(quats)
    flat_axis_idx = torch.argmin(log_scales, dim=-1)
    normals = torch.gather(R, 2, flat_axis_idx.view(-1,1,1).expand(-1,3,1)).squeeze(-1)
    mean_to_camera = camera_pos - means
    flip = torch.sign((normals*mean_to_camera).sum(-1, keepdim=True))
    flip = torch.where(flip == 0, torch.ones_like(flip), flip)
    return normals*flip

def smoothness_regularizer(means, normals, albedo, sample_size=2048, k=4):
    n = means.shape[0]
    if n <= sample_size:
        idx = torch.arange(n, device=means.device)
    else:
        idx = torch.randperm(n, device=means.device)[:sample_size]

    dists = torch.cdist(means[idx], means)  # (S,N)
    dists[torch.arange(len(idx), device=means.device), idx] = float("inf")  # exclude self
    knn_idx = dists.topk(k, largest=False).indices  # (S,k)

    sample_normals = normals[idx].unsqueeze(1)  # (S,1,3)
    neighbor_normals = normals[knn_idx]  # (S,k,3)
    normal_loss = (1.0 - (sample_normals * neighbor_normals).sum(-1)).mean()

    sample_albedo = albedo[idx].unsqueeze(1)  # (S,1,3)
    neighbor_albedo = albedo[knn_idx]  # (S,k,3)
    albedo_loss = (sample_albedo - neighbor_albedo).abs().sum(-1).mean()

    return normal_loss, albedo_loss


def make_optimizers(params):
    # Per-parameter learning rates following the defaults from the
    # original 3DGS paper by Kerbl et al. DefaultStrategy requires one
    # dedicated optimizer per parameter (each with exactly one param_group),
    # not a single combined optimizer with multiple param groups.
    lrs = {
        "means": 1.6e-4,
        "scales": 5e-3,
        "quats": 1e-3,
        "opacities": 5e-2,
        "albedo_logits": 2.5e-3,
        "roughness_logits": 2.5e-3,
        "light_log_intensity": 1e-4,
    }
    return {name: torch.optim.Adam([params[name]], lr=lrs[name]) for name in params}


def render(params, view, lightcoords, device):
    means = params["means"]
    scales = torch.exp(params["scales"])
    quats = params["quats"]
    opacities = torch.sigmoid(params["opacities"])
    camera_pos = torch.inverse(view["viewmat"])[:3, 3].to(device)
    normals = compute_normals(means, params["scales"], quats, camera_pos)

    light_pos = torch.tensor(lightcoords, dtype=torch.float32, device=device)
    means_to_light = light_pos - means  # (N,3), raw direction -- keep before normalizing, need its length for falloff
    dist_sq = (means_to_light ** 2).sum(-1, keepdim=True)  # (N,1) per-Gaussian squared distance to light
    light_dir = means_to_light / torch.sqrt(dist_sq)
    n_dot_l = F.relu((normals * light_dir).sum(-1, keepdim=True))
    intensity = torch.exp(params["light_log_intensity"])
    colors = torch.sigmoid(params["albedo_logits"]) * n_dot_l * intensity / dist_sq

    # rasterization() supports batched cameras via a leading dim; we only
    # have one camera per call, so add a size-1 batch dim and squeeze after.
    render_colors, render_alphas, info = rasterization(
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
        packed=False,
    )
    return render_colors[0], info

def render_normals(params, view, device):
    """Runs one forward pass through gsplat's rasterizer for a single view."""
    means = params["means"]
    scales = torch.exp(params["scales"])
    quats = params["quats"]
    opacities = torch.sigmoid(params["opacities"])
    camera_pos = torch.inverse(view["viewmat"])[:3, 3].to(device)
    normals = compute_normals(means, params["scales"], quats, camera_pos)
    normal_colors = (normals+1)/2

    # rasterization() supports batched cameras via a leading dim; we only
    # have one camera per call, so add a size-1 batch dim and squeeze after
    render_colors, render_alphas, _meta = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=normal_colors,
        viewmats=view["viewmat"].unsqueeze(0).to(device),
        Ks=view["Ks"].unsqueeze(0).to(device),
        width=view["width"],
        height=view["height"],
        sh_degree=None,  # no SH evaluation -- colors are used directly as final RGB
    )
    return render_colors[0]

def render_albedo(params, view, device):
    """Runs one forward pass through gsplat's rasterizer for a single view."""
    means = params["means"]
    scales = torch.exp(params["scales"])
    quats = params["quats"]
    opacities = torch.sigmoid(params["opacities"])
    albedo = torch.sigmoid(params["albedo_logits"])

    # rasterization() supports batched cameras via a leading dim; we only
    # have one camera per call, so add a size-1 batch dim and squeeze after
    render_colors, render_alphas, _meta = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=albedo,
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
def save_checkpoint(ckpt_dir, step, params, optimizers, strategy_state):
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"step_{step:06d}.pt")
    torch.save({
        "step": step,
        "params": {k: v.detach().cpu() for k, v in params.items()},
        "optimizers": {name: opt.state_dict() for name, opt in optimizers.items()},
        "strategy_state": {
            k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in strategy_state.items()
        },
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
    ckpt["strategy_state"] = {
        k: (v.to(device) if torch.is_tensor(v) else v) for k, v in ckpt["strategy_state"].items()
    }
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
            pred, _ = render(params, view, view["lightcoords"], device)
            target = view["image"].to(device)
            total_psnr += psnr(pred, target).item()
        return total_psnr / len(held_out_views)


def angular_error_deg(pred_normals, gt_normals):
    cos_sim = (pred_normals * gt_normals).sum(-1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cos_sim))


def evaluate_aovs(params, views_with_exr, device):
    with torch.no_grad():
        total_albedo_l1 = 0.0
        total_normal_error = 0.0
        for view in views_with_exr:
            pred_albedo = render_albedo(params, view, device)
            pred_normal = render_normals(params, view, device) * 2 - 1  # undo the (n+1)/2 mapping applied in render_normals
            pred_normal = F.normalize(pred_normal, p=2, dim=-1)

            gt_albedo = view["gt_albedo"].to(device)
            gt_normal = view["gt_normal"].to(device)
            mask = gt_albedo.sum(-1) > 1e-4

            total_albedo_l1 += (pred_albedo - gt_albedo).abs().sum(-1)[mask].mean().item()
            total_normal_error += angular_error_deg(pred_normal, gt_normal)[mask].mean().item()
        n = len(views_with_exr)
        return total_albedo_l1 / n, total_normal_error / n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1).to(device)
    print(f"[setup] device={device}, iters={args.iters}")

    all_views = load_split(args.data_dir, args.downsample)
    random.shuffle(all_views)
    held_out_views = all_views[-args.held_out:]
    train_views = all_views[:-args.held_out]
    # light5 views: light direction different from any of the training images
    test_light_views = load_split(args.data_dir, args.downsample, test_light=True, load_exr=True)
    print(f"[data] {len(train_views)} train views, {len(held_out_views)} held-out training views, "
          f"{len(test_light_views)} test (light5) views")

    # refine_stop_iter defaults to 15000 in gsplat, but we want to set it to args.iters so
    # densification (duplicate/split/prune) keeps running for the whole run (potentially beyond 15000)
    strategy = DefaultStrategy(verbose=False, refine_stop_iter=args.iters)

    ckpt = load_latest_checkpoint(args.ckpt_dir, device) if args.resume else None
    if ckpt is not None:
        params = {k: v.to(device) for k, v in ckpt["params"].items()}
        start_step = ckpt["step"] + 1
        optimizers = make_optimizers(params)
        for name, opt in optimizers.items():
            opt.load_state_dict(ckpt["optimizers"][name])
        strategy_state = ckpt["strategy_state"]
    else:
        params = init_gaussians(args.num_init_points, args.scene_extent, device)
        start_step = 0
        optimizers = make_optimizers(params)
        strategy_state = strategy.initialize_state(scene_scale=args.scene_extent)

    gs_optimizers = {k: optimizers[k] for k in GS_KEYS}
    strategy.check_sanity({k: params[k] for k in GS_KEYS}, gs_optimizers)

    for step in range(start_step, args.iters):
        view = train_views[random.randrange(len(train_views))]

        for opt in optimizers.values():
            opt.zero_grad()

        pred, info = render(params, view, view["lightcoords"], device)
        target = view["image"].to(device)

        gs_params = {k: params[k] for k in GS_KEYS}
        strategy.step_pre_backward(gs_params, gs_optimizers, strategy_state, step, info)

        pred_reshaped = pred.permute(2, 0, 1).unsqueeze(0)
        target_reshaped = target.permute(2, 0, 1).unsqueeze(0)
        loss = 0.8 * torch.abs(pred - target).mean() + 0.2 * (1 - ssim(pred_reshaped, target_reshaped))

        camera_pos = torch.inverse(view["viewmat"])[:3, 3].to(device)
        normals = compute_normals(params["means"], params["scales"], params["quats"], camera_pos)
        albedo = torch.sigmoid(params["albedo_logits"])
        normal_smooth_loss, albedo_smooth_loss = smoothness_regularizer(params["means"], normals, albedo)
        loss = loss + args.lambda_normal_smooth * normal_smooth_loss + args.lambda_albedo_smooth * albedo_smooth_loss

        loss.backward()

        strategy.step_post_backward(gs_params, gs_optimizers, strategy_state, step, info)
        params.update(gs_params)  # gs_params entries may have been reassigned to new (split/cloned/pruned) tensors

        for opt in optimizers.values():
            opt.step()

        if step > 0 and step % args.eval_every == 0:
            mean_psnr = evaluate(params, held_out_views, device)
            light5_psnr = evaluate(params, test_light_views, device)
            albedo_l1, normal_error_deg = evaluate_aovs(params, test_light_views, device)
            print(f"[eval] step {step}  held-out training PSNR={mean_psnr:.2f} dB  "
                  f"test (light5) PSNR={light5_psnr:.2f} dB  "
                  f"albedo L1={albedo_l1:.4f}  normal error={normal_error_deg:.2f} deg  "
                  f"num_gaussians={params['means'].shape[0]}")

        if step > 0 and step % args.ckpt_every == 0:
            save_checkpoint(args.ckpt_dir, step, params, optimizers, strategy_state)

    save_checkpoint(args.ckpt_dir, args.iters, params, optimizers, strategy_state)
    final_psnr = evaluate(params, held_out_views, device)
    final_light5_psnr = evaluate(params, test_light_views, device)
    final_albedo_l1, final_normal_error_deg = evaluate_aovs(params, test_light_views, device)
    print(f"[done] final held-out training PSNR={final_psnr:.2f} dB  "
          f"final test (light5) PSNR={final_light5_psnr:.2f} dB  "
          f"final albedo L1={final_albedo_l1:.4f}  final normal error={final_normal_error_deg:.2f} deg  "
          f"num_gaussians={params['means'].shape[0]}")


if __name__ == "__main__":
    main()
