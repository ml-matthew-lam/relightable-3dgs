"""Renders a checkpoint into an animated WEBP: a tour around a set of camera
positions on a sphere, with a smooth orbit transition between each pair, and
at each position a held camera with the light swinging back and forth on its
own sphere.

Example:
    python render_video.py --ckpt_dir checkpoints/lambertian_week3 \\
        --num_camera_positions 6 --camera_radius 4.5 --out tour.webp
"""
import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from train import render


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=None,
                    help="Explicit checkpoint path. If omitted, uses the latest step_*.pt in --ckpt_dir.")
    p.add_argument("--ckpt_dir", type=str, default="checkpoints")
    p.add_argument("--out", type=str, default="relight_video.webp")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # camera intrinsics
    p.add_argument("--data_dir", type=str, default="checkered_suzanne",
                    help="Used only to read camera_angle_x (FOV) from train/light1/transforms.json, "
                         "unless --fov_deg is given explicitly.")
    p.add_argument("--fov_deg", type=float, default=None)
    p.add_argument("--width", type=int, default=800)
    p.add_argument("--height", type=int, default=800)

    # camera positions: either explicit coordinates, or auto-placed on a circle
    p.add_argument("--camera_positions", type=str, default=None,
                    help='JSON list of explicit [x,y,z] world-space camera positions, '
                         'e.g. \'[[4,0,1],[0,4,1],[-4,0,1]]\'. Overrides --num_camera_positions/--camera_radius.')
    p.add_argument("--num_camera_positions", type=int, default=6)
    p.add_argument("--camera_radius", type=float, default=4.5)
    p.add_argument("--camera_elevation_deg", type=float, default=20.0,
                    help="Elevation used for auto-placed camera positions (ignored if --camera_positions is given).")

    # light motion: azimuth oscillates sinusoidally at fixed elevation/radius,
    # independently at each camera position
    p.add_argument("--light_radius", type=float, default=4.5)
    p.add_argument("--light_elevation_deg", type=float, default=30.0)
    p.add_argument("--light_azimuth_center_deg", type=float, default=0.0)
    p.add_argument("--light_azimuth_amplitude_deg", type=float, default=60.0)
    p.add_argument("--light_cycles", type=float, default=2.0,
                    help="Number of back-and-forth swings per camera position.")

    # frame counts
    p.add_argument("--transition_frames", type=int, default=60)
    p.add_argument("--dwell_frames", type=int, default=90)
    p.add_argument("--fps", type=int, default=24)
    return p.parse_args()


def find_latest_checkpoint(ckpt_dir):
    if not os.path.isdir(ckpt_dir):
        return None
    ckpts = sorted(f for f in os.listdir(ckpt_dir) if f.startswith("step_") and f.endswith(".pt"))
    if not ckpts:
        return None
    return os.path.join(ckpt_dir, ckpts[-1])


def spherical_to_cartesian(radius, azimuth_deg, elevation_deg):
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    x = radius * math.cos(el) * math.cos(az)
    y = radius * math.cos(el) * math.sin(az)
    z = radius * math.sin(el)
    return torch.tensor([x, y, z], dtype=torch.float32)


def slerp(p0, p1, t):
    """Spherical linear interpolation between two 3D points (not necessarily
    the same radius -- radius is interpolated linearly, direction via slerp)."""
    r0, r1 = p0.norm(), p1.norm()
    r = r0 + (r1 - r0) * t
    p0n, p1n = F.normalize(p0, dim=0), F.normalize(p1, dim=0)
    dot = torch.clamp((p0n * p1n).sum(), -1.0, 1.0)
    omega = torch.acos(dot)
    if omega.abs() < 1e-6:
        direction = p0n  # near-identical directions -- slerp is ill-conditioned, direction barely changes anyway
    else:
        sin_omega = torch.sin(omega)
        a = torch.sin((1 - t) * omega) / sin_omega
        b = torch.sin(t * omega) / sin_omega
        direction = a * p0n + b * p1n
    return direction * r


def look_at_c2w(eye, target=None, up=None):
    """Camera-to-world in Blender/OpenGL convention (camera looks down -z, y-up) --
    same convention train.py's load_split() expects before its flip+invert step.
    Degenerates when eye-target direction is parallel to up (straight overhead)."""
    target = torch.zeros(3) if target is None else target
    up = torch.tensor([0.0, 0.0, 1.0]) if up is None else up
    f = F.normalize(target - eye, dim=0)
    right = F.normalize(torch.cross(f, up, dim=0), dim=0)
    true_up = torch.cross(right, f, dim=0)
    c2w = torch.eye(4, dtype=torch.float32)
    c2w[:3, 0] = right
    c2w[:3, 1] = true_up
    c2w[:3, 2] = -f
    c2w[:3, 3] = eye
    return c2w


def build_view(cam_pos, Ks, width, height):
    c2w_gl = look_at_c2w(cam_pos)
    flip = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0]))
    c2w_cv = c2w_gl @ flip
    viewmat = torch.inverse(c2w_cv)
    return {"viewmat": viewmat, "Ks": Ks, "width": width, "height": height}


def to_pil(image_tensor):
    arr = image_tensor.clamp(0, 1).detach().cpu().numpy()
    return Image.fromarray((arr * 255).astype(np.uint8))


def main():
    args = parse_args()
    device = torch.device(args.device)

    ckpt_path = args.ckpt or find_latest_checkpoint(args.ckpt_dir)
    if ckpt_path is None:
        raise FileNotFoundError(f"no checkpoint found (looked in {args.ckpt_dir})")
    print(f"[load] checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    params = {k: v.to(device) for k, v in ckpt["params"].items()}

    if args.fov_deg is not None:
        fov_rad = math.radians(args.fov_deg)
    else:
        with open(os.path.join(args.data_dir, "train", "light1", "transforms.json")) as f:
            fov_rad = json.load(f)["camera_angle_x"]
    focal = 0.5 * args.width / math.tan(0.5 * fov_rad)
    Ks = torch.tensor([
        [focal, 0.0, args.width / 2.0],
        [0.0, focal, args.height / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32)

    if args.camera_positions:
        cam_positions = [torch.tensor(p, dtype=torch.float32) for p in json.loads(args.camera_positions)]
    else:
        cam_positions = [
            spherical_to_cartesian(args.camera_radius, 360.0 * i / args.num_camera_positions, args.camera_elevation_deg)
            for i in range(args.num_camera_positions)
        ]

    def light_position(t):
        az = args.light_azimuth_center_deg + args.light_azimuth_amplitude_deg * math.sin(
            2 * math.pi * args.light_cycles * t
        )
        return spherical_to_cartesian(args.light_radius, az, args.light_elevation_deg)

    def render_frame(cam_pos, light_pos):
        view = build_view(cam_pos, Ks, args.width, args.height)
        with torch.no_grad():
            image, _info = render(params, view, light_pos.tolist(), device)
        return to_pil(image)

    frames = []
    resting_light = light_position(0.0)
    for i, cam_pos in enumerate(cam_positions):
        if i > 0:
            print(f"[video] transition {i}/{len(cam_positions) - 1}")
            for f_idx in range(args.transition_frames):
                t = f_idx / args.transition_frames
                interp_cam = slerp(cam_positions[i - 1], cam_pos, t)
                frames.append(render_frame(interp_cam, resting_light))

        print(f"[video] camera position {i + 1}/{len(cam_positions)}: light sweep")
        for f_idx in range(args.dwell_frames):
            t = f_idx / args.dwell_frames
            frames.append(render_frame(cam_pos, light_position(t)))

    duration_ms = int(1000 / args.fps)
    frames[0].save(args.out, format="WEBP", save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)
    print(f"[done] saved {len(frames)} frames ({len(frames) / args.fps:.1f}s) to {args.out}")


if __name__ == "__main__":
    main()
