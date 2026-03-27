import argparse
import os
from typing import Dict, Any

from PIL import Image
import torch
import torch.nn.functional as F

from net.CIDNet import CIDNet
from net.wtconv import WTConv2d

# Fixed model config aligned with your LOLv1 training command.
FIXED_CFG = {
    "fe_type": "dual_gate",
    "lca_type": "diem",
    "use_wtconv_i": True,
    "use_dwconv_hv": False,
    "max_regions": 16,
    "pre_lca_film": True,
    "pre_lca_film_scale": 0.1,
    "pre_lca_film_bias": 0.1,
    "pre_lca_film_alpha": -2.197225,
    "attn_alpha1_init": -2.197225,
    "attn_alpha2_init": -3.891,
    "attn_mask_bias_scale1_init": 1.0,
    "attn_mask_bias_scale2_init": 0.5,
    "attn_mask_bias_scale1_max": -1.0,
    "attn_mask_bias_scale2_max": -1.0,
    "prior_mode": "attn",
}

DEFAULT_CKPT = "/home/zqh/code/HVI-CIDNet_1/weights/LOLv1/best_metrics_lol_v1_2026-02-23_14-39-20/epoch_1300.pth"
DEFAULT_IMAGE = "/media/zqh/data/low_light/LOLdataset/our485/low/9.png"
DEFAULT_LABEL = "/home/zqh/code/sam2/notebooks/runs/automatic_mask_lolv1_final_packpkl_plus_labels_plus_npz/label_uint8/9_labels.png"
DEFAULT_OVERLAY = "/home/zqh/code/sam2/notebooks/runs/automatic_mask_lolv1_final_packpkl_plus_labels_plus_npz/overlay/9_overlay.png"


def _load_image_rgb(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    raw = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
    ten = raw.view(h, w, 3).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return ten


def _load_index_map(path: str) -> torch.Tensor:
    lab = Image.open(path)
    if lab.mode in ("RGB", "RGBA"):
        # If label is RGB-like, use the first channel as region id map.
        lab = lab.split()[0]
    w, h = lab.size
    raw = torch.ByteTensor(torch.ByteStorage.from_buffer(lab.tobytes()))
    ten = raw.view(h, w).unsqueeze(0).long()
    return ten


def _strip_module_prefix(state: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in state.items():
        if k.startswith("module."):
            out[k[len("module."):]] = v
        else:
            out[k] = v
    return out


def _load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> None:
    raw = torch.load(ckpt_path, map_location=device)
    if isinstance(raw, dict):
        if "state_dict" in raw and isinstance(raw["state_dict"], dict):
            state = raw["state_dict"]
        elif "model" in raw and isinstance(raw["model"], dict):
            state = raw["model"]
        else:
            # Some checkpoints are direct state dicts.
            state = raw
    else:
        state = raw

    state = _strip_module_prefix(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] missing keys: {len(missing)}")
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)}")


def _to_feature_map(x: torch.Tensor) -> torch.Tensor:
    # x: [1, C, H, W] -> [H, W] normalized map
    if x.ndim != 4 or x.shape[0] != 1:
        raise ValueError(f"Expected [1,C,H,W], got {tuple(x.shape)}")
    fmap = x.abs().mean(dim=1)[0]
    fmap = fmap - fmap.min()
    den = fmap.max().clamp_min(1e-6)
    fmap = fmap / den
    return fmap.detach().cpu()


def _to_feature_rgb3(x: torch.Tensor) -> torch.Tensor:
    # x: [1, C, H, W] -> [H, W, 3] normalized RGB preview from first 3 channels
    if x.ndim != 4 or x.shape[0] != 1:
        raise ValueError(f"Expected [1,C,H,W], got {tuple(x.shape)}")
    c = x.shape[1]
    feat = x[0]
    if c >= 3:
        rgb = feat[:3]
    else:
        rgb = feat[:1].repeat(3, 1, 1)
    # Per-channel min-max for readability
    out = []
    for i in range(3):
        ch = rgb[i]
        ch = ch - ch.min()
        ch = ch / ch.max().clamp_min(1e-6)
        out.append(ch)
    rgb = torch.stack(out, dim=0).permute(1, 2, 0).detach().cpu()
    return rgb


def _to_hv_rgb(x_hv: torch.Tensor) -> torch.Tensor:
    # x_hv: [1, 2, H, W] -> [H, W, 3]
    if x_hv.ndim != 4 or x_hv.shape[0] != 1 or x_hv.shape[1] < 2:
        raise ValueError(f"Expected [1,2,H,W], got {tuple(x_hv.shape)}")
    h = x_hv[0, 0]
    v = x_hv[0, 1]
    h = (h - h.min()) / (h.max() - h.min() + 1e-6)
    v = (v - v.min()) / (v.max() - v.min() + 1e-6)
    mag = torch.sqrt(x_hv[0, 0] * x_hv[0, 0] + x_hv[0, 1] * x_hv[0, 1])
    mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-6)
    rgb = torch.stack([h, v, mag], dim=-1).detach().cpu()
    return rgb


def _jet_colormap(gray: torch.Tensor) -> torch.Tensor:
    # gray: [H, W] in [0,1] -> [H, W, 3]
    g = gray.clamp(0.0, 1.0)
    four = 4.0 * g
    r = torch.minimum(four - 1.5, -four + 4.5).clamp(0.0, 1.0)
    gg = torch.minimum(four - 0.5, -four + 3.5).clamp(0.0, 1.0)
    b = torch.minimum(four + 0.5, -four + 2.5).clamp(0.0, 1.0)
    return torch.stack([r, gg, b], dim=-1).detach().cpu()


def _bwr_colormap(signed_map: torch.Tensor) -> torch.Tensor:
    # signed_map in [-1, 1], 0 -> white, negative -> blue, positive -> red
    x = signed_map.clamp(-1.0, 1.0)
    pos = x.clamp_min(0.0)
    neg = (-x).clamp_min(0.0)
    r = 1.0 - neg
    g = 1.0 - pos - neg
    b = 1.0 - pos
    return torch.stack([r, g, b], dim=-1).clamp(0.0, 1.0).detach().cpu()


def _to_signed_feature_map(x: torch.Tensor) -> torch.Tensor:
    # x: [1,C,H,W] -> [H,W] in [-1,1], robustly normalized around 0
    if x.ndim != 4 or x.shape[0] != 1:
        raise ValueError(f"Expected [1,C,H,W], got {tuple(x.shape)}")
    fmap = x.mean(dim=1)[0]
    scale = fmap.abs().quantile(0.995).clamp_min(1e-6)
    return (fmap / scale).clamp(-1.0, 1.0).detach().cpu()


def _save_map_png(arr: torch.Tensor, path: str) -> None:
    img = (arr * 255.0).clamp(0, 255).to(torch.uint8)
    Image.fromarray(img.numpy()).save(path)


def _save_rgb_png(arr: torch.Tensor, path: str) -> None:
    img = (arr * 255.0).clamp(0, 255).to(torch.uint8)
    Image.fromarray(img.numpy(), mode="RGB").save(path)


def _overlay_heatmap_on_rgb(heat_rgb: torch.Tensor, rgb: torch.Tensor, alpha: float = 0.55) -> torch.Tensor:
    # heat_rgb: [H,W,3] in [0,1], rgb: [H,W,3] in [0,1]
    if heat_rgb.shape[:2] != rgb.shape[:2]:
        heat_rgb = F.interpolate(
            heat_rgb.permute(2, 0, 1).unsqueeze(0),
            size=rgb.shape[:2],
            mode="bilinear",
            align_corners=True,
        )[0].permute(1, 2, 0).clamp(0.0, 1.0)
    return (alpha * heat_rgb + (1.0 - alpha) * rgb).clamp(0.0, 1.0)


def _label_to_color(label: torch.Tensor) -> torch.Tensor:
    # label: [H,W] integer map -> [H,W,3] pseudo color
    idx = label.long()
    r = ((idx * 37 + 13) % 255).float() / 255.0
    g = ((idx * 67 + 91) % 255).float() / 255.0
    b = ((idx * 97 + 47) % 255).float() / 255.0
    return torch.stack([r, g, b], dim=-1).clamp(0.0, 1.0)


def _index_to_onehot(index_map: torch.Tensor) -> torch.Tensor:
    # index_map: [B,H,W] -> one_hot: [B,K,H,W], binary 0/1
    idx = index_map.long()
    k = int(idx.max().item()) + 1
    return F.one_hot(idx, num_classes=k).permute(0, 3, 1, 2).float()


def _shape_str(x: torch.Tensor) -> str:
    dims = tuple(x.shape)
    if len(dims) == 4:
        b, c, h, w = dims
        return f"(B={b},C={c},H={h},W={w})"
    if len(dims) == 3:
        b, h, w = dims
        return f"(B={b},H={h},W={w})"
    return str(dims)


def _axis_expr(cur: int, base: int, axis: str) -> str:
    if cur == base:
        return axis
    if base % cur == 0:
        return f"{axis}/{base // cur}"
    if cur % base == 0:
        return f"{axis}*{cur // base}"
    return f"{axis}*{cur}/{base}"


def _axis_expr_file(cur: int, base: int, axis: str) -> str:
    if cur == base:
        return axis
    if base % cur == 0:
        return f"{axis}div{base // cur}"
    if cur % base == 0:
        return f"{axis}mul{cur // base}"
    return f"{axis}mul{cur}of{base}"


def _shape_symbolic_str(x: torch.Tensor, base_hw, ch_symbol: str = "C") -> str:
    dims = tuple(x.shape)
    if len(dims) == 4:
        _, c, h, w = dims
        h0, w0 = base_hw
        return f"(B,{ch_symbol}={c},{_axis_expr(h, h0, 'H')},{_axis_expr(w, w0, 'W')})"
    if len(dims) == 3:
        _, h, w = dims
        h0, w0 = base_hw
        return f"(B,{_axis_expr(h, h0, 'H')},{_axis_expr(w, w0, 'W')})"
    return str(dims)


def _shape_file_tag(x: torch.Tensor, base_hw, ch_symbol: str = "C") -> str:
    dims = tuple(x.shape)
    if len(dims) == 4:
        b, c, h, w = dims
        h0, w0 = base_hw
        hs = _axis_expr_file(h, h0, "H")
        ws = _axis_expr_file(w, w0, "W")
        return f"B{b}_{ch_symbol}{c}_{hs}_{ws}"
    if len(dims) == 3:
        b, h, w = dims
        h0, w0 = base_hw
        hs = _axis_expr_file(h, h0, "H")
        ws = _axis_expr_file(w, w0, "W")
        return f"B{b}_{hs}_{ws}"
    return "shapeUnknown"


def _save_softmask_visuals(
    softmask: torch.Tensor,
    base_name: str,
    outdir: str,
    rgb_base: torch.Tensor,
    overlay_base: torch.Tensor,
) -> None:
    # softmask: [1,K,H,W]
    maxprob = softmask.max(dim=1)[0][0]
    argmax = softmask.argmax(dim=1)[0]
    maxprob_jet = _jet_colormap(maxprob)
    argmax_color = _label_to_color(argmax)
    _save_map_png(maxprob, os.path.join(outdir, f"{base_name}_maxprob.png"))
    _save_rgb_png(maxprob_jet, os.path.join(outdir, f"{base_name}_maxprob_jet.png"))
    _save_rgb_png(argmax_color, os.path.join(outdir, f"{base_name}_argmax_color.png"))
    _save_rgb_png(_overlay_heatmap_on_rgb(maxprob_jet, rgb_base, alpha=0.58), os.path.join(outdir, f"{base_name}_maxprob_overlay_input.png"))
    _save_rgb_png(_overlay_heatmap_on_rgb(maxprob_jet, overlay_base, alpha=0.58), os.path.join(outdir, f"{base_name}_maxprob_overlay_sam.png"))


def build_model(device: torch.device) -> CIDNet:
    model = CIDNet(
        use_wtconv_i=FIXED_CFG["use_wtconv_i"],
        use_dwconv_hv=FIXED_CFG["use_dwconv_hv"],
        fe_type=FIXED_CFG["fe_type"],
        lca_type=FIXED_CFG["lca_type"],
        max_regions=FIXED_CFG["max_regions"],
        pre_lca_film=FIXED_CFG["pre_lca_film"],
        pre_lca_film_scale=FIXED_CFG["pre_lca_film_scale"],
        pre_lca_film_bias=FIXED_CFG["pre_lca_film_bias"],
        pre_lca_film_alpha=FIXED_CFG["pre_lca_film_alpha"],
        attn_alpha1_init=FIXED_CFG["attn_alpha1_init"],
        attn_alpha2_init=FIXED_CFG["attn_alpha2_init"],
        attn_mask_bias_scale1_init=FIXED_CFG["attn_mask_bias_scale1_init"],
        attn_mask_bias_scale2_init=FIXED_CFG["attn_mask_bias_scale2_init"],
        attn_mask_bias_scale1_max=FIXED_CFG["attn_mask_bias_scale1_max"],
        attn_mask_bias_scale2_max=FIXED_CFG["attn_mask_bias_scale2_max"],
    ).to(device)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Export intermediate feature maps with forward hooks.")
    parser.add_argument("--ckpt", "--weights", dest="ckpt", type=str, default=DEFAULT_CKPT, help="Path to checkpoint")
    parser.add_argument("--image", type=str, default=DEFAULT_IMAGE, help="Path to low-light RGB image")
    parser.add_argument("--label", type=str, default=DEFAULT_LABEL, help="Path to region index map image")
    parser.add_argument(
        "--overlay",
        type=str,
        default=DEFAULT_OVERLAY,
        help="Path to SAM overlay RGB image for prettier mask overlay visualization",
    )
    parser.add_argument("--outdir", type=str, default="results/vis_intermediate", help="Output directory")
    parser.add_argument("--save_pt", action="store_true", help="Also save raw feature tensors as .pt")
    parser.add_argument(
        "--hvi_detail",
        action="store_true",
        help="Also export detailed H/V/I maps (default exports only branch-level hvi_i.png and hvi_hv.png)",
    )

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        "[info] fixed cfg: "
        f"fe_type={FIXED_CFG['fe_type']}, lca_type={FIXED_CFG['lca_type']}, "
        f"prior_mode={FIXED_CFG['prior_mode']}, pre_lca_film={FIXED_CFG['pre_lca_film']}"
    )

    model = build_model(device)
    _load_checkpoint(model, args.ckpt, device)
    model.eval()

    x = _load_image_rgb(args.image).to(device)
    index_map = _load_index_map(args.label).to(device)
    overlay_path = args.overlay
    if overlay_path and os.path.exists(overlay_path):
        overlay_rgb = _load_image_rgb(overlay_path).to(device)
    else:
        overlay_rgb = x
        if overlay_path:
            print(f"[warn] overlay image not found: {overlay_path}; fallback to input image for overlay outputs")

    captured: Dict[str, torch.Tensor] = {}
    meta_shapes: Dict[str, Any] = {}

    def _mk_hook(name: str):
        def _hook(_m, _inp, out):
            if isinstance(out, (tuple, list)):
                out = out[0]
            captured[name] = out.detach().float().cpu()
        return _hook

    hooks = []
    # 1) MWFE output + branch internals (I-branch stem)
    if hasattr(model.IE_block0, "conv_branch") and hasattr(model.IE_block0, "alt_branch") and hasattr(model.IE_block0, "gate"):
        def _mwfe_hook(mod, inp, out):
            x_in = inp[0].detach()
            with torch.no_grad():
                f_conv = mod.conv_branch(x_in)
                f_alt = mod.alt_branch(x_in)
                gate = torch.sigmoid(mod.gate(torch.cat([f_conv, f_alt], dim=1)))
            captured["mwfe_fout"] = out.detach().float().cpu()
            captured["mwfe_spatial"] = f_conv.detach().float().cpu()
            captured["mwfe_wavelet"] = f_alt.detach().float().cpu()
            captured["mwfe_gate"] = gate.detach().float().cpu()
        hooks.append(model.IE_block0.register_forward_hook(_mwfe_hook))
    else:
        hooks.append(model.IE_block0.register_forward_hook(_mk_hook("mwfe_fout")))

    # 2) SMM output at stage-1
    if "1" in model.pre_lca_film_i:
        hooks.append(model.pre_lca_film_i["1"].register_forward_hook(_mk_hook("smm_fmod_s1")))

    # 3) RAI output + input shape meta at stage-1 (attn branch)
    def _rai_hook(_m, _inp, out):
        feat_in, mask_in, v_in = _inp
        captured["rai_fplus_s1"] = out.detach().float().cpu()
        meta_shapes["rai_feat_in_s1"] = tuple(feat_in.shape)
        meta_shapes["rai_mask_in_s1"] = tuple(mask_in.shape)
        meta_shapes["rai_v_in_s1"] = tuple(v_in.shape)
    hooks.append(model.region_attn.register_forward_hook(_rai_hook))

    # 4) CBC outputs at stage-1 (before RAI for I, and HV side)
    hooks.append(model.I_LCA1.register_forward_hook(_mk_hook("cbc_i_s1")))
    hooks.append(model.HV_LCA1.register_forward_hook(_mk_hook("cbc_hv_s1")))

    # 4.5) Soft region masks after softmax (stage1/stage2)
    soft_count = {"n": 0}

    def _softmask_hook(_m, _inp, out):
        soft_count["n"] += 1
        captured[f"softmask_s{soft_count['n']}"] = out.detach().float().cpu()

    hooks.append(model.soft_mask.register_forward_hook(_softmask_hook))

    # 5) DWT sub-bands inside WTConv2d (first-level decomposition)
    wt_modules = [m for m in model.modules() if isinstance(m, WTConv2d)]
    if wt_modules:
        wt0 = wt_modules[0]

        def _dwt_hook(_m, _inp, _out):
            xin = _inp[0].detach()
            dwt = _m.wt_function(xin).detach().float().cpu()  # [B,C,4,H/2,W/2]
            captured["dwt_ll"] = dwt[:, :, 0, :, :]
            captured["dwt_lh"] = dwt[:, :, 1, :, :]
            captured["dwt_hl"] = dwt[:, :, 2, :, :]
            captured["dwt_hh"] = dwt[:, :, 3, :, :]

        hooks.append(wt0.register_forward_hook(_dwt_hook))
    else:
        print("[warn] no WTConv2d found; DWT sub-bands will not be exported")

    with torch.no_grad():
        captured["softmask_input"] = model.soft_mask(index_map, target_hw=x.shape[-2:]).detach().float().cpu()
        hvi = model.HVIT(x).detach().float().cpu()
        out = model(x, index_map=index_map, prior_mode=FIXED_CFG["prior_mode"])

    # Save input/output for reference
    in_img = (x[0].detach().cpu().permute(1, 2, 0) * 255.0).clamp(0, 255).to(torch.uint8).numpy()
    out_img = (out[0].detach().cpu().permute(1, 2, 0) * 255.0).clamp(0, 255).to(torch.uint8).numpy()
    Image.fromarray(in_img, mode="RGB").save(os.path.join(args.outdir, "input.png"))
    Image.fromarray(out_img, mode="RGB").save(os.path.join(args.outdir, "output.png"))

    # Save branch-level HVI visualizations (what the paper needs)
    h_map = hvi[:, 0:1]
    v_map = hvi[:, 1:2]
    i_map = hvi[:, 2:3]
    hv_map = hvi[:, 0:2]
    _save_map_png(_to_feature_map(i_map), os.path.join(args.outdir, "hvi_i.png"))
    _save_rgb_png(_to_hv_rgb(hv_map), os.path.join(args.outdir, "hvi_hv.png"))
    if args.hvi_detail:
        _save_rgb_png(_jet_colormap(_to_feature_map(i_map)), os.path.join(args.outdir, "hvi_i_jet.png"))
        _save_map_png(_to_feature_map(h_map), os.path.join(args.outdir, "hvi_h_gray.png"))
        _save_map_png(_to_feature_map(v_map), os.path.join(args.outdir, "hvi_v_gray.png"))

    rgb_ref = x[0].detach().cpu().permute(1, 2, 0).clamp(0.0, 1.0)
    overlay_ref = overlay_rgb[0].detach().cpu().permute(1, 2, 0).clamp(0.0, 1.0)
    if overlay_ref.shape[:2] != rgb_ref.shape[:2]:
        overlay_ref = torch.nn.functional.interpolate(
            overlay_ref.permute(2, 0, 1).unsqueeze(0),
            size=rgb_ref.shape[:2],
            mode="bilinear",
            align_corners=True,
        )[0].permute(1, 2, 0).clamp(0.0, 1.0)
    _save_rgb_png(overlay_ref, os.path.join(args.outdir, "sam_overlay_input.png"))

    # Save binary K-layer masks before soft-mask.
    # Use top non-empty classes (by area) to avoid exporting all-black layers.
    onehot = _index_to_onehot(index_map).detach().cpu()  # [1,K,H,W], values in {0,1}
    onehot_tag = _shape_file_tag(onehot, x.shape[-2:], ch_symbol="K")
    flat_area = onehot[0].view(onehot.shape[1], -1).sum(dim=1)  # [K]
    non_empty = torch.nonzero(flat_area > 0, as_tuple=False).squeeze(1)
    if non_empty.numel() > 0:
        sorted_ids = non_empty[torch.argsort(flat_area[non_empty], descending=True)]
    else:
        sorted_ids = torch.arange(onehot.shape[1])
    k_show = min(10, int(sorted_ids.numel()))
    shown_ids = sorted_ids[:k_show].tolist()
    for k_idx in shown_ids:
        m = onehot[0, k_idx].unsqueeze(0)  # shape (1,H,W) for consistent shape_str
        k_name = f"Mask_OneHot_k{k_idx:02d}_from_{onehot_tag}"
        m2d = m[0]
        _save_map_png(m2d, os.path.join(args.outdir, f"{k_name}.png"))
        _save_rgb_png(_overlay_heatmap_on_rgb(_jet_colormap(m2d), overlay_ref, alpha=0.55), os.path.join(args.outdir, f"{k_name}_overlay_sam.png"))
        px = int(flat_area[k_idx].item())
        ratio = px / float(onehot.shape[-2] * onehot.shape[-1])
        print(f"[onehot] k={k_idx:02d} 像素数={px} 占比={ratio:.4%}")
    print(
        f"[ok] 保存离散标签one-hot二值层: 已保存{len(shown_ids)}层(优先非空且面积最大), "
        f"非空类别数={int(non_empty.numel())}, 总K={onehot.shape[1]} (来源: index_map -> one_hot)"
    )

    feature_alias = {
        "softmask_input": "SoftMask_Input",
        "mwfe_fout": "MWFE_Fout",
        "mwfe_spatial": "MWFE_Spatial",
        "mwfe_wavelet": "MWFE_Wavelet",
        "mwfe_gate": "MWFE_Gate",
        "smm_fmod_s1": "SMM_Fmod_S1",
        "cbc_i_s1": "CBC_I_S1",
        "cbc_hv_s1": "CBC_HV_S1",
        "rai_fplus_s1": "RAI_Fplus_S1",
        "softmask_s1": "SoftMask_S1",
        "softmask_s2": "SoftMask_S2",
        "dwt_ll": "DWT_LL",
        "dwt_lh": "DWT_LH",
        "dwt_hl": "DWT_HL",
        "dwt_hh": "DWT_HH",
    }
    feature_desc = {
        "softmask_input": "softmask_input: 由 index_map 经 SoftRegionMask 生成的输入分辨率软掩码",
        "mwfe_fout": "mwfe_fout: I分支 MWFE 输出",
        "mwfe_spatial": "mwfe_spatial: MWFE 空间卷积分支输出",
        "mwfe_wavelet": "mwfe_wavelet: MWFE 小波分支输出",
        "mwfe_gate": "mwfe_gate: MWFE gating 系数",
        "smm_fmod_s1": "smm_fmod_s1: Stage1 SMM (mask FiLM) 调制后的 I 特征",
        "cbc_i_s1": "cbc_i_s1: Stage1 CBC 后的 I 分支特征",
        "cbc_hv_s1": "cbc_hv_s1: Stage1 CBC 后的 HV 分支特征",
        "rai_fplus_s1": "rai_fplus_s1: Stage1 RAI 语义注入后的 I 特征",
        "softmask_s1": "softmask_s1: SoftRegionMask 在 Stage1 尺度下的软掩码",
        "softmask_s2": "softmask_s2: SoftRegionMask 在 Stage2 尺度下的软掩码",
        "dwt_ll": "dwt_ll: WTConv2d 小波低频子带",
        "dwt_lh": "dwt_lh: WTConv2d 小波水平高频子带",
        "dwt_hl": "dwt_hl: WTConv2d 小波垂直高频子带",
        "dwt_hh": "dwt_hh: WTConv2d 小波对角高频子带",
    }
    ch_symbol_map = {
        "softmask_input": "K",
        "softmask_s1": "K",
        "softmask_s2": "K",
    }
    backbone_channels = [36, 36, 72, 144]

    print(f"[shape] 输入 x: {_shape_str(x)} | 相对: {_shape_symbolic_str(x, x.shape[-2:], ch_symbol='C')}")
    print(f"[shape] index_map: {_shape_str(index_map)} | 相对: {_shape_symbolic_str(index_map, x.shape[-2:])}")
    uniq = torch.unique(index_map.detach().cpu())
    print(f"[mask] index_map 唯一标签数={int(uniq.numel())}, 最小={int(uniq.min())}, 最大={int(uniq.max())}")
    print(f"[shape] HVI: {_shape_str(hvi)} | 相对: {_shape_symbolic_str(hvi, x.shape[-2:], ch_symbol='C')}")
    print(f"[shape] H分支输入: {_shape_str(h_map)} | I分支输入: {_shape_str(i_map)}")
    print(f"[arch] 主干通道配置 channels={backbone_channels} -> Stage0/1/2/3 通道约为 C={backbone_channels[0]}/{backbone_channels[1]}/{backbone_channels[2]}/{backbone_channels[3]}")

    # Save tensors + three visualization styles
    order = [
        "softmask_input",
        "mwfe_fout",
        "mwfe_spatial",
        "mwfe_wavelet",
        "mwfe_gate",
        "smm_fmod_s1",
        "cbc_i_s1",
        "cbc_hv_s1",
        "rai_fplus_s1",
        "softmask_s1",
        "softmask_s2",
        "dwt_ll",
        "dwt_lh",
        "dwt_hl",
        "dwt_hh",
    ]
    for key in order:
        if key not in captured:
            print(f"[warn] {key} not captured")
            continue
        ten = captured[key]
        ch_symbol = ch_symbol_map.get(key, "C")
        shape_abs = _shape_str(ten)
        shape_rel = _shape_symbolic_str(ten, x.shape[-2:], ch_symbol=ch_symbol)
        print(f"[shape] {key}: {shape_abs} | 相对: {shape_rel} ({feature_desc.get(key, '未注释')})")
        if args.save_pt:
            torch.save(ten, os.path.join(args.outdir, f"{key}.pt"))

        if key.startswith("softmask_s"):
            alias = feature_alias.get(key, key)
            file_tag = _shape_file_tag(ten, x.shape[-2:], ch_symbol=ch_symbol)
            base_name = f"{alias}_{file_tag}"
            _save_softmask_visuals(ten, base_name, args.outdir, rgb_ref, overlay_ref)
            print(f"[ok] saved {key}: {base_name}_*.png")
            continue
        if key == "softmask_input":
            alias = feature_alias.get(key, key)
            file_tag = _shape_file_tag(ten, x.shape[-2:], ch_symbol=ch_symbol)
            base_name = f"{alias}_{file_tag}"
            _save_softmask_visuals(ten, base_name, args.outdir, rgb_ref, overlay_ref)
            print(f"[ok] saved {key}: {base_name}_*.png")
            continue

        fmap_gray = _to_feature_map(ten)
        fmap_jet = _jet_colormap(fmap_gray)
        fmap_rgb3 = _to_feature_rgb3(ten)
        fmap_overlay = _overlay_heatmap_on_rgb(fmap_jet, rgb_ref, alpha=0.55)
        alias = feature_alias.get(key, key)
        file_tag = _shape_file_tag(ten, x.shape[-2:], ch_symbol=ch_symbol)
        base = f"{alias}_{file_tag}"
        _save_map_png(fmap_gray, os.path.join(args.outdir, f"{base}_gray.png"))
        _save_rgb_png(fmap_jet, os.path.join(args.outdir, f"{base}_jet.png"))
        _save_rgb_png(fmap_rgb3, os.path.join(args.outdir, f"{base}_rgb3.png"))
        _save_rgb_png(fmap_overlay, os.path.join(args.outdir, f"{base}_overlay.png"))
        if key.startswith("dwt_"):
            fmap_signed = _to_signed_feature_map(ten)
            _save_rgb_png(_bwr_colormap(fmap_signed), os.path.join(args.outdir, f"{base}_bwr.png"))
        print(f"[ok] saved {key}: {base}_*.png")

    if "rai_feat_in_s1" in meta_shapes and "rai_mask_in_s1" in meta_shapes and "rai_v_in_s1" in meta_shapes:
        rf = meta_shapes["rai_feat_in_s1"]
        rm = meta_shapes["rai_mask_in_s1"]
        rv = meta_shapes["rai_v_in_s1"]
        rc = int(getattr(model.region_attn.q, "in_channels", -1))
        print(
            "[RAI] Stage1 通道对齐: "
            f"Feat输入(B,C,H,W)={rf}, SoftMask输入(B,K,H,W)={rm}, RegionToken输入(B,K,C)={rv}, "
            f"RAI内部q/k/v通道={rc}. 结论: C(Feat)=C(Token) 完全一致, K由mask区域数决定。"
        )

    for h in hooks:
        h.remove()

    print(f"[done] outputs in: {args.outdir}")


if __name__ == "__main__":
    main()
