"""Run C-LiTE on a region of a whole-slide image and save one row per nucleus.

Pipeline:  WSI region  ->  nuclei (HoVer-Net)  ->  64x64 crop per nucleus  ->  C-LiTE CLS feature
           ->  (a) cell type from a linear probe   (b) label-free cluster from k-means

Outputs (in --out):
  region.jpg        the H&E region that was analysed (the viewer's base layer)
  cells.csv         cell_id, x, y, area, pred, prob, cluster        (x, y in region pixels)
  contours.joblib   list of (K,2) int arrays, one nucleus outline per row of cells.csv
  features.npz      feats (N,384 float16), cell_id
  meta.json         slide path, bounds, mpp, class names, cluster count

Examples
  # whole pipeline from a slide (needs tiatoolbox for reading the slide and HoVer-Net)
  python scripts/run_wsi.py --wsi TCGA-XX.svs --bounds 20000 15000 28000 21000 \
      --checkpoint c-lite.pth --probe pannuke_probe.joblib --out work/TCGA-XX

  # a small exported region (e.g. a single-level .svs crop at ~0.5 µm/px)
  python scripts/run_wsi.py --image crop.svs --image_mpp 0.5 \
      --checkpoint c-lite.pth --probe pannuke_probe.joblib --out work/crop

  # you already have a region image and HoVer-Net output for it
  python scripts/run_wsi.py --image region.png --nuclei region_hovernet.json \
      --checkpoint c-lite.pth --probe pannuke_probe.joblib --out work/sample1
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clite_model import IMG_SIZE, NORM_MEAN, NORM_STD, load_clite  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("run_wsi")

PANNUKE_CLASSES = ["Connective", "Dead", "Epithelial", "Inflammatory", "Neoplastic"]


# --------------------------------------------------------------------------- region
def read_region(args) -> np.ndarray:
    if args.image:
        if args.image.lower().endswith((".svs", ".tif", ".tiff")):
            import tifffile  # small single-level SVS/TIFF exports (needs imagecodecs for Aperio JPEG 2000)

            region = tifffile.imread(args.image, key=0)[..., :3]
        else:
            region = np.asarray(Image.open(args.image).convert("RGB"))
        if args.image_mpp and abs(args.image_mpp - args.mpp) > 1e-3:
            s = args.image_mpp / args.mpp
            h, w = region.shape[:2]
            region = np.asarray(Image.fromarray(region).resize((round(w * s), round(h * s)), Image.LANCZOS))
            log.info("Resampled image from %.3f to %.3f mpp (x%.2f)", args.image_mpp, args.mpp, s)
        elif not args.image_mpp:
            log.warning("--image_mpp not given: assuming the image is already at %.2f mpp.", args.mpp)
        return np.ascontiguousarray(region)
    from tiatoolbox.wsicore.wsireader import WSIReader

    reader = WSIReader.open(args.wsi)
    if args.bounds:
        bounds = tuple(args.bounds)
    else:
        w, h = reader.slide_dimensions(resolution=0, units="level")
        bounds = (0, 0, w, h)
        log.warning("No --bounds given: reading the whole slide at %.2f mpp (can be very large).", args.mpp)
    region = reader.read_bounds(bounds, resolution=args.mpp, units="mpp")
    return np.ascontiguousarray(region[..., :3])


# --------------------------------------------------------------------------- nuclei
def load_nuclei(path: Path):
    """HoVer-Net JSON ({"nuc": {id: {centroid, contour}}}), a TIAToolbox .dat, or a CSV with x,y."""
    if path.suffix == ".csv":
        with path.open() as f:
            rows = list(csv.DictReader(f))
        return [(float(r["x"]), float(r["y"])) for r in rows], [None] * len(rows)
    if path.suffix == ".dat":
        inst = joblib.load(path)
    else:
        data = json.loads(path.read_text())
        inst = data.get("nuc", data)
    cents, conts = [], []
    for v in inst.values():
        cents.append(tuple(map(float, v["centroid"])))
        c = v.get("contour")
        conts.append(np.asarray(c, dtype=np.int32) if c is not None else None)
    return cents, conts


def segment_nuclei(region: np.ndarray, work: Path, model_name: str, device: str):
    """Run HoVer-Net through TIAToolbox (1.x engine API) on the region in tile mode."""
    from tiatoolbox.models.engine.nucleus_instance_segmentor import NucleusInstanceSegmentor

    tile = work / "_seg_input.png"
    Image.fromarray(region).save(tile)
    seg_dir = work / "_hovernet"
    if seg_dir.exists():
        import shutil

        shutil.rmtree(seg_dir)
    seg = NucleusInstanceSegmentor(pretrained_model=model_name, batch_size=8, num_loader_workers=2)
    kw = dict(save_dir=str(seg_dir), mode="tile", crash_on_exception=True)
    try:
        out = seg.predict([str(tile)], device=device, **kw)  # tiatoolbox >= 1.5
    except TypeError:
        out = seg.predict([str(tile)], on_gpu=device == "cuda", **kw)  # older 1.x
    tile.unlink(missing_ok=True)
    return load_nuclei(Path(f"{out[0][1]}.dat"))


# --------------------------------------------------------------------------- features
@torch.no_grad()
def embed_cells(model, region: np.ndarray, cents, device, batch=512) -> np.ndarray:
    half = IMG_SIZE // 2
    padded = np.pad(region, ((half, half), (half, half), (0, 0)), mode="reflect")
    mean = torch.tensor(NORM_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(NORM_STD, device=device).view(1, 3, 1, 1)
    out = []
    for i in range(0, len(cents), batch):
        crops = []
        for x, y in cents[i : i + batch]:
            x0, y0 = round(x), round(y)  # = centroid - half + pad
            crops.append(padded[y0 : y0 + IMG_SIZE, x0 : x0 + IMG_SIZE])
        t = torch.from_numpy(np.stack(crops)).to(device).permute(0, 3, 1, 2).float().div_(255)
        t = (t - mean) / std
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            f = model.forward_features(t)["cls"].float()
        out.append(f.cpu().numpy())
        if (i // batch) % 20 == 0:
            log.info("  embedded %d / %d cells", min(i + batch, len(cents)), len(cents))
    return np.concatenate(out)


def cluster(feats: np.ndarray, k: int, seed=0) -> np.ndarray:
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import PCA

    x = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
    x = PCA(n_components=min(50, x.shape[1]), random_state=seed).fit_transform(x)
    lab = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=5, batch_size=4096).fit_predict(x)
    order = np.argsort(-np.bincount(lab, minlength=k))  # cluster 0 = largest
    remap = np.empty(k, int)
    remap[order] = np.arange(k)
    return remap[lab]


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--wsi", help="whole-slide image (any format TIAToolbox/OpenSlide reads)")
    src.add_argument("--image", help="an already-extracted region at --mpp")
    ap.add_argument("--image_mpp", type=float,
                    help="µm/px of --image, if it is not already at --mpp (the image is resampled)")
    ap.add_argument("--bounds", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"),
                    help="region of the WSI in level-0 pixels")
    ap.add_argument("--mpp", type=float, default=None,
                    help="resolution to analyse at (default: the probe's training mpp, else 0.5)")
    ap.add_argument("--nuclei", help="HoVer-Net .json / TIAToolbox .dat / CSV x,y in region pixels; "
                                     "omit to run HoVer-Net through TIAToolbox")
    ap.add_argument("--seg_model", default="hovernet_fast-pannuke")
    ap.add_argument("--checkpoint", required=True, help="C-LiTE .pth")
    ap.add_argument("--probe", help="joblib: {'model': sklearn classifier, 'classes': [...], 'mpp': 0.25}")
    ap.add_argument("--clusters", type=int, default=8, help="k for label-free clustering (0 = skip)")
    ap.add_argument("--min_area", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    work = Path(args.out)
    work.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    probe, classes, probe_mpp = None, None, None
    if args.probe:
        bundle = joblib.load(args.probe)
        if isinstance(bundle, dict):
            probe, classes, probe_mpp = bundle["model"], bundle.get("classes"), bundle.get("mpp")
        else:
            probe = bundle
        classes = classes or PANNUKE_CLASSES
    if args.mpp is None:
        args.mpp = probe_mpp or 0.5
    elif probe_mpp and abs(args.mpp - probe_mpp) > 1e-3:
        log.warning("Probe was trained at %.2f mpp but analysing at %.2f mpp.", probe_mpp, args.mpp)

    t0 = time.time()
    region = read_region(args)
    Image.fromarray(region).save(work / "region.jpg", quality=92)
    log.info("Region %dx%d px at %.2f mpp", region.shape[1], region.shape[0], args.mpp)

    if args.nuclei:
        cents, conts = load_nuclei(Path(args.nuclei))
    else:
        cents, conts = segment_nuclei(region, work, args.seg_model, device.type)
    areas = [int(abs(cv2_area(c))) if c is not None else -1 for c in conts]
    keep = [i for i, a in enumerate(areas) if a < 0 or a >= args.min_area]
    cents, conts, areas = [cents[i] for i in keep], [conts[i] for i in keep], [areas[i] for i in keep]
    log.info("%d nuclei", len(cents))

    model = load_clite(args.checkpoint, device)
    feats = embed_cells(model, region, cents, device)
    np.savez_compressed(work / "features.npz", feats=feats.astype(np.float16), cell_id=np.arange(len(cents)))

    pred = prob = None
    if probe is not None:
        p = probe.predict_proba(feats)
        idx = p.argmax(1)
        labels = probe.classes_[idx]
        if np.issubdtype(labels.dtype, np.integer):
            pred = [classes[int(lab)] for lab in labels]
        else:
            pred = [str(lab) for lab in labels]
        prob = p.max(1)
    clus = cluster(feats, args.clusters) if args.clusters else None

    with (work / "cells.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell_id", "x", "y", "area", "pred", "prob", "cluster"])
        for i, (x, y) in enumerate(cents):
            w.writerow([i, f"{x:.1f}", f"{y:.1f}", areas[i],
                        pred[i] if pred else "", f"{prob[i]:.3f}" if pred else "",
                        int(clus[i]) if clus is not None else ""])
    joblib.dump(conts, work / "contours.joblib", compress=3)
    (work / "meta.json").write_text(json.dumps({
        "wsi": args.wsi, "image": args.image, "bounds": args.bounds, "mpp": args.mpp,
        "width": int(region.shape[1]), "height": int(region.shape[0]),
        "classes": classes, "clusters": args.clusters, "seg_model": None if args.nuclei else args.seg_model,
    }, indent=2))
    log.info("Done in %.0fs -> %s", time.time() - t0, work)


def cv2_area(contour) -> float:
    c = np.asarray(contour, dtype=np.float64)
    if c.ndim != 2 or len(c) < 3:
        return 0.0
    x, y = c[:, 0], c[:, 1]
    return 0.5 * (np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


if __name__ == "__main__":
    main()
