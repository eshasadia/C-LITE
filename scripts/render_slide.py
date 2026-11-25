"""Turn the output of run_wsi.py into Deep Zoom (DZI) layers for the web viewer in docs/.

Writes, under docs/slides/:
  <id>_he.dzi        H&E base layer (JPEG tiles)
  <id>_pred.dzi      C-LiTE cell types, nuclei filled with class colours (transparent PNG tiles)
  <id>_clusters.dzi  label-free k-means clusters of the C-LiTE features   (transparent PNG tiles)
  <id>_gt.dzi        optional reference annotations, if --gt is given
and adds or updates the slide's entry in docs/slides/samples.json.

Usage
  python scripts/render_slide.py --work work/TCGA-XX --id TCGA-XX --name "Breast" \
      --subtitle "TCGA-BRCA, held out" [--gt reference.csv] [--max_side 12000]

--gt: CSV with x, y, label in region pixels (the same frame as cells.csv). Each cell is drawn
      as a filled dot in its label's colour.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import joblib
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None

# Saturated colours that stay readable on pink/purple H&E. Unknown classes get a fallback colour.
CLASS_COLORS = {
    "Neoplastic": "#e8262b", "Inflammatory": "#2f6bff", "Connective": "#18b05a",
    "Epithelial": "#ff9f0a", "Dead": "#1d1d1f",
    "Tumour": "#e8262b", "Tumor": "#e8262b", "Lymphocyte": "#2f6bff", "Plasma": "#8a4dff",
    "Macrophage": "#00c2b8", "Neutrophil": "#9bd31b", "Eosinophil": "#ff5fa2",
    "Stroma": "#18b05a", "Endothelial": "#00a3d9",
    "Epithelium": "#ff9f0a", "Endothelium": "#00a3d9", "Plasma cell": "#8a4dff", "Mitosis": "#ffd400",
}
FALLBACK = ["#8a4dff", "#00c2b8", "#ff5fa2", "#9bd31b", "#00a3d9", "#ffd400", "#a0522d", "#7f7f7f"]
CLUSTER_COLORS = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#46f0f0",
                  "#f032e6", "#bcf60c", "#fabebe", "#008080", "#e6beff", "#9a6324", "#800000",
                  "#aaffc3", "#808000"]


def hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def colour_map(names):
    out, k = {}, 0
    for n in names:
        if n in CLASS_COLORS:
            out[n] = CLASS_COLORS[n]
        else:
            out[n] = FALLBACK[k % len(FALLBACK)]
            k += 1
    return out


# --------------------------------------------------------------------------- drawing
def draw_layer(size, scale, items, radius=5, outline=True):
    """items: list of (rgb, contour or None, (x, y)). Returns an RGBA image."""
    w, h = size
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    r = max(1.5, radius * scale)
    for rgb, contour, (x, y) in items:
        if contour is not None and len(contour) >= 3:
            pts = [(float(px) * scale, float(py) * scale) for px, py in contour]
            d.polygon(pts, fill=rgb + (200,), outline=(tuple(int(c * 0.55) for c in rgb) + (255,)) if outline else None)
        else:
            cx, cy = x * scale, y * scale
            d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=rgb + (230,))
    return img


# --------------------------------------------------------------------------- DZI
DZI_XML = ('<?xml version="1.0" encoding="UTF-8"?>\n<Image xmlns="http://schemas.microsoft.com/deepzoom/2008" '
           'Format="{fmt}" Overlap="{ov}" TileSize="{ts}"><Size Width="{w}" Height="{h}"/></Image>\n')


def write_dzi(img: Image.Image, prefix: Path, fmt="jpg", tile=512, overlap=1, quality=82):
    """Write <prefix>.dzi and <prefix>_files/<level>/<col>_<row>.<fmt>. Skips empty PNG tiles."""
    w, h = img.size
    n_levels = math.ceil(math.log2(max(w, h))) + 1
    files = Path(f"{prefix}_files")
    if files.exists():
        import shutil

        shutil.rmtree(files)
    prefix.with_suffix(".dzi").write_text(DZI_XML.format(fmt=fmt, ov=overlap, ts=tile, w=w, h=h))
    level_img = img
    for level in range(n_levels - 1, -1, -1):  # full resolution first, halve each step
        lw, lh = level_img.size
        d = files / str(level)
        d.mkdir(parents=True, exist_ok=True)
        for col in range(math.ceil(lw / tile)):
            for row in range(math.ceil(lh / tile)):
                x0, y0 = col * tile, row * tile
                box = (max(0, x0 - overlap), max(0, y0 - overlap),
                       min(lw, x0 + tile + overlap), min(lh, y0 + tile + overlap))
                t = level_img.crop(box)
                if fmt == "png":
                    if t.getchannel("A").getbbox() is None:
                        t = Image.new("RGBA", (1, 1), (0, 0, 0, 0))  # tiny stand-in for empty tiles
                    t.save(d / f"{col}_{row}.png", optimize=True)
                else:
                    t.convert("RGB").save(d / f"{col}_{row}.jpg", quality=quality)
        if level:
            level_img = level_img.resize((max(1, math.ceil(lw / 2)), max(1, math.ceil(lh / 2))), Image.LANCZOS)
    return n_levels


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True, help="output folder of run_wsi.py")
    ap.add_argument("--id", required=True, help="short id used in file names, e.g. TCGA-A2-A0T2")
    ap.add_argument("--name", required=True, help="label in the slide list, e.g. 'Breast'")
    ap.add_argument("--subtitle", default="", help="second line in the slide list")
    ap.add_argument("--gt", help="reference CSV: x, y, label (region pixels)")
    ap.add_argument("--gt_name", default="Reference annotations")
    ap.add_argument("--docs", default="docs", help="the GitHub Pages folder")
    ap.add_argument("--max_side", type=int, default=12000, help="downscale so the long side is at most this")
    args = ap.parse_args()

    work, slides = Path(args.work), Path(args.docs) / "slides"
    slides.mkdir(parents=True, exist_ok=True)
    meta = json.loads((work / "meta.json").read_text())
    he = Image.open(work / "region.jpg").convert("RGB")
    with (work / "cells.csv").open() as f:
        cells = list(csv.DictReader(f))
    conts = joblib.load(work / "contours.joblib")

    scale = min(1.0, args.max_side / max(he.size))
    size = (round(he.size[0] * scale), round(he.size[1] * scale))
    if scale < 1:
        he = he.resize(size, Image.LANCZOS)
    xy = [(float(c["x"]), float(c["y"])) for c in cells]
    entry = {"id": args.id, "name": args.name, "subtitle": args.subtitle,
             "width": size[0], "height": size[1], "mpp": meta["mpp"] / scale,
             "nuclei": len(cells), "layers": {}}

    print(f"[{args.id}] H&E {size[0]}x{size[1]}")
    write_dzi(he, slides / f"{args.id}_he", fmt="jpg")

    if cells and cells[0]["pred"]:
        names = meta.get("classes") or sorted({c["pred"] for c in cells})
        cmap = colour_map(names)
        counts = Counter(c["pred"] for c in cells)
        items = [(hex_rgb(cmap[c["pred"]]), conts[i], xy[i]) for i, c in enumerate(cells)]
        write_dzi(draw_layer(size, scale, items), slides / f"{args.id}_pred", fmt="png")
        entry["layers"]["pred"] = {"label": "Cell types", "count": len(cells),
                                   "legend": [{"name": n, "color": cmap[n], "n": counts.get(n, 0)} for n in names]}
        print(f"[{args.id}] cell types: {dict(counts)}")

    if cells and cells[0]["cluster"] != "":
        k = max(int(c["cluster"]) for c in cells) + 1
        counts = Counter(int(c["cluster"]) for c in cells)
        items = [(hex_rgb(CLUSTER_COLORS[int(c["cluster"]) % 16]), conts[i], xy[i]) for i, c in enumerate(cells)]
        write_dzi(draw_layer(size, scale, items), slides / f"{args.id}_clusters", fmt="png")
        entry["layers"]["clusters"] = {"label": "Clusters", "count": len(cells),
                                       "legend": [{"name": f"Cluster {j + 1}", "color": CLUSTER_COLORS[j % 16],
                                                   "n": counts.get(j, 0)} for j in range(k)]}
        print(f"[{args.id}] {k} clusters")

    if args.gt:
        with open(args.gt) as f:
            gt = list(csv.DictReader(f))
        names = sorted({g["label"] for g in gt})
        cmap = colour_map(names)
        counts = Counter(g["label"] for g in gt)
        items = [(hex_rgb(cmap[g["label"]]), None, (float(g["x"]), float(g["y"]))) for g in gt]
        write_dzi(draw_layer(size, scale, items), slides / f"{args.id}_gt", fmt="png")
        entry["layers"]["gt"] = {"label": args.gt_name, "count": len(gt),
                                 "legend": [{"name": n, "color": cmap[n], "n": counts[n]} for n in names]}

    index = slides / "samples.json"
    samples = json.loads(index.read_text()) if index.exists() else []
    samples = [s for s in samples if s["id"] != args.id] + [entry]
    index.write_text(json.dumps(samples, indent=2))
    print(f"[{args.id}] added to {index}")


if __name__ == "__main__":
    main()
