"""Detect nuclei with StarDist (2D_versatile_he) and write a HoVer-Net-style JSON for run_wsi.py --nuclei.

Use this when HoVer-Net through TIAToolbox is not convenient. StarDist's H&E model was trained on
~40x images, so a ~20x image (0.5 µm/px) is upsampled 2x for detection, and the centroids and
outlines are scaled back to the image's own pixels.

  pip install stardist tensorflow-cpu tifffile imagecodecs
  python scripts/detect_nuclei_stardist.py --image crop.svs --upsample 2 --out nuclei.json
  python scripts/run_wsi.py --image crop.svs --image_mpp 0.5 --mpp 0.5 --nuclei nuclei.json ...

The output is in the coordinates of --image as given. If run_wsi.py resamples the image
(--image_mpp differs from --mpp), run detection on the resampled image instead.
"""
import argparse
import json

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--upsample", type=float, default=2.0, help="2 for ~0.5 µm/px images, 1 for ~0.25")
    ap.add_argument("--prob", type=float, default=0.4,
                    help="probability threshold; the model default (0.69) misses ~35%% of nuclei on TCGA crops")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from csbdeep.utils import normalize
    from stardist.models import StarDist2D

    if args.image.lower().endswith((".svs", ".tif", ".tiff")):
        import tifffile

        img = tifffile.imread(args.image, key=0)[..., :3]
    else:
        img = np.asarray(Image.open(args.image).convert("RGB"))
    s = args.upsample
    x = img if s == 1 else np.asarray(Image.fromarray(img).resize(
        (round(img.shape[1] * s), round(img.shape[0] * s)), Image.BICUBIC))
    model = StarDist2D.from_pretrained("2D_versatile_he")
    tiles = (max(1, x.shape[0] // 2048), max(1, x.shape[1] // 2048), 1)
    _, det = model.predict_instances(normalize(x, 1, 99.8, axis=(0, 1)), prob_thresh=args.prob, n_tiles=tiles)

    nuc = {}
    for i, (p, c, pr) in enumerate(zip(det["points"] / s, det["coord"] / s, det["prob"])):
        nuc[str(i)] = {"centroid": [float(p[1]), float(p[0])],
                       "contour": np.round(np.stack([c[1], c[0]], 1)).astype(int).tolist(),
                       "prob": float(pr)}
    with open(args.out, "w") as f:
        json.dump({"nuc": nuc}, f)
    print(f"{len(nuc)} nuclei -> {args.out}")


if __name__ == "__main__":
    main()
