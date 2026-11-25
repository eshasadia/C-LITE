# C-LiTE slide viewer (GitHub Pages)

A static web page, like the CytoFormer one, where people can pan and zoom held-out whole slides
and see C-LiTE's per-nucleus output drawn on the H&E:

- **Cell types:** a linear probe on frozen C-LiTE features (e.g. the PanNuke probe from the notebook)
- **Clusters:** k-means on the same features, with no labels. This shows label-free phenotyping, which suits a feature extractor.
- **Reference** (optional): ground-truth annotations in a second panel that pans and zooms with the first.

The site has no server or backend. Each layer is a Deep Zoom (DZI) tile pyramid in `docs/slides/`, shown with OpenSeadragon.

## Files

```
docs/
  index.html            the viewer (reads docs/slides/samples.json)
  .nojekyll
  slides/samples.json   one entry per slide; written by render_slide.py
scripts/
  clite_model.py        C-LiTE encoder + checkpoint loader (copied from the TIAToolbox notebook)
  run_wsi.py            slide region -> HoVer-Net nuclei -> C-LiTE features -> cell type + cluster
  render_slide.py       results -> coloured overlays -> DZI tiles + samples.json entry
```

Copy `docs/` and `scripts/` into the root of the C-LITE repo.

## 1. Save a linear probe (once)

At the end of `notebooks/14_clite_cell_feature_learning.ipynb`, after `logreg.fit(...)`, add:

```python
import joblib
joblib.dump({"model": logreg, "classes": CLASSES, "mpp": 0.25}, "pannuke_probe.joblib")
```

PanNuke tiles are 40× (~0.25 µm/px). `run_wsi.py` reads the slide at the probe's `mpp`, so the
crops it classifies are at the same scale the probe was trained on. For a probe trained on 20× data,
set `"mpp": 0.5`.

## 2. Run C-LiTE on a slide region (GPU)

```bash
pip install tiatoolbox torch torchvision scikit-learn joblib pillow
python scripts/run_wsi.py \
    --wsi /data/TCGA-A2-A0T2-01Z-00-DX1.svs \
    --bounds 30000 20000 38000 26000 \
    --checkpoint c-lite.pth --probe pannuke_probe.joblib \
    --clusters 8 --out work/TCGA-A2-A0T2
```

- `--bounds` is in level-0 pixels. Something around 8k × 6k at 40× (a few mm²) is a good demo size.
  At 0.25 mpp that region holds ~100–200k nuclei.
- Nuclei come from HoVer-Net (`hovernet_fast-pannuke`) through TIAToolbox's 1.x engine API.
  If you use TIAToolbox 2.x, or already have nuclei, pass `--nuclei` with a HoVer-Net JSON, a
  TIAToolbox `.dat` file or a CSV of `x,y` in region pixels. You can also pass `--image` with a
  pre-extracted region instead of `--wsi`; this can be a
  small single-level `.svs`/`.tif` crop, and `--image_mpp` tells the script its resolution so it is
  resampled to the probe's scale (needs `pip install tifffile imagecodecs`).
- The outputs are `region.jpg`, `cells.csv`, `contours.joblib`, `features.npz` and `meta.json`.

No HoVer-Net? `scripts/detect_nuclei_stardist.py` uses StarDist's H&E model (weights come from
GitHub, so it works where Hugging Face is blocked) and writes a JSON that `--nuclei` accepts.

## 3. Build the tiles

```bash
python scripts/render_slide.py --work work/TCGA-A2-A0T2 --id TCGA-A2-A0T2 \
    --name "Breast" --subtitle "TCGA-BRCA" [--gt reference.csv]
```

`--gt` is optional: a CSV of `x,y,label` in the same region frame (for example from a PUMA, MONKEY or
CoNIC/Lizard slide with expert labels). With it, the viewer's *Reference* button opens a second panel.
Repeat steps 2–3 for every slide. Each run adds an entry to `docs/slides/samples.json`.

## 4. Publish

Commit `docs/` (tiles included). Then go to **Settings → Pages → Deploy from a branch → main / docs**. The site will be at
`https://eshasadia.github.io/C-LITE/`. Add the link to the README.

Size: an 8k × 6k region is roughly 40–80 MB of tiles across all layers (`--max_side`, default 12000, caps the size). That is fine for
2–4 slides. For many more, host `docs/slides/` elsewhere (Hugging Face dataset, S3) and change
the `slides/` paths in `index.html`.

## Example included

`docs/slides/` already holds one slide: a 1600 × 1200 px crop of TCGA-BR-4267 (stomach
adenocarcinoma). It was run with C-LiTE\*U (`clite_u.pth`) and StarDist nuclei (2× upsample, threshold 0.4), and shows 2,288 nuclei in
6 k-means clusters. Its cell-type layer is provisional: the features were split into 20 k-means clusters, and each cluster was labelled by eye
(by Claude, not a pathologist; mapping in `slides/TCGA-BR-4267_cluster_labels.json`). Replace it with a trained probe's output. The cluster names in
`samples.json` were written by eye, and the page says so. Replace or add slides as you go.

## Cell-type classes

On a slide without a cell-type layer, the **Cell types** button lists the planned classes (Tumor, Epithelium,
Stroma, Endothelium, Lymphocyte, Plasma cell, Macrophage) and says they are not predicted yet. A layer can carry a `note` in
`samples.json` (shown under the layer buttons) to say how its labels were made.
To change the list, edit `PLANNED_CLASSES` in `docs/index.html`. Train the probe with exactly these
class names, and `render_slide.py` will colour them to match the legend.

## Viewer controls

Pick a slide, then switch between **Cell types** and **Clusters**, **Overlay** and **H&E only**,
change the opacity, and turn the **Reference** panel on or off. Click a legend row to hide that class.
Keyboard: `o` toggles the overlay, `c` switches the layer. The URL hash (`#TCGA-A2-A0T2`) links straight to a slide.

## Test locally

```bash
python -m http.server -d docs 8000   # open http://localhost:8000
```
