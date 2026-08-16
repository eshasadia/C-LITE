# # download_hest.py
import os
import zipfile
from huggingface_hub import snapshot_download
from tqdm import tqdm


def download_hest(ids, local_dir):
    """
    Download only patches/ and st/ for the given sample IDs.
    Skips wsis/ (multi-GB), cellvit_seg/, patches_vis/, spatial_plots/, etc.
    patches/*.h5 and st/*.h5ad are all that HESTCellDataset needs.
    """
    os.makedirs(local_dir, exist_ok=True)

    # One allow-pattern per sample, restricted to patches/ and st/ only
    allow_patterns = []
    for sid in ids:
        allow_patterns.append(f"patches/{sid}.h5")
        allow_patterns.append(f"st/{sid}.h5ad")

    print(f"Downloading {len(ids)} samples to {local_dir}")
    print(f"Files to fetch: {allow_patterns}\n")

    snapshot_download(
        repo_id="MahmoodLab/hest",
        allow_patterns=allow_patterns,
        ignore_patterns=["wsis/*", "cellvit_seg/*", "patches_vis/*",
                         "spatial_plots/*", "pixel_size_vis/*",
                         "xenium_seg/*", "transcripts/*"],
        repo_type="dataset",
        local_dir=local_dir,
    )

    print("\nDownload complete. Contents:")
    for folder in ["patches", "st"]:
        folder_path = os.path.join(local_dir, folder)
        if os.path.exists(folder_path):
            files = sorted(os.listdir(folder_path))
            print(f"\n  {folder}/")
            for f in files:
                size_mb = os.path.getsize(os.path.join(folder_path, f)) / 1e6
                print(f"    {f}  ({size_mb:.1f} MB)")
        else:
            print(f"\n  {folder}/ — NOT FOUND, check sample IDs")


# =============================================================================
# Pick your tissue below — uncomment one block
# =============================================================================

# HEST-breast (5 slides)
# ids       = ["NCBI783", "NCBI785", "TENX94", "TENX95", "TENX98"]
# local_dir = "/tia/phd-data/u5552013/ImmunoDINO_datasets/HEST/breast"

# HEST-bowel (5 slides)
# ids       = ["TENX149", "TENX148", "TENX147", "TENX139", "TENX111"]
# local_dir = "/tia/phd-data/u5552013/ImmunoDINO_datasets/HEST/bowel"

# HEST-lung (20 slides — large, download overnight)
ids       = ["NCBI856", "NCBI857", "NCBI858", "NCBI859", "NCBI860",
             "NCBI861", "NCBI864", "NCBI865", "NCBI866", "NCBI867",
             "NCBI870", "NCBI873", "NCBI875", "NCBI876", "NCBI879",
             "NCBI880", "NCBI881", "NCBI882", "NCBI883", "NCBI884"]
local_dir = "/tia/phd-data/u5552013/ImmunoDINO_datasets/HEST/lung"

# =============================================================================

download_hest(ids, local_dir)