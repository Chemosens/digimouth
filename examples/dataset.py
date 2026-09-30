"""Download the Digimouth experimental data used by the examples.

The data are published on Recherche Data Gouv:
    Peltier, Caroline (2026). Dataset of aroma release in vivo (Digimouth), version 1.0.
    https://doi.org/10.57745/OVC3RL  (licence Etalab 2.0)

Each file is downloaded the first time it is needed, checked against the MD5 checksum
published with the dataset, and kept in a local cache so later runs are immediate.

Cache folder: the DIGIMOUTH_DATA_DIR environment variable if set, otherwise
    Windows : %LOCALAPPDATA%\\digimouth\\data
    others  : ~/.cache/digimouth/data

Usage in an example:
    import pandas as pd
    from dataset import fetch, experimental_mean_curve

    senso = pd.read_excel(fetch("senso_sol.xlsx"))
    metadata = pd.read_csv(fetch("metadata_sol.csv"), sep=";")
    sol_expe = experimental_mean_curve("sol")     # mean measured curve, to compare with the model

In the metadata, rep == 0 marks warm-up trials (done by half of the subjects before the
three repetitions); they are left out of the mean curves.
"""
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

DOI = "10.57745/OVC3RL"
DATASET_VERSION = "1.0"
_DOWNLOAD_URL = "https://entrepot.recherche.data.gouv.fr/api/access/datafile/{id}"

# name -> (file id in the dataset, MD5 of the file, size in bytes)
# Files marked "original" were converted to .tab by the repository; the original
# .csv / .xlsx is requested with ?format=original.
FILES = {
    "conc_aci_sol.csv": (723038, "e14b19283ff91511a7ce8d3aa6591d6e", 102272116, "original"),
    "conc_aci_gel.csv": (723034, "40299951edf1196ab0f0e400e1df381d", 91625909, "original"),
    "conc_aci_gus.csv": (723037, "90e71fb4bb4c50945ec517f3864673e3", 92680024, "original"),
    "senso_sol.xlsx": (723039, "ef135f66504704bfb980caf325acd95a", 183980, "original"),
    "senso_gel.xlsx": (723040, "8241ad790604049cf34a6493646a51b7", None, ""),
    "senso_gus.xlsx": (723044, "0f19acc2eab3ba57c6ca675068f10454", None, ""),
    "metadata_sol.csv": (723035, "3e79dc071f3d6c6ebb3820fedaf27c8d", None, ""),
    "metadata_gel.csv": (723041, "f54793a0aa24aa55f041c45704b00ee5", None, ""),
    "metadata_gus.csv": (723042, "55470615f03484f4974263ea39112772", None, ""),
}


def data_dir():
    """Folder where downloaded files are kept."""
    if os.environ.get("DIGIMOUTH_DATA_DIR"):
        return Path(os.environ["DIGIMOUTH_DATA_DIR"])
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / "digimouth" / "data"
    return Path.home() / ".cache" / "digimouth" / "data"


def _md5(path):
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(name):
    """Return the local path of a dataset file, downloading it the first time."""
    if name not in FILES:
        raise KeyError(f"Unknown file {name!r}. Available files: {', '.join(sorted(FILES))}")
    file_id, md5, size, fmt = FILES[name]
    path = data_dir() / name
    if path.exists() and (size is None or path.stat().st_size == size):
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    url = _DOWNLOAD_URL.format(id=file_id) + ("?format=original" if fmt == "original" else "")
    partial = path.with_name(path.name + ".part")
    print(f"Downloading {name} from doi:{DOI} to {path.parent} ...", file=sys.stderr)
    with urllib.request.urlopen(url, timeout=60) as response, open(partial, "wb") as out:
        while True:
            block = response.read(1 << 20)
            if not block:
                break
            out.write(block)
    if _md5(partial) != md5:
        partial.unlink()
        raise RuntimeError(
            f"{name}: the downloaded file does not match the published checksum "
            f"(dataset doi:{DOI}, version {DATASET_VERSION}). Try again later."
        )
    partial.replace(path)
    return path


# Protocol names used in the published metadata -> names used in the examples.
_FOP_NAMES = {"sol": {"long": "rare", "fast": "freq"}}


def experimental_mean_curve(product, t_max=120):
    """Mean aroma release curve measured across subjects, one row per second and protocol.

    Computed from the published data (conc_aci_<product>.csv and metadata_<product>.csv):
    warm-up trials (rep == 0) are left out, times between 0 and t_max are rounded to the
    nearest second, and all measurements of each second are averaged.

    product: "sol", "gel" or "gus".
    Returns a DataFrame with columns time_bin, mean_intensity, sd_intensity, fop, std_err
    (std_err = sd_intensity / sqrt(number of recordings)).
    """
    import numpy as np
    import pandas as pd

    ptr = pd.read_csv(fetch(f"conc_aci_{product}.csv"), usecols=["time", "file", "conc_ACI"])
    metadata = pd.read_csv(fetch(f"metadata_{product}.csv"), sep=";")
    metadata = metadata.assign(file=metadata["file"].astype(str) + ".h5")[["file", "fop", "rep"]]
    data = ptr.merge(metadata, on="file")
    data = data[(data["rep"] != 0) & (data["time"] >= 0) & (data["time"] <= t_max)]
    data = data.assign(time_bin=np.round(data["time"]).astype(int))

    groups = data.groupby(["fop", "time_bin"])
    curve = groups["conc_ACI"].agg(mean_intensity="mean", sd_intensity="std").reset_index()
    curve["std_err"] = curve["sd_intensity"] / np.sqrt(groups["file"].nunique().to_numpy())
    curve["fop"] = curve["fop"].replace(_FOP_NAMES.get(product, {}))
    return curve[["time_bin", "mean_intensity", "sd_intensity", "fop", "std_err"]]


if __name__ == "__main__":
    # python dataset.py            -> download every file
    # python dataset.py senso_sol.xlsx metadata_sol.csv
    for name in sys.argv[1:] or FILES:
        print(fetch(name))
