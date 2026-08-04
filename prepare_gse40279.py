"""Download and prepare the first 164 samples of GSE40279.

The selected official GEO block contains GSM989827--GSM989990.  It is large
(~301 MB compressed), but much smaller than the 1.1 GB combined file and gives
the short-row biomedical setting needed by this project.

This script writes a memory-mappable samples-by-CpGs NumPy matrix plus ages and
identifiers.  It does not perform feature selection; selection must occur
inside each training fold in run_real_world_methylation.py.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import html
import json
import re
import shutil
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np


BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE40nnn/GSE40279/suppl"
BETA_NAME = "GSE40279_average_beta_GSM989827-GSM989990.txt.gz"
KEY_NAME = "GSE40279_sample_key.txt.gz"


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"Already present: {destination}")
        return
    partial = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {url}")
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (academic research; GSE40279 benchmark)",
            "Accept": "application/octet-stream,*/*",
        },
    )
    try:
        response = urllib.request.urlopen(request, timeout=120)
    except urllib.error.HTTPError as error:
        if error.code != 403:
            raise
        # Some NCBI mirrors reject direct programmatic file requests while
        # the accession-aware GEO endpoint remains available.
        fallback = (
            "https://www.ncbi.nlm.nih.gov/geo/download/"
            f"?acc=GSE40279&file={urllib.parse.quote(destination.name)}&format=file"
        )
        print(f"Direct mirror returned 403; retrying through {fallback}")
        request = urllib.request.Request(fallback, headers=dict(request.header_items()))
        response = urllib.request.urlopen(request, timeout=120)
    with response, partial.open("wb") as out:
        shutil.copyfileobj(response, out, length=1024 * 1024)
    partial.replace(destination)
    print(f"Wrote {destination}")


def split_fields(line: str) -> list[str]:
    return next(csv.reader([line.rstrip("\n\r")], delimiter="\t"))


def read_header(beta_path: Path) -> list[str]:
    with gzip.open(beta_path, "rt", newline="") as handle:
        header = split_fields(handle.readline())
    if len(header) < 2:
        raise ValueError("The beta-value file header has fewer than two columns")
    return [value.strip().strip('"') for value in header[1:]]


def parse_age(value: str) -> float | None:
    value = value.strip().strip('"')
    match = re.search(r"(?<!\d)(\d{1,3})(?:\.\d+)?\s*(?:y|yr|years?)?\b", value, re.I)
    if not match:
        return None
    age = float(match.group(1))
    return age if 0 < age < 125 else None


def fetch_age_map(cache_path: Path) -> dict[str, float]:
    """Recover study-ID ages from official GEO sample titles.

    The supplementary "sample key" maps study IDs to array positions; despite
    its name it does not contain age.  GEO sample titles have the stable form
    "age 67y 1001", so the accession page supplies the missing mapping.
    """
    if cache_path.exists():
        page = cache_path.read_text(errors="replace")
    else:
        url = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE40279"
        request = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0 (academic research; GSE40279 benchmark)"}
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            page = response.read().decode("utf-8", errors="replace")
        cache_path.write_text(page)
    text = html.unescape(re.sub(r"<[^>]+>", " ", page))
    text = re.sub(r"\s+", " ", text)
    return {
        study_id: float(age)
        for age, study_id in re.findall(r"age\s+(\d+(?:\.\d+)?)\s*y\s+(\d+)", text, re.I)
    }


def read_ages(key_path: Path, wanted_samples: list[str]) -> np.ndarray:
    """Map beta-matrix study IDs (for example X1001) to chronological age."""
    wanted = set(wanted_samples)
    found: dict[str, float] = {}
    with gzip.open(key_path, "rt", errors="replace") as handle:
        rows = [split_fields(line) for line in handle if line.strip()]

    # Each row normally contains one GSM id and one age field.  Locate them
    # rather than depending on a particular header spelling.
    for row in rows:
        sample = next((cell.strip().strip('"') for cell in row if cell.strip().strip('"') in wanted), None)
        if sample is None:
            continue
        age = None
        for cell in row:
            lower = cell.lower()
            if "age" in lower or re.fullmatch(r"\s*\d{1,3}\s*(?:y|yr|years?)?\s*", cell, re.I):
                age = parse_age(cell)
                if age is not None:
                    break
        if age is not None:
            found[sample] = age

    missing = [sample for sample in wanted_samples if sample not in found]
    if missing:
        geo_ages = fetch_age_map(key_path.parent / "GSE40279_accession.html")
        for sample in missing:
            study_id = sample.removeprefix("X")
            if study_id in geo_ages:
                found[sample] = geo_ages[study_id]
        missing = [sample for sample in wanted_samples if sample not in found]
    if missing:
        raise ValueError(
            f"Could not recover ages for {len(missing)} samples from GEO metadata. "
            f"First missing IDs: {missing[:5]}"
        )
    return np.asarray([found[sample] for sample in wanted_samples], dtype=np.float32)


def count_probes(beta_path: Path) -> int:
    with gzip.open(beta_path, "rt", errors="replace") as handle:
        next(handle)
        return sum(1 for line in handle if line.strip())


def write_matrix(beta_path: Path, output_path: Path, n_samples: int, n_probes: int) -> None:
    matrix = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.float32, shape=(n_samples, n_probes)
    )
    probe_ids_path = output_path.with_name("probe_ids.txt")
    with gzip.open(beta_path, "rt", errors="replace") as handle, probe_ids_path.open("w") as ids:
        next(handle)
        feature_index = 0
        for line in handle:
            if not line.strip():
                continue
            fields = split_fields(line)
            if len(fields) != n_samples + 1:
                raise ValueError(
                    f"Probe row {feature_index + 2} has {len(fields)} fields; expected {n_samples + 1}"
                )
            ids.write(fields[0].strip().strip('"') + "\n")
            values = [np.nan if value in {"", "NA", "NaN", "null"} else float(value) for value in fields[1:]]
            matrix[:, feature_index] = values
            feature_index += 1
            if feature_index % 50_000 == 0:
                print(f"Parsed {feature_index:,}/{n_probes:,} CpGs")
    matrix.flush()
    if feature_index != n_probes:
        raise ValueError(f"Parsed {feature_index} probes but counted {n_probes}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="data/gse40279")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    rawdir = outdir / "raw"
    rawdir.mkdir(parents=True, exist_ok=True)
    beta_path = rawdir / BETA_NAME
    key_path = rawdir / KEY_NAME

    if not args.skip_download:
        download(f"{BASE}/{BETA_NAME}", beta_path)
        download(f"{BASE}/{KEY_NAME}", key_path)

    if not beta_path.exists() or not key_path.exists():
        raise FileNotFoundError("Required GEO files are missing; rerun without --skip-download")

    sample_ids = read_header(beta_path)
    ages = read_ages(key_path, sample_ids)
    n_probes = count_probes(beta_path)
    print(f"Preparing {len(sample_ids)} people × {n_probes:,} CpGs")

    matrix_path = outdir / "methylation.npy"
    write_matrix(beta_path, matrix_path, len(sample_ids), n_probes)
    np.save(outdir / "ages.npy", ages)
    (outdir / "sample_ids.txt").write_text("\n".join(sample_ids) + "\n")
    metadata = {
        "accession": "GSE40279",
        "source": f"{BASE}/{BETA_NAME}",
        "n_samples": len(sample_ids),
        "n_cpgs": n_probes,
        "age_min": float(ages.min()),
        "age_max": float(ages.max()),
        "matrix_shape": [len(sample_ids), n_probes],
        "matrix_dtype": "float32",
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
