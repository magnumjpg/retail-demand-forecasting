"""Download and verify the UCI Online Retail II dataset.

    python src/download_data.py [--force]

Downloads https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip,
checks the sha256 of both the zip and the extracted workbook, and leaves a
manifest in data/raw/manifest.json. The 45 MB workbook is kept locally so the
project is self-contained (data/raw/ is git-ignored).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import zipfile
from datetime import datetime, timezone

import requests

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from config import (MANIFEST, RAW_XLSX, RAW_ZIP, SHEETS, URL, XLSX_SHA256,
                    ZIP_SHA256)


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path, expected: str) -> str:
    digest = sha256(path)
    if digest != expected:
        raise RuntimeError(f"sha256 mismatch for {path.name}: {digest} != {expected}")
    return digest


def download(force: bool = False) -> dict:
    if RAW_XLSX.exists() and not force:
        digest = verify(RAW_XLSX, XLSX_SHA256)
        print(f"[download] found {RAW_XLSX.name} (sha256 ok) - skipping download")
        if MANIFEST.exists():
            old = json.loads(MANIFEST.read_text(encoding="utf-8"))
            if (old.get("xlsx_sha256") == digest
                    and old.get("bytes") == RAW_XLSX.stat().st_size):
                print("[download] manifest still valid - reusing sheet audit")
                return old
        info = {"source": "local", "xlsx_sha256": digest}
    else:
        if force or not RAW_ZIP.exists():
            print(f"[download] GET {URL}")
            with requests.get(URL, stream=True, timeout=180) as r:
                r.raise_for_status()
                tmp = RAW_ZIP.with_suffix(".zip.part")
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
                tmp.replace(RAW_ZIP)
        zip_digest = verify(RAW_ZIP, ZIP_SHA256)
        print(f"[download] zip sha256 ok ({RAW_ZIP.stat().st_size/1e6:.1f} MB)")
        with zipfile.ZipFile(RAW_ZIP) as zf:
            member = next(n for n in zf.namelist() if n.lower().endswith(".xlsx"))
            with zf.open(member) as src, open(RAW_XLSX, "wb") as dst:
                dst.write(src.read())
        digest = verify(RAW_XLSX, XLSX_SHA256)
        print(f"[download] extracted {RAW_XLSX.name} (sha256 ok)")
        info = {"source": URL, "zip_sha256": zip_digest, "xlsx_sha256": digest}

    import pandas as pd
    with pd.ExcelFile(RAW_XLSX) as xl:
        sheets = list(xl.sheet_names)
        rows = {s: int(len(pd.read_excel(RAW_XLSX, sheet_name=s, usecols=[0])))
                for s in sheets}
    info.update({
        "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "file": str(RAW_XLSX.relative_to(MANIFEST.parents[2])),
        "bytes": RAW_XLSX.stat().st_size,
        "sheets": sheets,
        "sheet_rows": rows,
        "expected_sheets": SHEETS,
    })
    MANIFEST.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"[download] manifest -> {MANIFEST}")
    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()
    download(force=args.force)
