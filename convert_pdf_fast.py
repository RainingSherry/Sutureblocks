#!/usr/bin/env python3
"""High-performance parallel PDF to TXT converter."""

import os
import sys
import time
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path("/data/luolie/缝合模块")
LOG_FILE = ROOT / "pdf_conversion.log"
WORKERS = 96  # Conservative: leave headroom for system

def convert_one(pdf_path_str):
    pdf_path = Path(pdf_path_str)
    txt_path = pdf_path.with_suffix(".txt")

    if txt_path.exists():
        return ("SKIP", pdf_path_str)

    try:
        result = subprocess.run(
            ["pdftotext", "-enc", "UTF-8", str(pdf_path), str(txt_path)],
            capture_output=True, timeout=300
        )
        if result.returncode == 0 and txt_path.exists():
            return ("OK", pdf_path_str)
        else:
            return ("FAIL", pdf_path_str)
    except subprocess.TimeoutExpired:
        return ("TIMEOUT", pdf_path_str)
    except Exception as e:
        return ("ERROR", f"{pdf_path_str}: {e}")

def main():
    # Find all PDFs
    pdf_files = sorted(ROOT.rglob("*.pdf"))
    total = len(pdf_files)
    print(f"Found {total} PDFs to convert")

    # Count existing txt
    existing = sum(1 for f in pdf_files if f.with_suffix(".txt").exists())
    print(f"Already have .txt: {existing}")
    to_convert = [str(f) for f in pdf_files if not f.with_suffix(".txt").exists()]
    remaining = len(to_convert)
    print(f"Need to convert: {remaining}  (workers={WORKERS})")

    if not to_convert:
        print("Nothing to do.")
        return

    # Clear old log
    with open(LOG_FILE, "w") as f:
        f.write(f"Started {time.strftime('%Y-%m-%d %H:%M:%S')}  total={total}  to_convert={remaining}  workers={WORKERS}\n")

    results = {"OK": 0, "SKIP": existing, "FAIL": 0, "TIMEOUT": 0, "ERROR": 0}
    done = 0
    start = time.time()

    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(convert_one, p): p for p in to_convert}
        for future in as_completed(futures):
            status, detail = future.result()
            results[status] = results.get(status, 0) + 1
            done += 1

            with open(LOG_FILE, "a") as lf:
                lf.write(f"{status}: {detail}\n")

            # Progress every 20 files
            if done % 20 == 0 or done == remaining:
                elapsed = time.time() - start
                rate = done / elapsed if elapsed > 0 else 0
                eta = (remaining - done) / rate if rate > 0 else 0
                print(f"  [{done}/{remaining}] OK={results['OK']} FAIL={results['FAIL']}  {elapsed:.0f}s  ETA={eta:.0f}s")

    # Final summary
    total_ok = results["OK"]
    total_skip = results["SKIP"]
    total_fail = results["FAIL"]
    elapsed = time.time() - start

    print(f"\n=== DONE in {elapsed:.0f}s ===")
    print(f"  OK:    {total_ok}")
    print(f"  SKIP:  {total_skip}")
    print(f"  FAIL:  {total_fail}")

    with open(LOG_FILE, "a") as f:
        f.write(f"\nDONE {time.strftime('%Y-%m-%d %H:%M:%S')}  elapsed={elapsed:.0f}s\n")
        f.write(f"OK={total_ok}  SKIP={total_skip}  FAIL={total_fail}\n")

if __name__ == "__main__":
    main()
