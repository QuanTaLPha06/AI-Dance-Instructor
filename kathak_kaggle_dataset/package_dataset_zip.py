"""
Kathak Dataset Packaging Helper Script for Kaggle Upload
Packages processed .npy sample files into compressed .zip archives for fast upload to Kaggle.
"""

import os
import sys
import glob
import zipfile
import time

SOURCE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "kathak_dataset", "processed_samples"))
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

def package_dataset(max_files=None, samples_per_zip=500000):
    print("==========================================================")
    print("Kathak Kaggle Dataset Packaging Utility")
    print("==========================================================")
    print(f"Source Directory : {SOURCE_DIR}")
    print(f"Target Directory : {OUTPUT_DIR}")

    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Source directory does not exist: {SOURCE_DIR}")
        return

    sample_files = [f for f in os.listdir(SOURCE_DIR) if f.endswith(".npy")]
    total_found = len(sample_files)
    print(f"Total `.npy` sample files found: {total_found:,}")

    if max_files and max_files < total_found:
        sample_files = sample_files[:max_files]
        print(f"Packaging subset of {len(sample_files):,} sample files.")

    num_parts = (len(sample_files) + samples_per_zip - 1) // samples_per_zip

    start_time = time.time()
    for part in range(num_parts):
        chunk = sample_files[part * samples_per_zip : (part + 1) * samples_per_zip]
        zip_filename = f"kathak_processed_samples_part{part+1}.zip" if num_parts > 1 else "kathak_processed_samples.zip"
        target_zip = os.path.join(OUTPUT_DIR, zip_filename)

        print(f"\nCreating archive [{part+1}/{num_parts}]: {zip_filename} ({len(chunk):,} files)...")
        with zipfile.ZipFile(target_zip, 'w', compression=zipfile.ZIP_STORED) as zip_out:
            for i, filename in enumerate(chunk):
                filepath = os.path.join(SOURCE_DIR, filename)
                arcname = os.path.join("processed_samples", filename)
                zip_out.write(filepath, arcname=arcname)

                if (i + 1) % 50000 == 0 or (i + 1) == len(chunk):
                    print(f"   Progress: {i+1:,}/{len(chunk):,} files added...", flush=True)

        size_mb = os.path.getsize(target_zip) / (1024 * 1024)
        print(f"Part {part+1} created successfully: {zip_filename} ({size_mb:.2f} MB)")

    elapsed = time.time() - start_time
    print("\n" + "=" * 58)
    print(f"Packaging Complete in {elapsed/60:.2f} minutes!")
    print(f"Zipped files saved to: {OUTPUT_DIR}")
    print("==========================================================")

if __name__ == "__main__":
    max_f = int(sys.argv[1]) if len(sys.argv) > 1 else None
    package_dataset(max_files=max_f)
