r"""
Auto-generates labels.xlsx from a folder structure where each
top-level subfolder name IS the category label.

Example structure expected:
    data/
      drilling/
        file1.xlsx
        daily_reports/
          file2.pdf          <- still labeled 'drilling' (nested under it)
      npt_incidents/
        file3.xlsx

USAGE (recommended — pass full path directly, avoids path confusion):
    python generate_labels_from_folders.py "D:\SwarmLens\EOWR\eowr_agent\data"

USAGE (alternative — no argument, uses DATA_FOLDER below):
    python generate_labels_from_folders.py
"""

import os
import sys
import pandas as pd

# ─────────────────────────────────────────────
# CONFIG — only used if you don't pass a path as argument
# ─────────────────────────────────────────────
DATA_FOLDER = "data"
OUTPUT_PATH = "labels.xlsx"

ALLOWED_EXTENSIONS = {'.pdf', '.xlsx', '.xls', '.docx', '.doc', '.csv'}
IGNORE_FILES = {'.gitkeep', '.DS_Store', 'Thumbs.db'}


def main():
    # Priority: command-line argument > DATA_FOLDER variable
    if len(sys.argv) > 1:
        data_folder = sys.argv[1]
    else:
        data_folder = DATA_FOLDER

    data_folder = os.path.abspath(data_folder)   # resolve to full path, avoid confusion

    print(f"Looking for data in: {data_folder}")

    if not os.path.exists(data_folder):
        print(f"\nERROR: Folder not found: {data_folder}")
        print("\nTip: run this script with the FULL path to your data folder, e.g.:")
        print(r'  python generate_labels_from_folders.py "D:\SwarmLens\EOWR\eowr_agent\data"')
        return

    rows = []
    skipped = []

    for root, dirs, files in os.walk(data_folder):
        for fname in files:
            if fname in IGNORE_FILES:
                continue

            ext = os.path.splitext(fname)[1].lower()
            if ext not in ALLOWED_EXTENSIONS:
                skipped.append(fname)
                continue

            full_path = os.path.join(root, fname)
            rel_path  = os.path.relpath(full_path, data_folder)

            # Label = TOP-LEVEL folder name right under data_folder
            parts = rel_path.split(os.sep)
            label = parts[0]

            rows.append({
                'filename': fname,
                'label': label,
                'filepath': rel_path.replace('\\', '/')
            })

    if not rows:
        print(f"No supported files found in {data_folder}")
        return

    df = pd.DataFrame(rows)
    df = df.sort_values(['label', 'filename']).reset_index(drop=True)

    # Save labels.xlsx in the SAME folder as this script, not wherever you ran it from
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, OUTPUT_PATH)
    df.to_excel(output_path, index=False)

    print(f"\n{'='*55}")
    print(f"Found {len(df)} files across {df['label'].nunique()} labels.")
    print(f"Saved to: {output_path}")
    print(f"{'='*55}\n")

    print("Files per label:")
    print(df['label'].value_counts().to_string())

    if skipped:
        print(f"\n{len(skipped)} files skipped (unsupported type):")
        for s in skipped[:10]:
            print(f"  - {s}")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped)-10} more")

    print(f"\nNext step: open {output_path} — labels are already filled in.")


if __name__ == '__main__':
    main()
