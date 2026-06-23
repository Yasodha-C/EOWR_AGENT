r"""
Builds one combined training dataset from your 188 labeled files.

INPUT:
  1. labels.xlsx with 3 columns: filename, label, filepath
     (filepath = relative path INSIDE your data folder, e.g.
      "well_summary/EOWR_Summary_Campaign.xlsx")
  2. Your data folder (the one containing all the category subfolders)

OUTPUT:
  training_data.csv  — with columns: filename, label, markdown, quality
  
USAGE:
  python build_training_data.py <data path>
  """

import os
import sys
import pandas as pd

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
EXCEL_PATH   = "labels.xlsx"
DATA_FOLDER  = "data"               # used only if no command-line argument given
OUTPUT_PATH  = "training_data.csv"

FILENAME_COL = "filename"
LABEL_COL    = "label"
FILEPATH_COL = "filepath"

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
EXTRACTORS_DIR = os.path.join(PROJECT_ROOT, 'extractors')

sys.path.insert(0, THIS_DIR)            # in case it's already inside extractors/
sys.path.insert(0, EXTRACTORS_DIR)      # in case it's in a sibling folder (e.g. generators/)
sys.path.insert(0, PROJECT_ROOT)        # for schema.py at project root

from extractor_factory import extract

def main():
    data_folder = sys.argv[1] if len(sys.argv) > 1 else DATA_FOLDER
    data_folder = os.path.abspath(data_folder)

    excel_path = os.path.join(THIS_DIR, EXCEL_PATH)
    if not os.path.exists(excel_path):
        # Try the extractors folder as a fallback
        alt_path = os.path.join(EXTRACTORS_DIR, EXCEL_PATH)
        if os.path.exists(alt_path):
            excel_path = alt_path

    print(f"Labels file : {excel_path}")
    print(f"Data folder : {data_folder}\n")

    if not os.path.exists(excel_path):
        print(f"ERROR: Excel file not found: {excel_path}")
        return

    if not os.path.exists(data_folder):
        print(f"ERROR: Data folder not found: {data_folder}")
        print('Tip: pass the full path as an argument, e.g.')
        print(r'  python build_training_data.py "D:\SwarmLens\EOWR\eowr_agent\data"')
        return

    df_labels = pd.read_excel(excel_path)

    required_cols = {FILENAME_COL, LABEL_COL, FILEPATH_COL}
    if not required_cols.issubset(df_labels.columns):
        print(f"ERROR: Excel must have columns: {required_cols}")
        print(f"Found columns: {list(df_labels.columns)}")
        return

    print(f"Loaded {len(df_labels)} filename-label pairs\n")

    results = []
    failed  = []

    for idx, row in df_labels.iterrows():
        filename = str(row[FILENAME_COL]).strip()
        label    = str(row[LABEL_COL]).strip()
        filepath = str(row[FILEPATH_COL]).strip()

        full_path = os.path.join(data_folder, filepath)

        if not os.path.exists(full_path):
            print(f"  [MISSING] {filepath}")
            failed.append((filename, label, filepath, "file not found"))
            continue

        result = extract(full_path)

        if result.quality == 'failed':
            print(f"  [FAILED]  {filepath}  -- {result.warnings}")
            failed.append((filename, label, filepath, str(result.warnings)))
            continue

        status = '[OK]     ' if result.quality == 'full' else '[PARTIAL]'
        print(f"  {status} {filepath}  -> {label}")

        results.append({
            'filename': filename,
            'label': label,
            'markdown': result.markdown,
            'quality': result.quality
        })

    df_out = pd.DataFrame(results)
    output_path = os.path.join(THIS_DIR, OUTPUT_PATH)
    df_out.to_csv(output_path, index=False)

    print(f"\n{'='*55}")
    print(f"Done. {len(results)}/{len(df_labels)} files extracted successfully.")
    print(f"Saved to: {output_path}")

    if failed:
        print(f"\n{len(failed)} files failed or were missing:")
        for fname, lbl, fp, reason in failed:
            print(f"  - {fp} ({lbl}): {reason}")

        fail_path = output_path.replace('.csv', '_failed.csv')
        pd.DataFrame(failed, columns=['filename', 'label', 'filepath', 'reason']).to_csv(fail_path, index=False)
        print(f"\nFailed list saved to: {fail_path}")

    print(f"{'='*55}")
    print(f"\nNext step: upload '{OUTPUT_PATH}' back to Claude to train the classifier.")


if __name__ == '__main__':
    main()
