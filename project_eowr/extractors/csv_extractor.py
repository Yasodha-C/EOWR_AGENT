import pandas as pd
from pathlib import Path


def clean_dataframe(df):
    # Step 1: strip whitespace from all string cells
    # pandas 3.x: use map() instead of applymap()
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

    # Step 2: replace whitespace-only strings with NaN
    df = df.replace(r'^\s*$', float('nan'), regex=True)

    # Step 3: drop rows where ALL values are NaN (Option A)
    df = df.dropna(how='all')

    # Step 4: reset index
    df = df.reset_index(drop=True)

    return df


def detect_encoding(file_path):
    for enc in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']:
        try:
            pd.read_csv(file_path, encoding=enc, nrows=5)
            return enc
        except Exception:
            continue
    return 'utf-8'


def detect_delimiter(file_path, encoding):
    with open(file_path, encoding=encoding, errors='replace') as f:
        first_line = f.readline()
    for delim in [',', ';', '\t', '|']:
        if delim in first_line:
            return delim
    return ','


def extract_csv_to_markdown(file_path: str) -> dict:
    path = Path(file_path)
    warnings = []

    try:
        encoding  = detect_encoding(file_path)
        delimiter = detect_delimiter(file_path, encoding)

        df = pd.read_csv(file_path, encoding=encoding, sep=delimiter, dtype=str)

        original_rows = len(df)
        df = clean_dataframe(df)
        removed = original_rows - len(df)

        if removed > 0:
            warnings.append(f"Removed {removed} fully empty rows")

        partial = df.isnull().any(axis=1).sum()
        if partial > 0:
            warnings.append(f"{partial} rows have some missing values (kept)")

        markdown = df.to_markdown(index=False)

        quality = ('failed'  if not markdown.strip() else
                   'partial' if len(warnings) > 2 else
                   'full')

        return {
            'file'    : path.name,
            'rows'    : len(df),
            'columns' : list(df.columns),
            'markdown': markdown,
            'quality' : quality,
            'warnings': warnings
        }

    except Exception as e:
        return {
            'file'    : path.name,
            'rows'    : 0,
            'columns' : [],
            'markdown': '',
            'quality' : 'failed',
            'warnings': [f'Cannot read file: {e}']
        }


if __name__ == '__main__':
    result = extract_csv_to_markdown('/home/claude/test_wells.csv')
    print(f"File    : {result['file']}")
    print(f"Rows    : {result['rows']}")
    print(f"Columns : {result['columns']}")
    print(f"Quality : {result['quality']}")
    if result['warnings']:
        print(f"Warnings: {result['warnings']}")
    print()
    print('=' * 60)
    print(result['markdown'])
