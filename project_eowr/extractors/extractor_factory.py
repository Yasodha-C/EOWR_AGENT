import sys
import os

# Make sure the project root (one level up) is on the import path,
# so 'schema.py' can be found no matter how this script is run.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pathlib import Path
from csv_extractor  import extract_csv_to_markdown
from xlsx_extractor import extract_xlsx_to_markdown
from docx_extractor import extract_docx_to_markdown
from pdf_extractor  import extract_pdf_to_markdown
from schema import ExtractionResult, FileType, Quality


EXTRACTORS = {
    '.csv'  : (extract_csv_to_markdown,  FileType.CSV),
    '.xlsx' : (extract_xlsx_to_markdown, FileType.XLSX),
    '.xls'  : (extract_xlsx_to_markdown, FileType.XLSX),
    '.docx' : (extract_docx_to_markdown, FileType.DOCX),
    '.doc'  : (extract_docx_to_markdown, FileType.DOCX),
    '.pdf'  : (extract_pdf_to_markdown,  FileType.PDF),
}


def extract(file_path: str) -> ExtractionResult:
    """
    Single entry point for all file types.
    Always returns a validated ExtractionResult — never a raw dict.
    """
    path = Path(file_path)
    ext  = path.suffix.lower()

    if not path.exists():
        return ExtractionResult(
            file=path.name,
            file_type=FileType.PDF,
            markdown='',
            quality=Quality.FAILED,
            warnings=[f'File not found: {file_path}']
        )

    if ext not in EXTRACTORS:
        return ExtractionResult(
            file=path.name,
            file_type=FileType.PDF,
            markdown='',
            quality=Quality.FAILED,
            warnings=[f'Unsupported file type: {ext}. Supported: {list(EXTRACTORS.keys())}']
        )

    extractor_fn, file_type = EXTRACTORS[ext]
    raw_result = extractor_fn(file_path)

    return ExtractionResult(
        file      = raw_result.get('file', path.name),
        file_type = file_type,
        markdown  = raw_result.get('markdown', ''),
        quality   = raw_result.get('quality', 'failed'),
        warnings  = raw_result.get('warnings', []),
        rows      = raw_result.get('rows'),
        sheets    = raw_result.get('sheets'),
        pages     = raw_result.get('pages'),
    )


if __name__ == '__main__':
    import sys as _sys
    files = _sys.argv[1:] if len(_sys.argv) > 1 else [
        os.path.join(THIS_DIR, 'test_wells.csv'),
        os.path.join(THIS_DIR, 'test_pressure.xlsx'),
        os.path.join(THIS_DIR, 'test_completion_report.docx'),
        os.path.join(THIS_DIR, 'DSEC-AD-1X_Casing_and_Wellhead_Pressure_Tests.pdf'),
    ]

    for f in files:
        result = extract(f)
        print(f"File    : {result.file}")
        print(f"Type    : {result.file_type}")
        print(f"Quality : {result.quality}")
        if result.warnings:
            print(f"Warnings: {result.warnings}")
        print(result.markdown[:200])
        print('-' * 60)
