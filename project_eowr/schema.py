"""
Type-safe models for extractor output.
"""

from pydantic import BaseModel, Field, field_validator, ConfigDict
from enum import Enum
from typing import Optional
from pathlib import Path


class Quality(str, Enum):
    """Only these 3 values are ever allowed — nothing else."""
    FULL    = 'full'
    PARTIAL = 'partial'
    FAILED  = 'failed'


class FileType(str, Enum):
    CSV  = 'csv'
    XLSX = 'xlsx'
    DOCX = 'docx'
    PDF  = 'pdf'


class ExtractionResult(BaseModel):
    """
    The single, type-safe return shape for ALL extractors.
    Every extractor (csv, xlsx, docx, pdf) returns exactly this shape.
    """
    file: str
    file_type: FileType
    markdown: str
    quality: Quality
    warnings: list[str] = Field(default_factory=list)

    # Optional, format-specific extras — not every extractor uses these
    rows: Optional[int] = None       # CSV
    sheets: Optional[list[str]] = None   # XLSX
    pages: Optional[int] = None      # PDF

    @field_validator('file')
    @classmethod
    def file_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('file name cannot be empty')
        return v

    @field_validator('markdown')
    @classmethod
    def markdown_matches_quality(cls, v, info):
        # If quality says 'failed', markdown is allowed to be empty.
        # If quality says 'full' or 'partial', markdown MUST have content.
        quality = info.data.get('quality')
        if quality in (Quality.FULL, Quality.PARTIAL) and not v.strip():
            raise ValueError(
                f"quality is '{quality}' but markdown is empty — inconsistent state"
            )
        return v

    model_config = ConfigDict(use_enum_values=True)


class ClassifiedDocument(BaseModel):
    """
    The final, complete record for one processed document:
    extraction result + predicted category, bundled together.

    This is what gets passed downstream to chunking/embedding —
    every chunk inherits the 'category' field as metadata, so the
    LLM can filter to the right document type before searching.
    """
    extraction: ExtractionResult
    category: str
    confidence: float
    low_confidence: bool = False

    model_config = ConfigDict(use_enum_values=True)


