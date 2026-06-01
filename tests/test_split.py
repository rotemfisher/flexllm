"""
tests/test_split.py — Validate that pre-splitting multi-column PDFs improves
Docling extraction quality.

All tests run always. Docling quality tests auto-skip when source PDFs are absent
(e.g. in CI where data/model/ is not checked in).
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "etl"))

# Books with multi-column layout that must be pre-split before Docling.
SPLIT_CASES = [
    {
        "label":   "Physiology of Sport and Exercise",
        "path":    ROOT / "data/model/physiology/physiology-of-sport-and-exercise_compress.pdf",
        "n_parts": 2,
        "sample_pages": [50, 100],   # 0-indexed
    },
    {
        "label":   "Science of Running",
        "path":    ROOT / "data/model/running/science-of-running-1nbsped-9780241394519_compress.pdf",
        "n_parts": 3,
        "sample_pages": [30, 60],
    },
]


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_excerpt(src: Path, page_indices: list[int], dest: Path) -> None:
    """Write a subset of pages from src PDF to dest."""
    from pypdf import PdfReader, PdfWriter
    reader = PdfReader(str(src))
    writer = PdfWriter()
    for i in page_indices:
        writer.add_page(reader.pages[i])
    with open(str(dest), "wb") as f:
        writer.write(f)


def _split_excerpt(src: Path, page_indices: list[int], n_parts: int, dest: Path) -> None:
    """Write pre-split version of the same excerpt pages to dest."""
    from pypdf import PdfReader, PdfWriter
    from pypdf.generic import RectangleObject
    reader = PdfReader(str(src))
    part_writers = [PdfWriter() for _ in range(n_parts)]
    for pg_idx in page_indices:
        page = reader.pages[pg_idx]
        w = float(page.mediabox.width)
        h = float(page.mediabox.height)
        strip_w = w / n_parts
        for i, pw in enumerate(part_writers):
            pw.add_page(page)
            added = pw.pages[-1]
            added.cropbox  = RectangleObject((i * strip_w, 0, (i + 1) * strip_w, h))
            added.mediabox = RectangleObject((i * strip_w, 0, (i + 1) * strip_w, h))
    final = PdfWriter()
    for pg_pos in range(len(page_indices)):
        for pw in part_writers:
            final.add_page(pw.pages[pg_pos])
    with open(str(dest), "wb") as f:
        final.write(f)


def _docling_markdown(pdf_path: Path) -> str:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions, AcceleratorOptions, AcceleratorDevice,
    )
    opts = PdfPipelineOptions()
    opts.do_ocr = False
    opts.accelerator_options = AcceleratorOptions(device=AcceleratorDevice.CPU)
    conv = DocumentConverter(format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=opts, backend=PyPdfiumDocumentBackend)
    })
    return conv.convert(str(pdf_path)).document.export_to_markdown()


# ── structural tests (always run, no Docling) ─────────────────────────────────

@pytest.mark.parametrize("case", SPLIT_CASES, ids=[c["label"] for c in SPLIT_CASES])
def test_split_page_count(tmp_path, case):
    """Split PDF page count must be between original (all pass-through) and n_parts×original (all split)."""
    pytest.importorskip("pypdf")
    src = case["path"]
    if not src.exists():
        pytest.skip(f"Source PDF not found: {src.name}")

    from etl.embed_books import split_pdf_pages
    split_path = split_pdf_pages(src, case["n_parts"])

    from pypdf import PdfReader
    original_pages = len(PdfReader(str(src)).pages)
    split_pages    = len(PdfReader(str(split_path)).pages)
    assert original_pages < split_pages <= original_pages * case["n_parts"], (
        f"Expected between {original_pages + 1} and {original_pages * case['n_parts']} pages, got {split_pages}"
    )


@pytest.mark.parametrize("case", SPLIT_CASES, ids=[c["label"] for c in SPLIT_CASES])
def test_split_page_dimensions(tmp_path, case):
    """
    Every page in the split PDF must have:
    - height matching the original (same height)
    - width ≤ original width + 2 (no page is wider than the source)
    And at least one page must be narrower than the original (proving actual splits happened).
    """
    pytest.importorskip("pypdf")
    src = case["path"]
    if not src.exists():
        pytest.skip(f"Source PDF not found: {src.name}")

    from etl.embed_books import split_pdf_pages
    split_path = split_pdf_pages(src, case["n_parts"])

    import pypdfium2 as pdfium
    orig_doc  = pdfium.PdfDocument(str(src))
    split_doc = pdfium.PdfDocument(str(split_path))

    ref_w = orig_doc[case["sample_pages"][0]].get_width()
    ref_h = orig_doc[case["sample_pages"][0]].get_height()

    found_strip = False
    for i in range(len(split_doc)):
        pg = split_doc[i]
        assert abs(pg.get_height() - ref_h) < 5, (
            f"Split page {i}: height {pg.get_height():.1f} ≠ original {ref_h:.1f}"
        )
        assert pg.get_width() <= ref_w + 5, (
            f"Split page {i}: width {pg.get_width():.1f} exceeds original {ref_w:.1f}"
        )
        if pg.get_width() < ref_w - 5:
            found_strip = True

    assert found_strip, "No split strips found — nothing was actually split"


@pytest.mark.parametrize("case", SPLIT_CASES, ids=[c["label"] for c in SPLIT_CASES])
def test_split_strips_cover_full_width(tmp_path, case):
    """
    For every group of n_parts consecutive narrow pages in the split PDF,
    their widths must sum to the original page width (no gaps/overlaps).
    Pass-through (full-width) pages must retain the original width.
    """
    pytest.importorskip("pypdf")
    src = case["path"]
    if not src.exists():
        pytest.skip(f"Source PDF not found: {src.name}")

    from etl.embed_books import split_pdf_pages
    from pypdf import PdfReader
    split_path = split_pdf_pages(src, case["n_parts"])

    orig_reader  = PdfReader(str(src))
    split_reader = PdfReader(str(split_path))

    ref_w = float(orig_reader.pages[0].mediabox.width)
    strip_threshold = ref_w * 0.85  # narrower than this → it's a strip, not pass-through

    widths = [float(split_reader.pages[i].mediabox.width) for i in range(len(split_reader.pages))]

    groups_checked = 0
    i = 0
    while i < len(widths):
        w = widths[i]
        if w < strip_threshold:
            # Collect the next n_parts consecutive pages as one strip group
            group = widths[i : i + case["n_parts"]]
            assert len(group) == case["n_parts"], (
                f"Incomplete strip group starting at split index {i}: got {len(group)} pages"
            )
            total = sum(group)
            assert abs(total - ref_w) < 5, (
                f"Strip group at split index {i}: widths {group} sum to {total:.1f}, expected ~{ref_w:.1f}"
            )
            groups_checked += 1
            i += case["n_parts"]
        else:
            assert abs(w - ref_w) < 5, (
                f"Pass-through page at split index {i}: width {w:.1f} ≠ original {ref_w:.1f}"
            )
            i += 1

    assert groups_checked > 0, "No strip groups found — nothing was actually split"


# ── Docling quality tests (slow, opt-in) ──────────────────────────────────────

@pytest.mark.parametrize("case", SPLIT_CASES, ids=[c["label"] for c in SPLIT_CASES])
def test_docling_split_extracts_more_content(tmp_path, case):
    """
    Docling must extract substantially more text from the pre-split PDF than
    from the raw multi-column PDF.

    Expected improvement: split version yields ≥1.4× the character count,
    meaning Docling was silently dropping content from the wide multi-column page.
    """

    src = case["path"]
    if not src.exists():
        pytest.skip(f"Source PDF not found: {src.name}")

    pages = case["sample_pages"][:2]  # limit to 2 pages to keep it fast
    n     = case["n_parts"]

    before_pdf = tmp_path / "before.pdf"
    after_pdf  = tmp_path / "after.pdf"
    _make_excerpt(src,  pages, before_pdf)
    _split_excerpt(src, pages, n, after_pdf)

    md_before = _docling_markdown(before_pdf)
    md_after  = _docling_markdown(after_pdf)

    ratio = len(md_after) / max(len(md_before), 1)

    print(f"\n  {case['label']}")
    print(f"  Before : {len(md_before):,} chars")
    print(f"  After  : {len(md_after):,} chars")
    print(f"  Ratio  : {ratio:.2f}×")
    print(f"\n  --- BEFORE (first 400 chars) ---")
    print(f"  {md_before[:400]}")
    print(f"\n  --- AFTER  (first 400 chars) ---")
    print(f"  {md_after[:400]}")

    assert ratio >= 1.4, (
        f"Split should yield ≥1.4× more content but got {ratio:.2f}× "
        f"({len(md_before)} → {len(md_after)} chars)"
    )
