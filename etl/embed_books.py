"""
ETL: Parse coaching PDFs with Docling → chunk (markdown-aware) → embed (local) → Qdrant.

Run once:
    python etl/embed_books.py

Re-run is safe — skips books already embedded (checks by book key).
No API key needed: embeddings run locally via sentence-transformers + fastembed.
"""

import logging
import re
import uuid
from pathlib import Path

from langsmith import traceable

import pypdfium2 as pdfium
import tiktoken
import torch
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from fastembed import SparseTextEmbedding
from pypdf import PdfReader, PdfWriter
from pypdf.generic import RectangleObject
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

BOOKS = [
    # ── Running ──────────────────────────────────────────────────────────────────
    {
        "key":      "daniels",
        "title":    "Daniels Running Formula",
        "category": "running",
        "path":     "data/model/running/dokumen.pub_daniels-running-formula-4.pdf",
    },
    {
        "key":      "lore",
        "title":    "Lore of Running",
        "category": "running",
        "path":     "data/model/running/lore-of-running_compress.pdf",
    },
    {
        "key":        "science_run",
        "title":      "Science of Running",
        "category":   "running",
        "path":       "data/model/running/science-of-running-1nbsped-9780241394519_compress.pdf",
        "page_split": 3,
    },
    # ── Fitness ───────────────────────────────────────────────────────────────────
    {
        "key":      "strength",
        "title":    "Science and Practice of Strength Training",
        "category": "fitness",
        "path":     "data/model/fitness/science-and-practice-of-strength-training-3nbsped-1492592005-9781492592006_compress.pdf",
    },
    {
        "key":      "periodization",
        "title":    "Periodization: Theory and Methodology of Training",
        "category": "fitness",
        "path":     "data/model/fitness/periodization-theory-and-methodology-of-training-sixth-edition-9781492544807-1492544809-9781492544814_compress.pdf",
    },
    {
        "key":      "nsca",
        "title":    "Essentials of Strength Training and Conditioning",
        "category": "fitness",
        "path":     "data/model/fitness/essentials-of-strength-training-and-conditioning-e-book-edition_compress.pdf",
    },
    # ── Nutrition ─────────────────────────────────────────────────────────────────
    {
        "key":      "sport_nutrition",
        "title":    "Sport Nutrition",
        "category": "nutrition",
        "path":     "data/model/nutrition/sport-nutrition_compress.pdf",
    },
    {
        "key":      "clinical_sports_nutrition",
        "title":    "Clinical Sports Nutrition",
        "category": "nutrition",
        "path":     "data/model/nutrition/Share Clinical_Sports_Nutrition_by_Louise_Burke,_Vicki_Deakin_Mss_Telegram.pdf",
    },
    # ── Physiology ────────────────────────────────────────────────────────────────
    {
        "key":        "physiology_sport",
        "title":      "Physiology of Sport and Exercise",
        "category":   "physiology",
        "path":       "data/model/physiology/physiology-of-sport-and-exercise_compress.pdf",
        "page_split": 2,
    },
    {
        "key":        "clinical_sports_medicine",
        "title":      "Clinical Sports Medicine",
        "category":   "physiology",
        "path":       "data/model/physiology/Clinical Sports Medicine Khan & Khanum -  3rd Ed.pdf",
        "page_split": 2,
    },
]

# Directories whose every PDF is ingested automatically.
# Each PDF becomes its own entry; key = stem slug, title = stem cleaned up.
ARTICLE_DIRS = [
    {"dir": "data/model/nutrition/Taylor_&_Francis_Articles", "category": "nutrition"},
]

EMBED_MODEL    = "BAAI/bge-large-en-v1.5"   # 1024 dims, free, production-grade
SPARSE_MODEL   = "Qdrant/bm25"              # BM25 sparse vectors for hybrid search
COLLECTION_NAME = "coaching_books"
CHUNK_TOKENS   = 600
OVERLAP_TOKENS = 80
BGE_MAX_TOKENS = 512   # BGE-large-en-v1.5 hard context limit; silently truncates beyond this

# Layout quality thresholds (used to decide whether column-split fallback is needed)
_LAYOUT_LONG_PARA_CHARS = 800   # paragraphs longer than this chars are suspicious
_LAYOUT_BAD_PARA_RATIO  = 0.30  # if >30% of text paragraphs are suspiciously long → broken

PROJECT_ROOT = Path(__file__).parent.parent
QDRANT_PATH  = PROJECT_ROOT / "data" / "qdrant_db"

# ── Tokenizer ─────────────────────────────────────────────────────────────────

enc = tiktoken.get_encoding("cl100k_base") 

def token_len(text: str) -> int:
    return len(enc.encode(text))


# ── Markdown-aware chunking ───────────────────────────────────────────────────

def _sub_chunk(text: str, max_tokens: int, overlap: int) -> list[str]:
    """Split long text into overlapping chunks, snapping cuts to word boundaries."""
    tokens = enc.encode(text)
    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        raw = enc.decode(tokens[start:end])

        # Snap cut to nearest whitespace in the last ~15% of the window to avoid mid-word splits
        if end < len(tokens):
            snap_from = max(1, int(len(raw) * 0.85))
            last_ws = max(raw.rfind("\n", snap_from), raw.rfind(" ", snap_from))
            if last_ws > snap_from:
                raw = raw[:last_ws]

        chunk_text = raw.strip()
        if chunk_text:
            chunks.append(chunk_text)
        if end >= len(tokens):
            break
        start += max_tokens - overlap

    return [c for c in chunks if c.strip()]


def _split_table_rows(table_text: str, heading: str, max_tokens: int) -> list[dict]:
    """
    Split a large markdown table by rows so no chunk exceeds max_tokens.
    Each sub-chunk repeats the header rows so the table remains self-contained.
    Falls back to a single chunk if the structure cannot be parsed cleanly.
    """
    lines = table_text.splitlines()
    header_rows: list[str] = []
    data_rows: list[str] = []
    separator_seen = False
    for line in lines:
        if not separator_seen:
            header_rows.append(line)
            if re.match(r"^\|[-| :]+\|?\s*$", line.strip()):
                separator_seen = True
        else:
            data_rows.append(line)

    if not data_rows or not separator_seen:
        full = (f"{heading}\n{table_text}" if heading else table_text).strip()
        return [{"text": full, "chunk_type": "table", "section": heading}]

    header_text = "\n".join(header_rows)
    results: list[dict] = []
    current_rows: list[str] = []

    for row in data_rows:
        candidate_body = "\n".join([header_text] + current_rows + [row])
        candidate_full = (f"{heading}\n{candidate_body}" if heading else candidate_body).strip()
        if token_len(candidate_full) > max_tokens and current_rows:
            body = "\n".join([header_text] + current_rows)
            full = (f"{heading}\n{body}" if heading else body).strip()
            results.append({"text": full, "chunk_type": "table", "section": heading})
            current_rows = [row]
        else:
            current_rows.append(row)

    if current_rows:
        body = "\n".join([header_text] + current_rows)
        full = (f"{heading}\n{body}" if heading else body).strip()
        results.append({"text": full, "chunk_type": "table", "section": heading})

    return results


_IMAGE_PLACEHOLDER = re.compile(r"<!--\s*image\s*-->", re.IGNORECASE)
_ARTIFACT_PATTERN  = re.compile(r"\d{4}-\d{2}-\d{2}|\w+\.indd \d+|^\s*\d+\s*$", re.MULTILINE)
def _is_meaningful(text: str, min_words: int = 8) -> bool:
    """Return False for chunks that are mostly image placeholders or PDF artifacts."""
    cleaned = _IMAGE_PLACEHOLDER.sub("", text).strip()
    cleaned = _ARTIFACT_PATTERN.sub("", cleaned).strip()
    words = len(cleaned.split())
    return words >= min_words


def chunk_markdown(markdown: str, max_tokens: int = CHUNK_TOKENS,
                   overlap: int = OVERLAP_TOKENS) -> list[dict]:
    """
    Split Docling markdown output into smart chunks.

    Strategy:
    1. Split at ## / ### headers — natural section boundaries.
    2. Markdown tables → single chunk, type="table" (never split).
    3. Long text sections → sub-chunk with overlap.

    Returns list of {"text": str, "chunk_type": str, "section": str}
    """
    # Split the markdown at every heading line (## or ###)
    # Keep the heading attached to the section that follows it
    header_pattern = re.compile(r"^(#{1,3} .+)$", re.MULTILINE)
    parts = header_pattern.split(markdown)

    # parts alternates between: non-heading text, heading, non-heading text, ...
    # Pair each heading with the text that follows it
    sections: list[tuple[str, str]] = []  # (heading, body)
    current_heading = ""
    for part in parts:
        if header_pattern.match(part.strip()):
            current_heading = part.strip()
        else:
            if part.strip():
                sections.append((current_heading, part.strip()))

    results: list[dict] = []

    for heading, body in sections:
        # Split body into table blocks vs text blocks
        lines = body.splitlines()
        buffer_lines: list[str] = []
        in_table = False

        def flush_text():
            text = "\n".join(buffer_lines).strip()
            if not text:
                return
            full = (f"{heading}\n{text}" if heading else text).strip()
            if token_len(full) <= max_tokens:
                results.append({"text": full, "chunk_type": "text", "section": heading})
            else:
                for sub in _sub_chunk(full, max_tokens, overlap):
                    results.append({"text": sub, "chunk_type": "text", "section": heading})

        for line in lines:
            is_pipe = line.strip().startswith("|")
            if is_pipe and not in_table:
                # Entering a table — flush accumulated text first
                flush_text()
                buffer_lines = [line]
                in_table = True
            elif not is_pipe and in_table:
                # Leaving a table — split by rows if it exceeds the embedding model's context
                table_text = "\n".join(buffer_lines).strip()
                if table_text:
                    results.extend(_split_table_rows(table_text, heading, BGE_MAX_TOKENS))
                buffer_lines = [line]
                in_table = False
            else:
                buffer_lines.append(line)

        # Flush whatever remains
        if in_table:
            table_text = "\n".join(buffer_lines).strip()
            if table_text:
                results.extend(_split_table_rows(table_text, heading, BGE_MAX_TOKENS))
        else:
            flush_text()

    return [r for r in results if _is_meaningful(r["text"])]


# ── IDs ───────────────────────────────────────────────────────────────────────

def stable_id(book_key: str, chunk_idx: int) -> str:
    """Human-readable string ID stored in the Qdrant payload for debugging."""
    return f"{book_key}_c{chunk_idx:05d}"


def stable_qdrant_id(book_key: str, chunk_idx: int) -> str:
    """Deterministic UUID derived from the string ID — Qdrant requires UUID or int64."""
    return str(uuid.uuid5(uuid.NAMESPACE_OID, stable_id(book_key, chunk_idx)))


# ── PDF page splitter ─────────────────────────────────────────────────────────

_TMP_DIR = PROJECT_ROOT / "data" / "_presplit"


_GUTTER_THRESHOLD = 0.25  # gutter region must be below this fraction of page max density


def _detect_page_gutters(page, n_parts: int, bins: int = 400) -> list[float] | None:
    """
    Detect column gutter positions for a single PDF page.

    Returns a list of n_parts-1 gutter fractions (as proportions of page width)
    if the page has a clear n_parts-column layout, or None if no clear gutters
    are found (single-column page or near-empty page).

    A gutter is detected when the minimum text density in the expected
    column-gap region falls below _GUTTER_THRESHOLD * page_max_density.
    """
    w = page.get_width()
    if w == 0:
        return None

    density: list[int] = [0] * bins
    tp = page.get_textpage()
    if tp.count_chars() < 30:
        return None

    for i in range(tp.count_chars()):
        box = tp.get_charbox(i)
        if box[2] <= box[0]:
            continue
        x_l = max(0,        int(box[0] / w * bins))
        x_r = min(bins - 1, int(box[2] / w * bins))
        for b in range(x_l, x_r + 1):
            density[b] += 1

    max_d = max(density)
    if max_d == 0:
        return None
    norm = [d / max_d for d in density]

    gutters: list[float] = []
    for col in range(1, n_parts):
        expected = col / n_parts
        lo = max(0,        int((expected - 0.15) * bins))
        hi = min(bins - 1, int((expected + 0.15) * bins))
        region = norm[lo:hi]
        if not region:
            return None
        min_val = min(region)
        if min_val > _GUTTER_THRESHOLD:
            return None  # No clear whitespace gap → not n_parts columns
        local_min_idx = region.index(min_val)
        gutters.append((lo + local_min_idx) / bins)

    return sorted(gutters)


def split_pdf_pages(src_path: Path, n_parts: int) -> Path:
    """
    Split each PDF page into n_parts vertical strips, but only when the page
    actually has n_parts columns (detected per-page via character-density analysis).
    Single-column pages (chapter openers, full-width figures, etc.) pass through
    unchanged so text is never cut through mid-column.

    The result is cached in data/_presplit/. Cache is invalidated when source mtime changes.
    """
    _TMP_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = _TMP_DIR / f"{src_path.stem}_split{n_parts}.pdf"

    if tmp_path.exists() and tmp_path.stat().st_mtime >= src_path.stat().st_mtime:
        logger.info("  [cache] using pre-split PDF: %s", tmp_path.name)
        return tmp_path

    logger.info("  Analysing per-page layout for %s …", src_path.name)
    pdfium_doc = pdfium.PdfDocument(str(src_path))
    reader     = PdfReader(str(src_path))
    n_pages    = len(reader.pages)

    # First pass: decide per-page whether to split and where.
    page_gutters: list[list[float] | None] = [
        _detect_page_gutters(pdfium_doc[i], n_parts) for i in range(n_pages)
    ]
    n_split = sum(1 for g in page_gutters if g is not None)
    logger.info("  %d/%d pages detected as %d-column (rest pass through)", n_split, n_pages, n_parts)

    # Second pass: build the output PDF.
    # Split pages need n_parts independent copies (one per strip); we use one
    # PdfWriter per strip. Passthrough pages go into a separate writer.
    pass_writer  = PdfWriter()
    part_writers = [PdfWriter() for _ in range(n_parts)]

    # page_info[i] = ("pass", idx_in_pass_writer) or ("split", idx_in_part_writers)
    page_info: list[tuple[str, int]] = []
    pass_counter  = 0
    split_counter = 0

    for pg_idx in range(n_pages):
        page    = reader.pages[pg_idx]
        gutters = page_gutters[pg_idx]

        if gutters is None:
            pass_writer.add_page(page)
            page_info.append(("pass", pass_counter))
            pass_counter += 1
        else:
            w = float(page.mediabox.width)
            h = float(page.mediabox.height)
            boundaries = [0.0] + gutters + [1.0]
            for i, pw in enumerate(part_writers):
                pw.add_page(page)
                added = pw.pages[-1]
                added.cropbox  = RectangleObject((boundaries[i] * w, 0, boundaries[i + 1] * w, h))
                added.mediabox = RectangleObject((boundaries[i] * w, 0, boundaries[i + 1] * w, h))
            page_info.append(("split", split_counter))
            split_counter += 1

    # Interleave: passthrough → 1 output page; split → n_parts output pages.
    final = PdfWriter()
    for kind, idx in page_info:
        if kind == "pass":
            final.add_page(pass_writer.pages[idx])
        else:
            for pw in part_writers:
                final.add_page(pw.pages[idx])

    total_out = pass_counter + split_counter * n_parts
    with open(str(tmp_path), "wb") as f:
        final.write(f)
    logger.info("  Pre-split done → %d pages saved to %s", total_out, tmp_path.name)
    return tmp_path


def _slug(name: str) -> str:
    """Convert a filename stem to a safe qdrantid key."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:60]


def books_from_dir(entry: dict) -> list[dict]:
    """Return a BOOKS-style entry for every PDF in the given directory."""
    directory = PROJECT_ROOT / entry["dir"]
    if not directory.is_dir():
        return []
    results = []
    for pdf in sorted(directory.glob("*.pdf")):
        stem = pdf.stem
        results.append({
            "key":      _slug(stem),
            "title":    stem,
            "category": entry["category"],
            "path":     str(pdf.relative_to(PROJECT_ROOT)),
        })
    return results


# ── OCR detection ─────────────────────────────────────────────────────────────

def _check_ocr_needed(markdown: str, pdf_path: Path,
                      min_chars_per_page: int = 150) -> bool:
    """Return True if extracted text is suspiciously sparse (likely a scanned PDF)."""
    try:
        import pypdfium2 as pdfium
        doc = pdfium.PdfDocument(str(pdf_path))
        n_pages = len(doc)
        doc.close()
    except Exception:
        return False
    if n_pages == 0:
        return False
    return len(markdown.strip()) / n_pages < min_chars_per_page


def _layout_score(markdown: str, pdf_path: Path) -> float:
    """
    Layout quality score in [0.0, 1.0]. Higher = better.

    Primary signal: the fraction of text paragraphs that are NOT suspiciously
    long. Merged columns inflate paragraph length to roughly 2× normal, so a
    high fraction of oversized paragraphs indicates broken column parsing.
    Returns 0.0 for empty or very sparse documents.
    """
    try:
        import pypdfium2 as pdfium
        doc = pdfium.PdfDocument(str(pdf_path))
        n_pages = len(doc)
        doc.close()
    except Exception:
        n_pages = 1

    if n_pages == 0 or not markdown.strip():
        return 0.0
    if len(markdown.strip()) / n_pages < 150:
        return 0.0  # too sparse to trust

    paragraphs = [p.strip() for p in re.split(r"\n{2,}", markdown) if p.strip()]
    text_paras = [p for p in paragraphs if not p.startswith("#") and len(p.split()) > 5]
    if len(text_paras) < 5:
        return 0.5  # not enough content to judge

    long_count = sum(1 for p in text_paras if len(p) > _LAYOUT_LONG_PARA_CHARS)
    return 1.0 - (long_count / len(text_paras))


def _layout_is_broken(markdown: str, pdf_path: Path) -> bool:
    """Return True if the layout quality score is below the acceptable threshold."""
    return _layout_score(markdown, pdf_path) < (1.0 - _LAYOUT_BAD_PARA_RATIO)


# ── Per-book processing ───────────────────────────────────────────────────────

@traceable(name="etl:embed_coaching_book", run_type="tool")
def process_book(
    book: dict,
    client: QdrantClient,
    dense_model: SentenceTransformer,
    sparse_model: SparseTextEmbedding,
    converter: DocumentConverter,
) -> None:
    """
    Parse → chunk → embed → upsert one book into Qdrant.

    Skips books already present unless book["_force"] is set (which triggers a delete + re-embed).
    Falls back to per-page column splitting when layout quality is low, and optionally runs OCR
    when book["_ocr"] is set and extracted text is suspiciously sparse.
    """
    path = PROJECT_ROOT / book["path"]
    if not path.exists():
        logger.info("  [SKIP] file not found: %s", path)
        return

    book_filter = Filter(must=[FieldCondition(key="book", match=MatchValue(value=book["key"]))])
    existing_count = client.count(COLLECTION_NAME, count_filter=book_filter).count
    if existing_count > 0:
        if not book.get("_force"):
            logger.info("  [SKIP] already embedded (%d chunks)", existing_count)
            return
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=FilterSelector(filter=book_filter),
        )
        logger.info("  [REEMBED] deleted %d old chunks", existing_count)

    # ── Parse: try Docling natively; fall back to column-split if layout is broken ──
    logger.info("  Parsing with Docling (this may take a few minutes)...")
    result   = converter.convert(str(path))
    markdown = result.document.export_to_markdown()
    logger.info("  Markdown length: %d chars", len(markdown))

    n_parts  = book.get("page_split")
    pdf_path = path  # updated below if split is applied

    if n_parts:
        orig_score = _layout_score(markdown, path)
        if _layout_is_broken(markdown, path):
            logger.info("  Broken layout detected (score=%.2f) — retrying with %d-column page split …", orig_score, n_parts)
        else:
            logger.info("  Layout score %.2f — applying configured %d-column page split …", orig_score, n_parts)

        split_path     = split_pdf_pages(path, n_parts)
        split_result   = converter.convert(str(split_path))
        split_markdown = split_result.document.export_to_markdown()
        split_score    = _layout_score(split_markdown, split_path)

        logger.info("  Layout score: original=%.2f  after-split=%.2f", orig_score, split_score)
        if split_score >= orig_score:
            logger.info("  Using split parse.")
            pdf_path = split_path
            markdown = split_markdown
        else:
            logger.warning("  Split did not improve layout — reverting to original parse.")

    if _check_ocr_needed(markdown, pdf_path):
        if book.get("_ocr"):
            logger.info("  Sparse text detected — re-parsing with OCR enabled (slow)…")
            ocr_pipeline = PdfPipelineOptions()
            ocr_pipeline.do_ocr = True
            ocr_pipeline.accelerator_options = AcceleratorOptions(device=AcceleratorDevice.CPU)
            ocr_converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=ocr_pipeline,
                        backend=PyPdfiumDocumentBackend,
                    )
                }
            )
            result   = ocr_converter.convert(str(pdf_path))
            markdown = result.document.export_to_markdown()
            logger.info("  Markdown length after OCR: %d chars", len(markdown))
        else:
            logger.warning(
                "  Very sparse text detected — %s may contain scanned pages. "
                "Re-run with --ocr %s to enable OCR processing.",
                book["key"], book["key"],
            )

    chunks = chunk_markdown(markdown)
    tables = sum(1 for c in chunks if c["chunk_type"] == "table")
    logger.info("  Chunks: %d total  |  %d tables  |  %d text", len(chunks), tables, len(chunks) - tables)

    texts = [c["text"] for c in chunks]

    logger.info("  Embedding and storing (runs locally)...")
    BATCH = 64  # smaller batch — BGE-large is memory-hungry on CPU
    for batch_start in range(0, len(texts), BATCH):
        batch_texts    = texts[batch_start : batch_start + BATCH]
        batch_chunks   = chunks[batch_start : batch_start + BATCH]
        batch_end_idx  = batch_start + len(batch_texts)

        dense_vecs  = dense_model.encode(
            batch_texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False
        ) # makes 1024-dim dense vectors for semantic search
        sparse_vecs = list(sparse_model.embed(batch_texts)) # makes sparse BM25 vectors for keyword search

        points = [
            PointStruct(
                id=stable_qdrant_id(book["key"], batch_start + j),
                vector={
                    "dense": dense_vecs[j].tolist(),
                    "sparse": SparseVector(
                        indices=sparse_vecs[j].indices.tolist(),
                        values=sparse_vecs[j].values.tolist(),
                    ),
                },
                payload={
                    "book":        book["key"],
                    "book_title":  book["title"],
                    "category":    book["category"],
                    "chunk_type":  batch_chunks[j]["chunk_type"],
                    "section":     batch_chunks[j]["section"],
                    "chunk_index": batch_start + j,
                    "chunk_id":    stable_id(book["key"], batch_start + j),
                    "text":        batch_texts[j],
                },
            )
            for j in range(len(batch_texts))
        ]
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        logger.info("    %d/%d chunks stored", batch_end_idx, len(texts))

    logger.info("  Done: %s", book["key"])


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Embed coaching books into Qdrant.")
    parser.add_argument(
        "--reembed", metavar="KEY", nargs="+",
        help="Force re-embedding for these book keys (deletes existing chunks first).",
    )
    parser.add_argument(
        "--ocr", metavar="KEY", nargs="+",
        help="Enable OCR for these book keys (use for scanned PDFs).",
    )
    args = parser.parse_args()
    reembed_keys = set(args.reembed or [])
    ocr_keys     = set(args.ocr or [])

    logger.info("Dense model  : %s", EMBED_MODEL)
    logger.info("Sparse model : %s", SPARSE_MODEL)
    logger.info("Qdrant path  : %s", QDRANT_PATH)

    QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(QDRANT_PATH))

    existing_collections = {c.name for c in client.get_collections().collections}
    if COLLECTION_NAME not in existing_collections:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "dense": VectorParams(size=1024, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False)),
            },
        )
        logger.info("Created collection '%s'", COLLECTION_NAME)

    logger.info("Collection has %d chunks at start", client.count(COLLECTION_NAME).count)

    logger.info("Loading dense embedding model…")
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"
    dense_model  = SentenceTransformer(EMBED_MODEL, device=DEVICE)
    logger.info("Loading sparse (BM25) model…")
    sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)

    # Force CPU: MPS (Apple Silicon) doesn't support float64 needed by Docling's layout model.
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.accelerator_options = AcceleratorOptions(device=AcceleratorDevice.CPU)

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=PyPdfiumDocumentBackend,
            )
        }
    )

    all_books = list(BOOKS)
    for entry in ARTICLE_DIRS:
        all_books.extend(books_from_dir(entry))

    failed: list[str] = []
    for book in all_books:
        if book["key"] in reembed_keys:
            book = {**book, "_force": True}
        if book["key"] in ocr_keys:
            book = {**book, "_ocr": True}
        logger.info("[%s] %s", book["key"], book["title"])
        try:
            process_book(book, client, dense_model, sparse_model, converter)
        except Exception:
            logger.exception("[%s] Failed — skipping", book["key"])
            failed.append(book["key"])

    logger.info("Final collection size: %d chunks", client.count(COLLECTION_NAME).count)
    logger.info("Qdrant stored at: %s", QDRANT_PATH)
    if failed:
        logger.warning("Books that failed and were skipped: %s", ", ".join(failed))


if __name__ == "__main__":
    main()
