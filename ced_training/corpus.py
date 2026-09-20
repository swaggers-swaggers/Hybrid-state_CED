"""Model-free article boundaries and exact supervision budgeting for WikiText."""
import re


def article_title(text):
    match = re.fullmatch(r"\s*=\s*([^=]+?)\s*=\s*", text)
    return match.group(1).strip() if match else None


def title_key(title):
    return " ".join(title.casefold().split())


def complete_articles(rows):
    """Drop the initial partial article; yield only articles starting at a top-level heading."""
    current = None
    for row_id, text in rows:
        if not text.strip():
            continue
        title = article_title(text)
        if title is not None:
            if current:
                yield current
            current = {"title": title, "first_row": row_id, "last_row": row_id, "texts": []}
        if current is not None:
            current["texts"].append(text)
            current["last_row"] = row_id
    if current:
        yield current


def loss_mask(remaining, width=256):
    """Mask is indexed by input position: 1 means predict the next token in this window."""
    if remaining < 1 or width < 2:
        raise ValueError("Need a positive target budget and width >= 2")
    count = min(remaining, width - 1)
    return [1] * count + [0] * (width - count)


def chunk_spans(row_counts, start, count):
    """Map a global effective-target interval to (row, first_valid, end_valid) spans."""
    if type(start) is not int or type(count) is not int or start < 0 or count < 1:
        raise ValueError("Invalid chunk interval")
    spans, cursor = [], 0
    end = start + count
    for index, size in enumerate(row_counts):
        if size < 0:
            raise ValueError("Negative row target count")
        first, last = max(start, cursor), min(end, cursor + size)
        if first < last:
            spans.append((index, first - cursor, last - cursor))
        cursor += size
        if cursor >= end:
            return spans
    raise ValueError("Chunk extends beyond the available effective targets")
