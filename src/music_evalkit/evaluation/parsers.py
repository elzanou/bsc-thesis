import json
import re
from dataclasses import dataclass


@dataclass
class MCQPrediction:
    """Parsed MCQ prediction."""

    reason: str | None  # Model's reason before selecting
    letter: str | None  # A, B, C, or D
    option_text: str | None  # The actual option text
    raw: str  # Original response


@dataclass
class PairwisePrediction:
    """Parsed pairwise prediction."""

    reason: str | None  # Model's reason before selecting
    choice: str | None  # "A" or "B"
    raw: str  # Original response


@dataclass
class OpenEndedBasicPrediction:
    """Parsed open-ended basic prediction."""

    reason: str | None
    category: str | None
    mistake: str | None
    feedback: str | None
    raw: str
    parse_success: bool = True  # False if JSON parsing failed entirely


# Regex to find a letter near answer-related keywords, or the LAST standalone
# occurrence if no keyword context is found.  Searching near keywords first
# prevents matching the English article "A" in reasoning text.
_ANSWER_KEYWORD_RE = re.compile(
    r"(?:answer|choice|select|pick|choose|option)\s*[:=]?\s*[^A-D]*\b([ABCD])\b",
    re.IGNORECASE,
)


def _extract_letter(response_text: str, valid: str = "ABCD") -> str | None:
    """Extract the most likely answer letter from free text.

    Strategy:
    1. Look for a letter near answer-related keywords.
    2. Fall back to the LAST standalone letter (avoids matching article "A"
       at the beginning of a sentence).
    """
    upper = response_text.upper()

    # Try keyword-anchored extraction first
    match = _ANSWER_KEYWORD_RE.search(response_text)
    if match and match.group(1).upper() in valid:
        return match.group(1).upper()

    # Fall back to the LAST standalone letter
    pattern = r"\b([" + valid + r"])\b"
    matches = list(re.finditer(pattern, upper))
    if matches:
        return matches[-1].group(1)

    return None


def parse_mcq_response(response_text: str, options: list[str]) -> MCQPrediction:
    """Parse MCQ response to extract answer letter and map to option text.

    Args:
        response_text: Model's response (e.g., '{"answer": "A"}')
        options: List of option texts [opt_A, opt_B, opt_C, opt_D]

    Returns:
        MCQPrediction with letter, option_text, and raw response.
    """
    reason = None
    letter = None
    option_text = None

    try:
        # Try to parse as JSON
        data = json.loads(response_text)
        reason = data.get("reason")
        letter = data.get("answer", "").strip().upper()
    except json.JSONDecodeError:
        # Extract letter from raw text (last match or keyword-anchored)
        letter = _extract_letter(response_text, "ABCD")

    # Map letter to option text
    if letter and letter in "ABCD":
        idx = ord(letter) - ord("A")
        if idx < len(options):
            option_text = options[idx].strip()
    else:
        # Fallback: model may have written the category name directly — try exact match
        mapped = text_to_letter(letter, options) if letter else None
        if mapped:
            letter = mapped
            idx = ord(letter) - ord("A")
            option_text = options[idx].strip() if idx < len(options) else None
        else:
            letter = None

    return MCQPrediction(reason=reason, letter=letter, option_text=option_text, raw=response_text)


def parse_pairwise_response(response_text: str) -> PairwisePrediction:
    """Parse pairwise response to extract choice (A or B).

    Args:
        response_text: Model's response (e.g., '{"reason": "...", "answer": "A"}')

    Returns:
        PairwisePrediction with reasoning, choice, and raw response.
    """
    reason = None
    choice = None

    try:
        # Try to parse as JSON
        data = json.loads(response_text)
        reason = data.get("reason")
        answer = str(data.get("answer", "")).strip().upper()
        if answer in ("A", "B"):
            choice = answer
    except json.JSONDecodeError:
        # Extract A or B from raw text (last match or keyword-anchored)
        choice = _extract_letter(response_text, "AB")

    return PairwisePrediction(reason=reason, choice=choice, raw=response_text)


# Regex character class that matches regular quotes and common smart/curly quotes
_QUOTE = r"""['\"'\u2018\u2019\u201c\u201d]"""


VALID_CATEGORIES = frozenset({
    "articulation", "dynamics", "harmony", "no_mistake",
    "pitch", "rhythm_and_timing", "technique", "tempo",
})


def normalize_category(raw: str | None) -> str | None:
    """Normalize a predicted category to a valid category name.

    Handles formatting issues from Flamingo models:
    - Whitespace/case variations
    - Rhythm spelling variants (rhythm.and timing, rhythm-and-timing, etc.)
    - no_mistake typos and Greek character substitutions

    Combined labels (e.g., "pitch, technique") are NOT recovered — they
    indicate comprehension issues, not formatting problems.

    Returns None if the category cannot be mapped to a valid one.
    """
    if not raw:
        return None

    cat = raw.strip().lower()

    # Already valid
    if cat in VALID_CATEGORIES:
        return cat

    # Combined labels — do not recover
    if "," in cat:
        return None

    # Rhythm variants: anything starting with "rhythm" maps to rhythm_and_timing
    if cat.startswith("rhythm"):
        return "rhythm_and_timing"

    # no_mistake typos and Greek chars
    if cat.startswith("no_m"):
        return "no_mistake"

    return None


_NO_MISTAKE_PATTERN = re.compile(r"^\{no_mistake[}\s:]")


def _try_json_parse(text: str) -> OpenEndedBasicPrediction | None:
    """Try to parse text as JSON. Returns prediction or None on failure."""
    try:
        # Fix missing opening quote: {reason": → {"reason":
        fixed = re.sub(r'^\{(\w)', r'{"\1', text)
        data = json.loads(fixed)
        # Handle wrong key names in parsed JSON
        category = data.get("category") or data.get("dictionary") or data.get("directory")
        return OpenEndedBasicPrediction(
            reason=data.get("reason"),
            category=category,
            mistake=data.get("mistake"),
            feedback=data.get("feedback"),
            raw=text,
            parse_success=True,
        )
    except (json.JSONDecodeError, AttributeError):
        return None


def parse_open_ended_strict(response_text: str) -> OpenEndedBasicPrediction:
    """Parse open-ended response using JSON only — no fallbacks.

    If json.loads() fails, all fields are None and parse_success is False.
    No regex fallback, no quote fixing, no key name mapping.
    """
    try:
        data = json.loads(response_text.strip())
        return OpenEndedBasicPrediction(
            reason=data.get("reason"),
            category=data.get("category"),
            mistake=data.get("mistake"),
            feedback=data.get("feedback"),
            raw=response_text,
            parse_success=True,
        )
    except (json.JSONDecodeError, AttributeError):
        return OpenEndedBasicPrediction(
            reason=None,
            category=None,
            mistake=None,
            feedback=None,
            raw=response_text,
            parse_success=False,
        )


def parse_open_ended_recover(response_text: str) -> OpenEndedBasicPrediction:
    """Parse open-ended response with maximum recovery.

    Recovery pipeline:
    1. Try standard JSON parse
    2. Try fixing single quotes → double quotes
    3. Try fixing wrong key names (dictionary/directory → category)
    4. Detect {no_mistake} pattern
    5. Fall back to regex field extraction

    Category normalisation is applied to any extracted category.
    """
    text = response_text.strip()

    # 1. Try standard JSON
    pred = _try_json_parse(text)
    if pred is not None:
        pred.category = normalize_category(pred.category)
        return pred

    # 2. Try single quotes → double quotes
    if "'" in text:
        fixed = text.replace("'", '"')
        pred = _try_json_parse(fixed)
        if pred is not None:
            pred.category = normalize_category(pred.category)
            return pred

    # 2b. Detect {no_mistake} pattern (before bare-key fix which would misparse it)
    if _NO_MISTAKE_PATTERN.match(text):
        return OpenEndedBasicPrediction(
            reason=None,
            category="no_mistake",
            mistake=None,
            feedback=None,
            raw=response_text,
            parse_success=True,
        )

    # 2c. Try quoting bare keys + fixing single-quote values
    #     Handles: {reason: 'text', category: 'pitch'} → {"reason": "text", "category": "pitch"}
    fixed = re.sub(r'(\{|,)\s*(\w+)\s*:', r'\1 "\2":', text)
    if fixed != text:
        fixed = fixed.replace("'", '"')
        pred = _try_json_parse(fixed)
        if pred is not None:
            pred.category = normalize_category(pred.category)
            return pred

    # 3. Try fixing wrong key names
    fixed = text.replace('"dictionary"', '"category"').replace('"directory"', '"category"')
    if fixed != text:
        pred = _try_json_parse(fixed)
        if pred is not None:
            pred.category = normalize_category(pred.category)
            return pred

    # 4. Regex fallback
    category = None
    mistake = None
    feedback = None

    m = re.search(rf"""{_QUOTE}category{_QUOTE}\s*:\s*{_QUOTE}([^'\"'\u2018\u2019\u201c\u201d]+){_QUOTE}""", text)
    if m:
        category = normalize_category(m.group(1))

    m = re.search(rf"""{_QUOTE}mistake{_QUOTE}\s*:\s*{_QUOTE}([^'\"'\u2018\u2019\u201c\u201d]+){_QUOTE}""", text)
    if m:
        mistake = m.group(1)

    m = re.search(rf"""{_QUOTE}feedback{_QUOTE}\s*:\s*{_QUOTE}([^'\"'\u2018\u2019\u201c\u201d]+){_QUOTE}""", text)
    if m:
        feedback = m.group(1)

    # Also try extracting from wrong key names via regex
    if category is None:
        for key in ("dictionary", "directory"):
            m = re.search(rf"""{_QUOTE}{key}{_QUOTE}\s*:\s*{_QUOTE}([^'\"'\u2018\u2019\u201c\u201d]+){_QUOTE}""", text)
            if m:
                category = normalize_category(m.group(1))
                break

    has_content = category is not None or mistake is not None or feedback is not None

    return OpenEndedBasicPrediction(
        reason=None,
        category=category,
        mistake=mistake,
        feedback=feedback,
        raw=response_text,
        parse_success=has_content,
    )


def text_to_letter(text: str, options: list[str]) -> str | None:
    """Convert option text to letter (A/B/C/D).

    Args:
        text: The option text to find
        options: List of option texts

    Returns:
        Letter (A/B/C/D) or None if not found.
    """
    text_normalized = text.strip().lower()
    for i, opt in enumerate(options):
        if opt.strip().lower() == text_normalized:
            return chr(ord("A") + i)
    return None
