"""
src/preprocessing/clean_text.py

Text cleaning and preprocessing for hate speech detection on
code-mixed Hinglish social media text.

Key features:
- URL / mention / hashtag removal
- Repeated-character and elongated-word normalization  (haaate → hate)
- Light punctuation retention for emphasis markers (! ? kept as tokens)
- Emoji preservation option (emojis mapped to placeholder tokens)
- Devanagari script preserved alongside Roman text
- Hinglish subset detection (post contains Devanagari OR known Hindi tokens)
"""

import re
import unicodedata
import pandas as pd
from typing import Optional, Tuple


# ── Hinglish / Hindi signal tokens (common transliterations + Devanagari range)
_DEVANAGARI_RE = re.compile(r'[\u0900-\u097F]')

# Common Hinglish transliteration words that signal code-mixing
_HINGLISH_TOKENS = frozenset({
    'kya', 'hai', 'nahi', 'nahin', 'hain', 'tha', 'thi', 'the',
    'mera', 'meri', 'tera', 'teri', 'tumhara', 'apna', 'apni',
    'yaar', 'bhai', 'bhen', 'behan', 'dost', 'log', 'sab',
    'bahut', 'zyada', 'thoda', 'acha', 'achha', 'accha', 'theek',
    'bilkul', 'sahi', 'galat', 'bekaar', 'bakwaas',
    'karo', 'karna', 'karein', 'karte', 'kar',
    'aao', 'jao', 'agar', 'toh', 'tou', 'aur', 'par', 'lekin',
    'kyunki', 'isliye', 'phir', 'fir',
    'bohot', 'bohat', 'kaafi',
    'isko', 'usko', 'inko', 'unko', 'mujhe', 'tumhe', 'use',
    'desh', 'bharat', 'india', 'hindustan',
    'gali', 'gaali', 'maa', 'baap',
})


# ── Emoji unicode ranges for detection / replacement ──────────────────────────
_EMOJI_PATTERN = re.compile(
    r'[\U0001F1E0-\U0001F1FF'   # Flags
    r'\U0001F300-\U0001F5FF'    # Symbols & pictographs
    r'\U0001F600-\U0001F64F'    # Emoticons
    r'\U0001F680-\U0001F6FF'    # Transport & map
    r'\U0001F700-\U0001F77F'    # Alchemical
    r'\U0001F780-\U0001F7FF'    # Geometric extended
    r'\U0001F800-\U0001F8FF'    # Supplemental arrows-C
    r'\u2600-\u26FF'            # Misc symbols
    r'\u2700-\u27BF'            # Dingbats
    r']'
)


def is_hinglish(text: str) -> bool:
    """
    Detect whether a piece of text is code-mixed Hinglish.

    A post is classified as Hinglish if it contains:
    - At least one Devanagari character, OR
    - At least one known Hindi transliteration token alongside English tokens

    Args:
        text (str): Raw or cleaned text

    Returns:
        bool: True if the text is likely Hinglish / code-mixed

    Example:
        >>> is_hinglish("yaar this is bakwaas")
        True
        >>> is_hinglish("This is pure English text")
        False
    """
    if _DEVANAGARI_RE.search(text):
        return True
    tokens = set(re.findall(r'[a-zA-Z]+', text.lower()))
    has_english = bool(tokens - _HINGLISH_TOKENS)
    has_hindi   = bool(tokens & _HINGLISH_TOKENS)
    return has_english and has_hindi


def _normalize_repeated_chars(text: str, max_repeat: int = 2) -> str:
    """
    Collapse runs of the same character to at most `max_repeat` occurrences.

    Examples:
        haaate  → haate  (max_repeat=2)
        sooooo  → soo
        !!!!!!  → !!

    This is important for social media text where elongation is used for
    emphasis and hate speech intensity.
    """
    return re.sub(r'(.)\1{' + str(max_repeat) + r',}', r'\1' * max_repeat, text)


def _normalize_elongated_words(text: str) -> str:
    """
    Normalize elongated words (3+ vowel repetitions in a single token)
    back to a single vowel.

    Examples:
        looooove → love
        haaate   → hate   (after repeated-char pass reduces to haate → hate)

    Applied AFTER _normalize_repeated_chars.
    """
    return re.sub(r'([aeiouAEIOU])\1+', r'\1', text)


def clean_text(
    text: str,
    keep_emojis: bool = False,
    normalize_repeated: bool = True,
    keep_emphasis_punct: bool = True,
) -> str:
    """
    Clean and preprocess text for hate speech detection.

    Handles code-mixed Hinglish text by:
    - Removing URLs, @mentions
    - Removing hashtag symbol but keeping the word (#hate → hate)
    - Normalizing repeated characters (haaate → haate)
    - Normalizing elongated vowels (looove → love)
    - Optionally preserving emojis as <EMOJI> placeholder tokens
    - Optionally retaining ! and ? as emphasis markers (converted to tokens)
    - Preserving Devanagari script alongside Roman text
    - Converting to lowercase

    Args:
        text (str): Raw input text
        keep_emojis (bool): Replace emojis with <EMOJI> tokens. Default: False
        normalize_repeated (bool): Collapse repeated chars. Default: True
        keep_emphasis_punct (bool): Convert ! → <EXCL>, ? → <QUES>. Default: True

    Returns:
        str: Cleaned text

    Example:
        >>> clean_text("Check @user http://t.co/x #Haate yaar!!! 😡")
        'check haate yaar <excl>'
    """
    if not isinstance(text, str):
        return ""

    # 1. Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # 2. Remove @mentions
    text = re.sub(r'@\w+', '', text)

    # 3. Strip hashtag symbol, keep the word
    text = re.sub(r'#(\w+)', r'\1', text)

    # 4. Handle emojis
    if keep_emojis:
        text = _EMOJI_PATTERN.sub(' <emoji> ', text)
    else:
        text = _EMOJI_PATTERN.sub(' ', text)

    # 5. Emphasis punctuation → placeholder tokens (before stripping punctuation)
    if keep_emphasis_punct:
        text = re.sub(r'!+', ' <excl> ', text)
        text = re.sub(r'\?+', ' <ques> ', text)
    else:
        text = re.sub(r'[!?]+', ' ', text)

    # 6. Normalize repeated characters (haaate → haate)
    if normalize_repeated:
        text = _normalize_repeated_chars(text, max_repeat=2)
        text = _normalize_elongated_words(text)

    # 7. Remove remaining special characters, keep:
    #    - ASCII alphanumeric
    #    - Devanagari  (U+0900–U+097F)
    #    - Placeholder tokens (< >) already inserted above
    #    - Spaces
    text = re.sub(r'[^a-zA-Z0-9\s\u0900-\u097F<>_]', ' ', text)

    # 8. Lowercase
    text = text.lower()

    # 9. Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def preprocess_dataframe(
    df: pd.DataFrame,
    text_column: str,
    keep_emojis: bool = False,
    normalize_repeated: bool = True,
    keep_emphasis_punct: bool = True,
    tag_hinglish: bool = True,
) -> pd.DataFrame:
    """
    Apply text cleaning to a DataFrame and optionally tag Hinglish rows.

    Adds columns:
    - `{text_column}_cleaned`  : cleaned text
    - `is_hinglish`            : bool flag (only if tag_hinglish=True)

    Args:
        df (pd.DataFrame): Input dataframe
        text_column (str): Column containing raw text
        keep_emojis (bool): Preserve emojis as <emoji> tokens. Default: False
        normalize_repeated (bool): Collapse repeated chars. Default: True
        keep_emphasis_punct (bool): Keep ! and ? as emphasis tokens. Default: True
        tag_hinglish (bool): Add `is_hinglish` column. Default: True

    Returns:
        pd.DataFrame: Copy of df with new columns

    Raises:
        ValueError: If text_column not found in dataframe

    Example:
        >>> df = pd.DataFrame({'text': ['yaar bahut bakwaas hai', 'pure English text']})
        >>> out = preprocess_dataframe(df, 'text')
        >>> out['is_hinglish'].tolist()
        [True, False]
    """
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in dataframe")

    df_copy = df.copy()

    df_copy[f'{text_column}_cleaned'] = df_copy[text_column].apply(
        lambda x: clean_text(
            str(x),
            keep_emojis=keep_emojis,
            normalize_repeated=normalize_repeated,
            keep_emphasis_punct=keep_emphasis_punct,
        )
    )

    if tag_hinglish:
        # Tag using the RAW text (before cleaning) to preserve Devanagari signals
        df_copy['is_hinglish'] = df_copy[text_column].apply(
            lambda x: is_hinglish(str(x))
        )

    return df_copy


# ── quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_cases = [
        ("Check this link http://example.com #HateSpeech @user123", False),
        ("यह देशद्रोही है!!! 😡😡 @twitter #Boycott", True),
        ("yaar yeh bahut bakwaas hai, seriously!!!!", True),
        ("looooove this movie sooooo muchhh", False),
        ("haaate you people so much!!!!! 😠", False),
        ("This is pure English text with no Hindi", False),
        ("#मोदी भक्त हो तो follow करो 😂😂", True),
    ]

    print("=" * 72)
    print("CLEAN_TEXT — SELF TEST")
    print("=" * 72)

    for raw, expected_hinglish in test_cases:
        cleaned = clean_text(raw, keep_emojis=True)
        flag    = is_hinglish(raw)
        marker  = "✓" if flag == expected_hinglish else "✗"
        print(f"\n  Raw:      {raw}")
        print(f"  Cleaned:  {cleaned}")
        print(f"  Hinglish: {flag}  {marker}")

    print("\n" + "=" * 72)
    print("DATAFRAME PREPROCESSING")
    print("=" * 72)

    import pandas as pd
    df = pd.DataFrame({
        'text': [r for r, _ in test_cases],
        'label': [1, 1, 0, 0, 1, 0, 1],
    })
    out = preprocess_dataframe(df, 'text', keep_emojis=True)
    print(out[['text', 'text_cleaned', 'is_hinglish', 'label']].to_string(index=False))