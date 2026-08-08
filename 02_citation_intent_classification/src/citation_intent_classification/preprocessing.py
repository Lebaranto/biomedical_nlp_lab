import html
import re
import unicodedata

from nltk.tokenize import word_tokenize


DOI_PATTERN = re.compile(r"(?:https?://(?:dx\.)?doi\.org/)?\b10\.\d{4,9}/[-._;()/:a-zA-Z0-9]+", re.IGNORECASE)
URL_PATTERN = re.compile(r"(?:https?://|www\.|//)[^\s<>()]+", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_PATTERN = re.compile(r"(?<!\w)(?:\+\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?){2,}\d{2,4}(?!\w)")
NUMBER_PATTERN = re.compile(r"^[+-]?(?:\d+(?:[.,]\d+)?|[.,]\d+)(?:[eE][+-]?\d+)?$")
NUMBER_RANGE_PATTERN = re.compile(r"^[+-]?\d+(?:[.,]\d+)?[-–—][+-]?\d+(?:[.,]\d+)?$")

SPECIAL_TOKEN_TO_PLACEHOLDER = {
    "<cite>": "zzzcitetokenzzz",
    "<url>": "zzzurltokenzzz",
    "<doi>": "zzzdoitokenzzz",
    "<email>": "zzzemailtokenzzz",
    "<phone>": "zzzphonetokenzzz",
    "<unknown_section>": "zzzunknownsectiontokenzzz",
}
PLACEHOLDER_TO_SPECIAL_TOKEN = {value: key for key, value in SPECIAL_TOKEN_TO_PLACEHOLDER.items()}
KEPT_SYMBOLS = {"%", "<", ">", "=", "≤", "≥", "±"}
MOJIBAKE_FRAGMENTS = {"ï¿½", "Ãƒ", "Ã‚", "Ã¢"}


def add_cite_token(data):
    data = data.copy()
    data["string"] = data.apply(
        lambda row: row["string"][: row["citeStart"]] + " <cite> " + row["string"][row["citeEnd"] :],
        axis=1,
    )
    return data


def normalize_unicode(text):
    text = html.unescape(str(text))
    text = unicodedata.normalize("NFKC", text)
    text = "".join(" " if unicodedata.category(char) in {"Cc", "Cf"} else char for char in text)
    return re.sub(r"\s+", " ", text).strip()


def replace_structured_patterns(text):
    text = DOI_PATTERN.sub(" <doi> ", text)
    text = EMAIL_PATTERN.sub(" <email> ", text)
    text = URL_PATTERN.sub(" <url> ", text)
    return PHONE_PATTERN.sub(" <phone> ", text)


def protect_special_tokens(text):
    for special_token, placeholder in SPECIAL_TOKEN_TO_PLACEHOLDER.items():
        text = text.replace(special_token, placeholder)
    return text


def is_only_punctuation_or_symbols(token):
    return bool(token) and all(unicodedata.category(char)[0] in {"P", "S"} for char in token)


def normalize_token(token):
    token = token.strip().strip('"`')
    if not token:
        return None
    if token in PLACEHOLDER_TO_SPECIAL_TOKEN:
        return PLACEHOLDER_TO_SPECIAL_TOKEN[token]
    if any(fragment in token for fragment in MOJIBAKE_FRAGMENTS):
        return None
    if NUMBER_PATTERN.fullmatch(token) or NUMBER_RANGE_PATTERN.fullmatch(token):
        return "<num>"
    if token in KEPT_SYMBOLS:
        return token
    if is_only_punctuation_or_symbols(token):
        return None
    return token


def clean_and_tokenize(text):
    text = replace_structured_patterns(normalize_unicode(text).lower())
    text = protect_special_tokens(text)
    return [token for token in (normalize_token(tok) for tok in word_tokenize(text)) if token is not None]


def preprocess_text(data, source_column="string", save_column="tokenized_string"):
    data = data.copy()
    data[save_column] = data[source_column].apply(clean_and_tokenize)
    return data

