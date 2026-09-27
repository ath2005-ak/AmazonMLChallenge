import re
import unicodedata

# Common legal suffixes to normalize or strip during fingerprinting
LEGAL_SUFFIXES_PATTERN = re.compile(
    r'\b(pvt|private|ltd|limited|inc|incorporated|corp|corporation|llc|llp|co|company|sarl|sas|gmbh|s\.a\.r\.l|s\.a\.s|s\.a|n\.v|b\.v)\b',
    flags=re.IGNORECASE
)

# Address abbreviation mappings
ADDRESS_ABBR = {
    r'\brd\b': 'road',
    r'\bst\b': 'street',
    r'\bave\b': 'avenue',
    r'\bdr\b': 'drive',
    r'\bblvd\b': 'boulevard',
    r'\bfl\b': 'floor',
    r'\bapt\b': 'apartment',
    r'\bste\b': 'suite',
    r'\btq\b': 'taluk',
    r'\bdist\b': 'district',
    r'\bpo\b': 'post office',
    r'\bno\b': 'number',
}

PINCODE_PATTERN = re.compile(r'\b\d{5,6}\b')

def clean_text(text: str) -> str:
    """Basic unicode normalization and lowercasing."""
    if not text or not isinstance(text, str):
        return ""
    
    text = unicodedata.normalize('NFC', text).lower()
    # Replace & with and
    text = text.replace('&', ' and ')
    # Replace non-alphanumeric chars except whitespace with space
    text = re.sub(r'[^\w\s]', ' ', text)
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_business_name(name: str, strip_legal: bool = False) -> str:
    """Normalizes business name."""
    cleaned = clean_text(name)
    if strip_legal:
        cleaned = LEGAL_SUFFIXES_PATTERN.sub('', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def extract_name_fingerprint(name: str) -> str:
    """Returns sorted token fingerprint of cleaned business name."""
    cleaned = clean_business_name(name, strip_legal=True)
    tokens = sorted([t for t in cleaned.split() if len(t) > 1])
    return " ".join(tokens)

def clean_address(address: str) -> str:
    """Normalizes address string."""
    cleaned = clean_text(address)
    for pattern, repl in ADDRESS_ABBR.items():
        cleaned = re.sub(pattern, repl, cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def extract_pincode(address: str) -> str:
    """Extracts 5 or 6 digit pincode/zipcode if present."""
    if not address or not isinstance(address, str):
        return ""
    matches = PINCODE_PATTERN.findall(address)
    return matches[0] if matches else ""

