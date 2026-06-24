""" Library for filtering text
"""
import re
import fasttext

EMAIL_RE = re.compile(r"[0-9a-zA-Z_.+-]+@[0-9a-zA-Z.]+\.[0-9a-zA-Z]+")
PHONE_NUMBER_RE = re.compile(
    r"""
    (?<![\w.:-])
    (?:
        (?:\+?\d{1,3}[\s.-]?)?
        (?:
            \(\d{3}\)[\s.-]? |
            \d{3}[\s.-]
        )
        \d{3}[\s.-]?\d{4}
        |
        \d{8,11}
    )
    (?![\w.:-])
    """,
    re.VERBOSE,
)
IPS_RE = re.compile(r"(?:[0-9]{1,3}\.){3}[0-9]{1,3}|(?:[0-9a-f]{1,4}:){7}[0-9a-f]{1,4}|(?:[0-9a-f]{1,4}:){0,6}:[0-9a-f]{1,4}")

EMAIL_MASK = "|||EMAIL_ADDRESS|||"
PHONE_NUMBER_MASK = "|||PHONE_NUMBER|||"
IPS_MASK = "|||IP_ADDRESS|||"

def language_identificatin(text: str, ident_model_path: str = "data/models/lid.176.bin") -> tuple[str, float]:
    """Identify the language of a text document.

    FastText's Python API only accepts one line per prediction, so document
    text is folded into a single whitespace-separated line before inference.

    Args:
        text: Input text to classify.
        ident_model_path: Path to the fastText language identification model.

    Returns:
        Predicted fastText language label and confidence score.
    """
    text = " ".join(text.splitlines())
    model = fasttext.load_model(ident_model_path)
    label, score = model.predict(text)
    return label[0], score[0]

def mask_emails(text: str) -> tuple[str, int]:
    """Mask email addresses in text.

    Args:
        text: Input text that may contain email addresses.

    Returns:
        Text with email addresses replaced and the number of replacements.
    """
    return EMAIL_RE.subn(EMAIL_MASK, text)

def mask_phone_numbers(text: str) -> tuple[str, int]:
    """Mask phone numbers in text.

    Args:
        text: Input text that may contain phone numbers.

    Returns:
        Text with phone numbers replaced and the number of replacements.
    """
    return PHONE_NUMBER_RE.subn(PHONE_NUMBER_MASK, text)

def mask_ips(text: str) -> tuple[str, int]:
    """Mask IP addresses in text.

    Args:
        text: Input text that may contain IPv4 or IPv6 addresses.

    Returns:
        Text with IP addresses replaced and the number of replacements.
    """
    return IPS_RE.subn(IPS_MASK, text)

def mask_pii(text: str) -> tuple[str, int]:
    """Mask supported personally identifiable information in text.

    Applies email, phone number, and IP address masking in sequence.

    Args:
        text: Input text that may contain supported PII patterns.

    Returns:
        Text with supported PII replaced and the total number of replacements.
    """
    text, email_count = mask_emails(text)
    text, phone_number_count = mask_phone_numbers(text)
    text, ips_count = mask_ips(text)
    return text, email_count + phone_number_count + ips_count
