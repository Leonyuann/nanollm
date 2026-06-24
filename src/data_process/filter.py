""" Library for filtering text
"""
import re

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

def mask_emails(text: str) -> tuple[str, int]:
    return EMAIL_RE.subn(EMAIL_MASK, text)

def mask_phone_numbers(text: str) -> tuple[str, int]:
    return PHONE_NUMBER_RE.subn(PHONE_NUMBER_MASK, text)

def mask_ips(text: str) -> tuple[str, int]:
    return IPS_RE.subn(IPS_MASK, text)

def mask_pii(text: str) -> tuple[str, int]:
    text, email_count = mask_emails(text)
    text, phone_number_count = mask_phone_numbers(text)
    text, ips_count = mask_ips(text)
    return text, email_count + phone_number_count + ips_count