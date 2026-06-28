""" Library for filtering text
"""
import re
import fasttext
import nltk
nltk.download("punkt_tab")  

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

def gopher_quality_filter(text: str, language : str= "english" )-> bool:
    """ Filter text based on a subset of Gopher quality criteria.

    Applied criteria:
    - Average word length between 3 and 10 characters.
    - Total number of non-symbol words between 50 and 100,000.
    - Average word length between 3 and 10 characters.
    - Less than 30% of lines end with an ellipsis ("...").

    Args:
        text: Input text to filter.
        language: Language of the text.

    Returns:
        True if the text passes the quality filter, False otherwise.
    """
    if __language_filter(language) is False:
        return False
    
    words = nltk.word_tokenize(text,language)
    
    if __word_length_filter(words) is False:
        return False
    
    if __text_length_filter(words) is False:
        return False
    
    if __ellipsis_ending_filter(text) is False:
        return False
    
    if __alphabetic_character_filter(words) is False:
        return False
    
    return True

def __word_length_filter(words: list[str], min_length = 3, max_length = 10) -> bool:
    length = 0
    for i, word in enumerate(words,start=1):
        length = length - 1 / i *(length-len(word))

    if length < min_length or length > max_length:
        return False
    return True 

def __text_length_filter(words: list[str], min_length = 50, max_length = 100000) -> bool:
    if len(words) < min_length or len(words) > max_length:
        return False
    return True

def __ellipsis_ending_filter(text:str, max_ratio=0.3) -> bool:
    ellipsis_num = sum(1 for _ in re.finditer(r"\.\.\.", text))
    line_num = sum(1 for _ in re.finditer("\n", text))
    if line_num == 0:
        return True
    if ellipsis_num / line_num > max_ratio:
        return False
    return True

def __alphabetic_character_filter(words: list[str], min_ratio =0.8) -> bool:
    num_alpha = 0
    for word in words:
        m = re.search(r"[a-zA-Z]", word)
        if m is not None:
            num_alpha += 1

    if num_alpha/len(words) < min_ratio:
        return False
    
    return True

def __language_filter(language: str, target_language: str ="english") -> bool:
    if language != target_language:
        return False
    return True
