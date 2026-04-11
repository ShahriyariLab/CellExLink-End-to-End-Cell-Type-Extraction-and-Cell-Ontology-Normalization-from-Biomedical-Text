import re


_TOKEN_FINDER = re.compile(r"[^\W_]+|[^\w\s]|_|,", re.UNICODE)


def split_tokens(value):
    return _TOKEN_FINDER.findall(str(value))


def _replace_tail(word, suffix, replacement):
    return word[:-len(suffix)] + replacement


def normalize_token(value):
    word = str(value)

    if not word.endswith("s"):
        return word

    if word.endswith("viruses"):
        return _replace_tail(word, "uses", "us")

    if word.endswith("ies"):
        if not word.endswith(("eies", "aies")):
            return _replace_tail(word, "ies", "y")

    if word.endswith("es"):
        if not word.endswith(("aes", "ees", "oes")):
            if word.endswith("sses"):
                return _replace_tail(word, "es", "")
            return _replace_tail(word, "es", "e")

    if word.endswith(("us", "ss")):
        return word

    return _replace_tail(word, "s", "")


def normalize_text(value):
    return " ".join(normalize_token(part) for part in split_tokens(value))