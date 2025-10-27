def capitalize(s: str) -> str:
    """Return s with its first character uppercased.

    Does not modify the case of the rest of the string (unlike str.capitalize()).
    """
    if not isinstance(s, str):
        raise TypeError("capitalize() expects a str")
    return s[0].upper() + s[1:]