def clean_text(raw_text: str) -> str:
    """
    Cleans the input text by collapsing extra whitespace, tabs and line breaks
    into single spaces, and stripping leading/trailing whitespace.

    Args:
        raw_text (str): The raw input text.

    Returns:
        str: The cleaned text, with original letter case preserved.
    """
    return " ".join(raw_text.split())


def word_count(text: str) -> int:
    return len(text.split())