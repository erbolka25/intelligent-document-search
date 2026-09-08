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
def parse_chunk_size(value: str) -> int:
    """
    Parses a string representing a chunk size and returns it as an integer.

    Args:
        value (str): The chunk size as a string.

    Returns:
        int: The chunk size as an integer.

    Raises:
        ValueError: If the input value is not a valid integer.
    """
    try:
        return int(value.strip())
    except ValueError:
        raise ValueError(f"Invalid chunk size: {value}. Must be an integer.")
def is_valid_similarity_score(value: float) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("Score must be a float or int, not bool or other types")
    return 0.0 <= value <= 1.0
def format_time(seconds:float)->str:
    """
    Formats a time duration given in seconds into a human-readable string.
    Args:
        seconds (float): The time duration in seconds.

    Returns:
        str: A formatted string representing the time duration.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remaining_seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"
def average(numbers: list[float])->float:
   total_sum=0.0
   count = 0
   for num in numbers:
       total_sum += num
       count += 1
   if count == 0:
       return 0.0  
   return total_sum / count