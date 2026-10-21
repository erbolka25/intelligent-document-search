import random


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

def truncate(text: str, max_length: int) -> str:
    if len(text) <= max_length:
        return text
    if max_length <= 3:
        return "..."  # Если лимит слишком маленький, просто возвращаем точки
    return text[:max_length - 3] + "..."

def is_probably_sentence_end(char: str) -> bool:
    return char in {'.', '!', '?'}


def count_tokens_approx(text:str)->int:
  length = len(text)
  length = length // 4
  return length
def split_sentences(text:str)->list:
    return [s.strip()for s in text.split('.') if s.strip()]
def get_snippet(text:str, index:int, window:int)->str:
    start = max(0, index - window)
    end = index + window + 1
    return text[start:end]
def top_n(items:list,scores:list, n:int)->list:
    return [item for item, score in sorted(zip(items, scores), key=lambda x: x[1], reverse=True)[:n]]
def chunking_list(items:list, chunk_size:int)->list:
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
def dedupe_preserve_order(items:list)-> list:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
def long_chunks(chunks:list)->list:
    return [c for c in chunks if len(c)>50]
def flatten(list_of_lists:list)->list:
    return [item for sublist in list_of_lists for item in sublist]
def build_doc_index(docs:list[dict])->dict:
    return {doc['id']: doc ["content"] for doc in docs}
def word_frequency(text:str)->dict:
    words = text.split()
    freq = {}
    for word in words:
        word = word.lower()
        word = word.strip('.,!?;:')  # Remove punctuatio
        freq[word] = freq.get(word, 0) + 1
    return freq
def find_common_words(text1:str, text2:str)->set:
    clean_1=text1.lower().replace('.','')
    clean_2=text2.lower().replace('.','')
    words1 = set(clean_1.split())
    words2 = set(clean_2.split())
    return words1.intersection(words2)
def is_duplicate(content:str,seen_hashes:set)->bool:
    content_hash = hash(content)
    if content_hash in seen_hashes:
        return True
    seen_hashes.add(content_hash)
    return False
def inverted_index(docs:list[dict])->dict:
    index = {}
    for doc in docs:
        doc_id = doc['id']
        words = doc['content'].lower().split()
        for word in words:
            word = word.strip('.,!?;:')  # Remove punctuation
            if word not in index:
                index[word] = set()
            index[word].add(doc_id)
    return index
def classify_score(score:float)->str:
    if score < 0.5:
        return "low"
    elif score < 0.8:
        return "medium"
    else:
        return "high"
def retry_with_limit(max_attempts: int) -> bool:
    attempts = 0
    success = False  # Создаем флаг успеха перед циклом
    
    while attempts < max_attempts:
        attempts += 1
        success = random.random() < 0.5
        if success:
            break  # Выходим из цикла по условию, если success == True
            
    return success  # Возвращаем сам флаг (он будет True, если сработал break, и False, если попытки кончились) 
def build_length_index(docs:list[dict])->dict:
    return {doc['id']: len(doc['content']) for doc in docs}
def filter_by_threshold(results:list[tuple], threshold:float)->list:
    return [doc_id for doc_id, score in results if score >= threshold]
def chunk_generation(text:str, chunk_size:int)->list:
    if chunk_size <= 0:
        return
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]

def search(query: str, top_k: int = 5, *, rerank: bool = False) -> dict:
     erbol = {
        "query": query,
        "top_k": top_k,
        "rerank": rerank
    }
     return erbol