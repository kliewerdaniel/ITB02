# utils.py
import re
from typing import List

# Common English stop words to filter out
stop_words = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 'is', 'it',
              'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were', 'will', 'with'}

def extract_keywords(text: str) -> List[str]:
    # Add null check and ensure max 5 elements
    words = [word for word in text.split() if word not in stop_words]
    if not words:
        return []
        
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    return sorted(word_freq, key=word_freq.get, reverse=True)[:5]

def count_words(text):
    return len([
        word for word in re.findall(r'\b\w+\b', text)
        if len(word) > 1 and not word.isnumeric()
    ])