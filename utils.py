# utils.py
import re
from typing import List

def extract_keywords(text: str) -> List[str]:
    """
    Extract key words from text using simple techniques
    """
    # Remove punctuation and convert to lowercase
    text = re.sub(r'[^\w\s]', '', text.lower())
    
    # Split into words and remove common stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at'}
    words = [word for word in text.split() if word not in stop_words]
    
    # Return top 5 most frequent words
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    return sorted(word_freq, key=word_freq.get, reverse=True)[:5]