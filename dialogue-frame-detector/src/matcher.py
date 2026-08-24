"""
matcher.py
----------
Handles fuzzy text matching.
Used by both the Visual Search (OCR) and Audio Search (Whisper) to determine 
if the extracted text matches the user's target dialogue.
"""

from rapidfuzz import fuzz
import string

def clean_text(text: str) -> str:
    """
    Standardize text for comparison by lowercasing and removing punctuation.
    """
    if not text:
        return ""
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove extra spaces
    text = " ".join(text.split())
    return text

def is_match(extracted_text: str, target_dialogue: str, threshold: float = 80.0) -> bool:
    """
    Compare the extracted text from the video (OCR or Whisper) against the target dialogue.
    Uses RapidFuzz to allow for minor typos or OCR misreads (e.g. '0' vs 'O').
    
    Returns True if the similarity score is above the threshold.
    """
    clean_extracted = clean_text(extracted_text)
    clean_target = clean_text(target_dialogue)
    
    if not clean_extracted or not clean_target:
        return False
        
    # partial_ratio checks if the target dialogue is a substring of the extracted text
    # (since the extracted text might contain a whole sentence, but we only search for a phrase)
    score = fuzz.partial_ratio(clean_target, clean_extracted)
    
    return score >= threshold

def get_match_score(extracted_text: str, target_dialogue: str) -> float:
    """
    Returns the raw match score (0-100) for debugging/logging.
    """
    clean_extracted = clean_text(extracted_text)
    clean_target = clean_text(target_dialogue)
    
    if not clean_extracted or not clean_target:
        return 0.0
        
    return fuzz.partial_ratio(clean_target, clean_extracted)
