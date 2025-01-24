# image_analysis.py
from pydantic import BaseModel
import requests
from PIL import Image
import io
import ollama
import os
import json

class ImageAnalysis(BaseModel):
    setting: str
    characters: list[str]
    mood: str
    objects: list[str]
    potential_conflicts: list[str]

    @classmethod
    def from_llava_response(cls, response):
        return cls(
            setting=response.get("setting_description", "Unknown setting"),
            # Extract character descriptions from nested objects
            characters=[item.get("description", "") 
                      for item in response.get("characters", [])
                      if isinstance(item, dict)],
            # Flatten mood analysis structure
            mood=response.get("mood_analysis", "Neutral").split(". ")[0],
            # Extract object names from nested structure
            objects=[item.get("object", "") 
                   for item in response.get("significant_objects", [])
                   if isinstance(item, dict)],
            potential_conflicts=response.get("potential_conflicts", [])
        )
        

class MultimodalAnalyzer:
    def __init__(self, model="gemma2:27b"):
        self.model = model
        
    def _load_image(self, image_source):
        """Load image from file path or URL with validation"""
        try:
            if isinstance(image_source, str):
                if image_source.startswith(('http://', 'https://')):
                    response = requests.get(image_source, timeout=10)
                    response.raise_for_status()
                    if not response.headers.get('Content-Type', '').startswith('image/'):
                        raise ValueError("URL does not point to an image")
                    return response.content
                else:
                    if not os.path.exists(image_source):
                        raise FileNotFoundError(f"Image file not found: {image_source}")
                    with open(image_source, 'rb') as f:
                        content = f.read()
                        Image.open(io.BytesIO(content)).verify()  # Validate image format
                        return content
            elif isinstance(image_source, bytes):
                Image.open(io.BytesIO(image_source)).verify()
                return image_source
            else:
                raise ValueError("Invalid image source type")
        except Exception as e:
            print(f"Image Loading Error: {str(e)}")
            raise
    
    
    def analyze(self, image_source):
        image_bytes = self._load_image(image_source)
        if self.model == "gemma2:27b":
            return self._analyze_with_llava(image_bytes)
        else:
            raise ValueError(f"Unsupported model: {self.model}")

    def _analyze_with_llava(self, image_bytes):
        try:
            response = ollama.generate(
                model="gemma2:27b",  # Use official LLaVA model instead of gemma2
                prompt="""Analyze this image and return JSON with:
            - setting description
            - list of characters
            - mood analysis
            - list of significant objects
            - potential conflicts""",
                images=[image_bytes],  # Pass raw bytes directly
                format="json",
                stream=False
        )
# Validate response structure
            if not response or 'response' not in response:
                raise ValueError("Invalid response format from LLaVA")
            
            parsed = json.loads(response['response'])
            return ImageAnalysis.from_llava_response(parsed)
        
        except Exception as e:
        # Add error logging
            print(f"LLaVA Error: {str(e)}")
            if 'response' in locals():
                print(f"Raw Response: {response.get('response', 'No response')}")
            raise RuntimeError("Image analysis failed") from e

