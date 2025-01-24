# story_generator.py
import ollama
from .rag_manager import NarrativeRAG
from .utils import extract_keywords 

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class StoryEngine:
    def __init__(self):
        self.llm = Ollama(model="gemma2:27b")
        self.rag = NarrativeRAG()
        
    def generate_chapter(self, context):
        retrieved = self.rag.retrieve_context(context["latest_summary"])
        prompt = self._build_prompt(context, retrieved)
        
        chapter = self.llm.generate(prompt)
        self._validate_chapter(chapter)
        self._update_rag(chapter)
        
        return chapter

    def _build_prompt(self, context, retrieved):
        return f"""
        Write a 300-word story chapter continuing from:
        {context['summary']}
        
        Retrieved Context:
        {retrieved}
        
        Requirements:
        - Maintain {context['mood']} tone
        - Advance conflicts: {', '.join(context['conflicts'])}
        - End with a cliffhanger
        """

    def _validate_chapter(self, chapter):
        # Custom validation logic
        if len(chapter.split()) < 250:
            raise ValueError("Chapter too short")
            
    def _update_rag(self, chapter):
        self.rag.index_context(
            document=chapter,
            metadata={
                "chapter": context["current_chapter"],
                "keywords": extract_keywords(chapter)
            }
        )