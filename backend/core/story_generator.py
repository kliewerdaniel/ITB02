# story_generator.py
from langchain_community.llms import Ollama
from .rag_manager import NarrativeRAG
from .utils import extract_keywords 
from langchain_ollama import OllamaLLM  # Replace old import


from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class StoryEngine:
    def __init__(self):
        self.llm = OllamaLLM(model="gemma2:27b")  # Updated class name
        self.rag = NarrativeRAG()
        
    def generate_chapter(self, context):
        # Change from 'latest_summary' to 'summary'
        retrieved = self.rag.retrieve_context(context["summary"])  # Fixed key name
        
        prompt = self._build_prompt(context, retrieved)
        chapter = self.llm.invoke(prompt)  # Updated generation method
        self._validate_chapter(chapter)
        self._update_rag(chapter, context)  # Pass context to update
        
        return chapter

    def _update_rag(self, chapter, context):
        keywords = extract_keywords(chapter)
        # Convert list to comma-separated string
        keyword_str = ", ".join(keywords) if keywords else "none"
    
        self.rag.index_context(
            document=chapter,
            metadata={
                "chapter": str(context["current_chapter"]),  # Ensure string type
                "keywords": keyword_str  # Now a string instead of list
        }
    )

    def _build_prompt(self, context, retrieved):
        return f"""
    Write a detailed story chapter of at least 300 words continuing from:
    {context['summary']}
    
    Retrieved Context:
    {retrieved}
    
    Requirements (STRICTLY FOLLOW):
    - Maintain {context['mood']} tone
    - Develop conflicts: {', '.join(context['conflicts'])}
    - End with a suspenseful cliffhanger
    - Minimum 300 words (VERY IMPORTANT)
    - Use descriptive language
    - Focus on character actions and dialogue
    
    Formatting Rules:
    - No markdown
    - Paragraphs separated by newlines
    - Direct speech in quotes
    
    Failure to meet word count will result in chapter rejection!
    """

    def _validate_chapter(self, chapter):
        words = chapter.split()
        if len(words) < 275:  # Give 25-word buffer
        # Attempt to expand the content instead of failing
            expanded = self.llm.invoke(f"Expand this chapter to 300 words: {chapter}")
            if len(expanded.split()) < 250:
                raise ValueError("Chapter too short after expansion")
            return expanded
        return chapter