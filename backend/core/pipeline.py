# pipeline.py
from backend.core.image_analysis import MultimodalAnalyzer
import ollama
from backend.core.story_generator import StoryEngine  # Dot indicates same directory
from backend.core.rag_manager import NarrativeRAG
from transformers import pipeline


class NarrativePipeline:
    def __init__(self):
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    def _summarize_story(self, text, max_length=150):
        summary = self.summarizer(
            text,
            max_length=max_length,
            min_length=30,
            do_sample=False,
            truncation=True
        )
        return summary[0]['summary_text']

    
    def run(self, image_path):
        try:
            # Step 1: Image Analysis
            analyzer = MultimodalAnalyzer()
            analysis = analyzer.analyze(image_path)
            
            # Step 2: Initialize RAG
            rag = NarrativeRAG()
            rag.index_context(
                document=analysis.json(),
                metadata={"type": "initial_analysis"}
            )

            # Step 3: Generate Story
            story = []
            summary = "Initial story setup"
            for chapter_num in range(1, 6):
                context = {
                    "current_chapter": chapter_num,
                    "summary": summary,  # This is the correct key
                    "mood": analysis.mood,
                    "conflicts": analysis.potential_conflicts
                }
                
                chapter = StoryEngine().generate_chapter(context)
                story.append(chapter)
                
                # Update summary every 3 chapters instead of 5
                if chapter_num % 3 == 0:
                    summary = self._summarize_story(story[-3:])
                chapter = generate_chapter_with_retry(StoryEngine(), context)
  
            return story
            # In NarrativePipeline.run()
        except Exception as error:
            # Add logging for diagnostics
            print(f"Pipeline Error: {str(error)}")
            print(f"Error Type: {type(error).__name__}")
            raise RuntimeError(f"Pipeline failed: {str(error)}") from error
        
        
        
def _summarize_story(self, chapters):
        try:
            summary_prompt = "Summarize this story arc in 3 sentences:"
            return ollama.generate(
                model="gemma2:27b",
                prompt=summary_prompt + "\n".join(chapters),
                stream=False
            )
        except Exception as error:
            print(f"Summarization failed: {str(error)}")
            return "Summary unavailable"
        
        
def generate_chapter_with_retry(engine, context, retries=3):
    for attempt in range(retries):
        try:
            chapter = engine.generate_chapter(context)
            return chapter
        except ValueError as e:
            if "too short" in str(e) and attempt < retries-1:
                context["summary"] += "\n[Previous attempt was too short]"
                continue
            raise
    return None

