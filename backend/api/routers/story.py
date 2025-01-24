import os
import json
import tempfile
import io
from fastapi import APIRouter, File, UploadFile, HTTPException
from backend.core.pipeline import NarrativePipeline
from PIL import Image
allowed_formats = {'JPEG', 'PNG'}  # PIL uses uppercase format names


router = APIRouter()
@router.post("/generate-story")
async def generate_story(image: UploadFile = File(...)):
    temp_file_path = None
    file_ext = ".tmp"  # Default fallback extension
    try:
        # Validate MIME type
        if not image.content_type.startswith('image/'):
            raise HTTPException(400, "Invalid file type. Must be an image")

        # Read and validate content
        content = await image.read()
        if len(content) < 1024:
            raise HTTPException(400, "File too small (minimum 1KB)")
        if len(content) > 10_000_000:
            raise HTTPException(400, "File too large (maximum 10MB)")

        # Verify image content and get format
        allowed_formats = {'JPEG', 'PNG'}
        with Image.open(io.BytesIO(content)) as img:
            img.verify()
            if img.format not in allowed_formats:
                raise HTTPException(400, "Unsupported format. Use JPEG/PNG")
            file_ext = f".{img.format.lower()}"  # Now properly defined

        # Create temp file with validated extension
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            tmp_file.write(content)
            temp_file_path = tmp_file.name

        # Process pipeline
        pipeline = NarrativePipeline()
        story = pipeline.run(temp_file_path)
        return {"story": story}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Processing error: {str(e)}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)