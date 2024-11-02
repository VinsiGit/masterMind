from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import List, Optional
from PIL import Image
import io
from dishify import FoodImageClassifier

app = FastAPI()
classifier = FoodImageClassifier()

def load_images(files: List[UploadFile]) -> List[Image.Image]:
    """Load multiple images from uploaded files."""
    images = []
    for file in files:
        image = Image.open(io.BytesIO(file.file.read()))
        images.append(image)
    return images

@app.post("/api/generate_recipe")
async def generate_recipe(files: List[UploadFile] = File(...), text: Optional[str] = Form(None)):
    """
    Endpoint to upload multiple images and generate a recipe based on the classified items.

    Parameters:
    - files: List of image files to classify and generate a recipe from
    - text: Additional text for preferences or dietary restrictions

    Returns:
    - A JSON response containing the generated recipe.
    """
    # Load the uploaded images
    images = load_images(files)

    # Classify each image
    classified_items = classifier.classify_images(images)

    # Generate the recipe based on classified items and additional text
    recipe = classifier.generate_recipe(classified_items, text)

    return JSONResponse(content={"recipe": recipe})
