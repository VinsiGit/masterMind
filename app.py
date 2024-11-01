from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io
from dishify import FoodImageClassifier

app = FastAPI()
classifier = FoodImageClassifier()


def load_image(file: UploadFile) -> Image.Image:
    """
    Helper function to load an image from an uploaded file and save it to a directory called 'debug'.
    """
    # Load the image
    image = Image.open(io.BytesIO(file.file.read()))

    return image

@app.post("/api/generate_recipe")
async def generate_recipe(file: UploadFile = File(...)):
    """
    Endpoint to upload an image and generate a recipe based on the classified items.

    Parameters:
    - file: Image file to classify and generate a recipe from

    Returns:
    - A JSON response containing the generated recipe.
    """
    # Load the uploaded image
    image = load_image(file)

    # Classify the image
    classified_items = classifier.classify_images([image])

    # Generate the recipe based on classified items
    recipe = classifier.generate_recipe(classified_items)

    return JSONResponse(content={"recipe": recipe})