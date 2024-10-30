from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from PIL import Image
import io
from dishify import FoodImageClassifier

app = FastAPI()
classifier = FoodImageClassifier()
templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")

def load_image(file: UploadFile) -> Image.Image:
    """
    Helper function to load an image from an uploaded file.
    """
    image = Image.open(io.BytesIO(file.file.read()))
    return image

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Serve the HTML upload page.
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/generate_recipe")
async def generate_recipe(file: UploadFile = File(...)):
    """
    Endpoint to upload an image and generate a recipe based on the classified items.

    Parameters:
    - file: Image file to classify and generate a recipe from

    Returns:
    - A JSON response containing the generated recipe.
    """
    try:
        # Load the uploaded image
        image = load_image(file)

        # Classify the image
        classified_items = classifier.classify_images([image])

        # Generate the recipe based on classified items
        recipe = classifier.generate_recipe(classified_items)

        return JSONResponse(content={"recipe": recipe})

    except Exception as e:
        return JSONResponse(status_code=400, content={"message": str(e)})
