import torch
import logging
from transformers import AutoModelForImageClassification, AutoImageProcessor
from torchvision import transforms
from PIL import Image
from ollama import Client  # Import Client for Ollama interaction
import markdown2
from typing import Optional

# Set up logging
logging.basicConfig(level=logging.INFO)


class FoodImageClassifier:
    CONFIDENCE_THRESHOLD = 0.33

    def __init__(self, model_name='llama3.2:3b'):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Lazy loading for models
        self._category_processor = None
        self._category_model = None
        self._fruit_veg_model = None

        # Define preprocessing pipeline
        self.preprocess = transforms.Compose([
            transforms.Lambda(self.make_square),  # Make image square
            transforms.Resize((224, 224)),  # Resize to 224x224
            transforms.ToTensor()  # Convert to tensor
        ])

        # Initialize Ollama client for Docker network connection
        self.ollama_client = Client(host="http://dishify-ollama:11434")

    def make_square(self, img: Image.Image) -> Image.Image:
        """Make the image square by adding white padding."""
        width, height = img.size
        if width > height:
            padding = (0, (width - height) // 2, 0, (width - height) // 2)  # (left, top, right, bottom)
        else:
            padding = ((height - width) // 2, 0, (height - width) // 2, 0)  # (left, top, right, bottom)

        # Create a new white background image
        padded_img = Image.new("RGB", (max(width, height), max(width, height)), (255, 255, 255))  # White background
        padded_img.paste(img, padding[0:2])  # Paste the original image onto the padded image

        return padded_img

    @property
    def category_processor(self):
        if self._category_processor is None:
            self._category_processor = AutoImageProcessor.from_pretrained("Kaludi/food-category-classification-v2.0")
        return self._category_processor

    @property
    def category_model(self):
        if self._category_model is None:
            self._category_model = AutoModelForImageClassification.from_pretrained(
                "Kaludi/food-category-classification-v2.0")
            self._category_model.to(self.device)
        return self._category_model

    @property
    def fruit_veg_model(self):
        if self._fruit_veg_model is None:
            self._fruit_veg_model = AutoModelForImageClassification.from_pretrained(
                "jazzmacedo/fruits-and-vegetables-detector-36")
            self._fruit_veg_model.to(self.device)
        return self._fruit_veg_model

    def _validate_image(self, image: Image.Image) -> Image.Image:
        """Ensures image is in JPEG format."""
        if image.format != 'JPEG':
            image = image.convert("RGB")
            image.save("temp_image.jpg", "JPEG")
            image = Image.open("temp_image.jpg")
        return image

    def preprocess_images(self, images: list[Image.Image]) -> torch.Tensor:
        """Validates and preprocesses images into a batch tensor."""
        validated_images = [self._validate_image(img) for img in images]
        preprocessed_images = [self.preprocess(img).unsqueeze(0) for img in validated_images]
        batch = torch.cat(preprocessed_images, dim=0).to(self.device)
        return batch

    def classify_images(self, images: list[Image.Image]) -> list[tuple]:
        """Classifies each image in the batch and returns labels with confidence."""
        batch = self.preprocess_images(images)
        results = []

        # Perform category classification
        with torch.no_grad():
            category_outputs = self.category_model(batch)

        # Iterate through each image and classify
        for idx, category_logits in enumerate(category_outputs.logits):
            category_idx = torch.argmax(category_logits).item()
            category_label = self.category_model.config.id2label[category_idx]

            # Assign item model based on category label
            item_model = self.fruit_veg_model if category_label in {"Fruit", "Vegetable"} else None
            if item_model is None:
                results.append((category_label, None))
                continue

            with torch.no_grad():
                item_logits = item_model(batch[idx].unsqueeze(0))
                item_probs = torch.softmax(item_logits.logits, dim=1)
                item_confidence, item_idx = torch.max(item_probs, dim=1)
                item_label = item_model.config.id2label[item_idx.item()]

            # Append to results based on confidence threshold
            if item_confidence.item() >= self.CONFIDENCE_THRESHOLD:
                results.append((category_label, item_label))
            else:
                results.append((category_label, None))

        return results

    def generate_recipe(self, classified_items: list[tuple], instructions: Optional[str] = None) -> str:
        """Generates a recipe using classified ingredients and optional user instructions."""
        ingredients = [item if item else category for category, item in classified_items]
        ingredient_list = ", ".join(ingredients)

        # Construct the prompt with user-provided instructions
        prompt = f"Create a unique recipe using the following ingredients: {ingredient_list}."
        if instructions:
            prompt += f" Please also consider the following: {instructions}"

        try:
            response = self.ollama_client.chat(model=self.model_name, messages=[
                {'role': 'user', 'content': prompt},
            ])

            if response and 'message' in response:
                markdown_content = response['message']['content']
                html_content = markdown2.markdown(markdown_content).replace('\n\n', '<br>')
                wrapped_html = f'<div>{html_content}</div>'
                return wrapped_html
            else:
                logging.warning("Failed to generate recipe: No response or response format unexpected.")
                return "<div>Failed to generate a recipe. Please try again.</div>"

        except Exception as e:
            logging.error(f"Error generating recipe: {e}")
            return "<div>Failed to generate a recipe due to an error. Please try again.</div>"
