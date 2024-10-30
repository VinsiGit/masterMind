import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor
from torchvision import transforms
from PIL import Image
import ollama
import markdown2  # Make sure to install this library using pip


class FoodImageClassifier:
    CONFIDENCE_THRESHOLD = 0.5

    def __init__(self, model_name='llama3.2:1b'):
        self.model_name = model_name

        self.category_processor = AutoImageProcessor.from_pretrained("Kaludi/food-category-classification-v2.0")
        self.category_model = AutoModelForImageClassification.from_pretrained(
            "Kaludi/food-category-classification-v2.0")

        self.fruit_veg_model = AutoModelForImageClassification.from_pretrained(
            "jazzmacedo/fruits-and-vegetables-detector-36")

        self.category_to_model = {
            "Fruit": self.fruit_veg_model,
            "Vegetable": self.fruit_veg_model
        }

        self.preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])

    def _validate_image(self, image: Image.Image) -> Image.Image:
        if image.format != 'JPEG':
            image = image.convert("RGB")
            image.save("temp_image.jpg", "JPEG")
            image = Image.open("temp_image.jpg")
        return image

    def classify_images(self, images: list[Image.Image]) -> list[tuple]:
        validated_images = [self._validate_image(img) for img in images]
        preprocessed_images = [self.preprocess(img).unsqueeze(0) for img in validated_images]
        batch = torch.cat(preprocessed_images, dim=0)

        results = []
        with torch.no_grad():
            category_outputs = self.category_model(batch)

        for idx, category_logits in enumerate(category_outputs.logits):
            category_idx = torch.argmax(category_logits).item()
            category_label = self.category_model.config.id2label[category_idx]

            item_model = self.category_to_model.get(category_label)
            if item_model is None:
                results.append((category_label, None))
                continue

            with torch.no_grad():
                item_logits = item_model(batch[idx].unsqueeze(0))
                item_probs = torch.softmax(item_logits.logits, dim=1)
                item_confidence, item_idx = torch.max(item_probs, dim=1)
                item_confidence = item_confidence.item()
                item_idx = item_idx.item()
                item_label = item_model.config.id2label[item_idx]

            if item_confidence >= self.CONFIDENCE_THRESHOLD:
                results.append((category_label, item_label))
            else:
                results.append((category_label, None))

        return results

    def generate_recipe(self, classified_items: list[tuple]) -> str:
        ingredients = [item if item else category for category, item in classified_items]
        ingredient_list = ", ".join(ingredients)

        response = ollama.chat(model=self.model_name, messages=[
            {
                'role': "user",
                'content': f"Create a unique recipe using the following ingredients: {ingredient_list}."
            },
        ])

        # Convert markdown response to HTML
        markdown_content = response['message']['content']
        html_content = markdown2.markdown(markdown_content)

        # Replace new line characters with <br>
        html_content = html_content.replace('\n', '<br>')

        # Wrap in a div
        wrapped_html = f'<div>{html_content}</div>'

        return wrapped_html

