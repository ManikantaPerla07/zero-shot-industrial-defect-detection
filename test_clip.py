import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image = Image.new("RGB", (224, 224), "gray")

texts = [
    "a normal industrial product",
    "a defective industrial product"
]

inputs = processor(
    text=texts,
    images=image,
    return_tensors="pt",
    padding=True
)

with torch.inference_mode():
    outputs = model(**inputs)

probs = outputs.logits_per_image.softmax(dim=1)

print("Image shape:", image.size)
print("Probabilities:", probs[0].tolist())
print("Predicted:", texts[probs.argmax().item()])
print("CLIP inference: OK")
