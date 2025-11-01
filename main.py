from fastapi import FastAPI, Body
from PIL import Image
import base64, io
import torch
from transformers import CLIPProcessor, CLIPModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # allow any origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

rock_labels = ["granite","basalt","limestone","sandstone","obsidian","marble","slate","gneiss"]

@app.post("/api")
def classify_image(img_base64: str = Body()):
    # convert base64 -> PIL Image
    img_bytes = base64.b64decode(img_base64)
    image = Image.open(io.BytesIO(img_bytes))

    # CLIP processing
    inputs = processor(
        text=rock_labels,
        images=image,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1).tolist()[0]

    # top 5 simple beginner sorting
    pairs = list(zip(rock_labels, probs))
    pairs.sort(key=lambda x: x[1], reverse=True)
    top5 = pairs[:5]

    return {"top5": [{"label": p[0], "score": float(p[1])} for p in top5]}