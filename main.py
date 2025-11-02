from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS MUST BE RIGHT AFTER app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# NOW import heavy modules
from PIL import Image
import base64, io
import torch
from transformers import CLIPProcessor, CLIPModel


# GLOBALS (empty)
model = None
processor = None

# your labels
labels = ["granite", "basalt", "limestone", "sandstone", "obsidian", "marble", "slate", "gneiss"]

def get_clip():
    global model, processor
    if model is None:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    return model, processor


@app.get("/")
def home():
    return {"status":"ok"}

@app.post("/api")
def classify_image(img_base64: str = Body()):
    model, processor = get_clip()  # lazy load here

    img_bytes = base64.b64decode(img_base64)
    image = Image.open(io.BytesIO(img_bytes))

    inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1).tolist()[0]

    pairs = sorted(zip(labels, probs), key=lambda x: x[1], reverse=True)[:5]
    return {"top5": [{"label": p[0], "score": float(p[1])} for p in pairs]}