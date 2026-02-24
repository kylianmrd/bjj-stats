from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import io

from src.model_loader import load_model
from src.preprocess import get_preprocess

app = FastAPI()

model, classes, checkpoint = load_model()
preprocess = get_preprocess(checkpoint)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    x = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    top_prob, top_idx = torch.max(probs, dim=0)

    response = {
        "predictions": [
            {"label": classes[i], "probability": float(probs[i])}
            for i in range(len(classes))
        ],
        "top_label": classes[int(top_idx)],
        "confidence": float(top_prob),
    }

    return JSONResponse(content=response)