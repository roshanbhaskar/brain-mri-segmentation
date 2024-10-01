from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from brain_segmentation import BrainSegmenter
import io
from starlette.responses import StreamingResponse

app = FastAPI()
segmenter = BrainSegmenter()

@app.post("/segment/")
async def segment_brain(file: UploadFile = File(...)):

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    mask = segmenter.segment(image)
    
    # Encode result
    _, encoded_mask = cv2.imencode('.png', mask)
    
    return StreamingResponse(io.BytesIO(encoded_mask.tobytes()), media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)