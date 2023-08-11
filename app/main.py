from fastapi import FastAPI,Request
from app.hog import gethog
import base64
import numpy as np
import cv2

app = FastAPI()

def readb64(url):
    encode_data = url.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encode_data),np.uint8)
    img = cv2.imdecode(nparr,cv2.IMREAD_GRAYSCALE)
    return img
@app.get('/api/gethog')
async def read_str(request: Request):
    item = await request.json()
    item_str = item['img']
    img = readb64(item_str)
    hog = gethog(img)
    return {"hog":hog.tolist()}
