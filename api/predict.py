from fastapi import FastAPI, Request
import uvicorn
# Import your model file
from main import YourModelClass

app = FastAPI()
model = YourModelClass()

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    prediction = model.predict(data)
    return {"prediction": prediction}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
