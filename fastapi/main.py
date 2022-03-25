from fastapi import FastAPI
app = FastAPI()


# main.py
from fastapi import FastAPI
app = FastAPI()

@app.get("/")
def hello():
    return {"message":"Hello World"}