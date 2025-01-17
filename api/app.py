from fastapi import FastAPI

app = FastAPI()


@app.get(path="/")
def hello():
    return {"msg": "hello!"}
