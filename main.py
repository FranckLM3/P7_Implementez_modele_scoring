from fastapi import FastAPI
from fastapi.responses import JSONResponse
from credit_scorer_object import credit_scorer
from pydantic import BaseModel


#create the application
app = FastAPI(
    title = "Credit Score API",
    version = 1.0,
    description = "Simple API to make predict cluster of Olist client."
)

#creating the classifier

scorer = credit_scorer('preprocessor', 'classifier')

#Model
class Customer(BaseModel):
    id: int

@app.post("/",tags = ["credit_score"])
def get_prediction(client_id:Customer):

    features = scorer.transfrom(client_id.dict())
    pred = scorer.make_prediction(features)

    return JSONResponse({"Credit score":pred})