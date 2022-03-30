from fastapi import FastAPI
from fastapi.responses import JSONResponse
from credit_scorer_object import credit_scorer
from pydantic import BaseModel
import pandas as pd
from lightgbm import LGBMClassifier

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

df = pd.read_csv('data/application_train_sample.csv',
                            engine='pyarrow',
                            verbose=False,
                            encoding='ISO-8859-1',
                            )

@app.post("/",tags = ["credit_score"])
def get_prediction(client_id:Customer):

    features = scorer.transfrom(df, client_id.dict())
    pred = scorer.make_prediction(features)

    return JSONResponse({"Credit score":pred})