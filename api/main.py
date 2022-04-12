from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from api.credit_scorer_object import credit_scorer
from pydantic import BaseModel
import pandas as pd

#create the application
app = FastAPI(
    title = "Credit Score API",
    version = 1.0,
    description = "Simple API to make predict cluster of Olist client."
)

#creating the classifier

scorer = credit_scorer('api/pipeline', 'api/classifier')

#Model
class Customer(BaseModel):
    id: int

df = pd.read_csv('dataset_sample.csv',
                            engine='pyarrow',
                            verbose=False,
                            encoding='ISO-8859-1',
                            )

@app.post("/",tags = ["credit_score"])
def get_prediction(client_id:Customer):

    if client_id.dict()['id'] not in df['SK_ID_CURR'].unique():
        raise HTTPException(
            status_code=404, detail=f"Client ID {client_id.dict()['id']} not found")

    features = scorer.transfrom(df, client_id.dict())
    prob, info_default = scorer.make_prediction(features)

    return JSONResponse({"Credit score":round(prob[0], 3),
                         "Advice": info_default})
