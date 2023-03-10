"""Main module."""
import uvicorn
from typing import List
from fastapi import FastAPI, status
from api.routers import router
from api.endpoints.performance import calc_performance
from api.endpoints.aderencia import calc_aderencia

# Setting the logger formatting and poiting to correct file.
import logging
logging.basicConfig(filename='./monitoring.log', level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

app = FastAPI(title='Monitoramento de modelos', version="1.0.0")

data = {'latest-performance': {'model': '', 'value': {}}, 'latest-aderencia': {}}


@app.get("/")
async def read_root():
    """Hello World message."""
    return {"Hello World": "from FastAPI"}


# View latest performance POST response.
@app.get("/performance/", status_code=status.HTTP_200_OK)
async def read_performance():
    return data['latest-performance']


# The POST request calls calc_performance for determining the metrics.
# The request body is defined as a list of dictionaries and the response
# model is a dictionary with both volumetry and roc score.
@app.post("/performance/{model_from}", response_model=dict, status_code=status.HTTP_201_CREATED)
async def write_performance(model_from: str, json_data: List[dict]):
    
    logging.info("Starting API post request to /performance/" + model_from + ".")
    performance = calc_performance(model_from, json_data)
    logging.info("Finished POST, performance metrics successfully calculated!")

    data['latest-performance']['model'] = model_from
    data['latest-performance']['value'] = performance

    return performance


# View latest aderencia POST response.
@app.get("/aderencia/", status_code=status.HTTP_200_OK)
async def read_aderencia():
    return data['latest-aderencia']


# The POST request calls calc_aderencia for determining the metrics.
# The request body is defined as a dictionary, as well as the response model.
# The request one is a single key dictionary whose value is the relative path to the dataset.
# The response one has keys containing the KS test returned metrics.
@app.post("/aderencia/", response_model=dict, status_code=status.HTTP_201_CREATED)
async def write_aderencia(path: dict):

    logging.info("Starting API post request to /aderencia/")
    aderencia = calc_aderencia(path)
    logging.info("Finished POST, KStest metrics successfully calculated!")

    data['latest-aderencia'] = aderencia

    return aderencia


app.include_router(router, prefix="/v1")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
