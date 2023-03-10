"""Main module."""
import uvicorn
from typing import List
from fastapi import FastAPI, status
from api.routers import router
from api.endpoints.performance import calc_performance
from api.endpoints.aderencia import calc_aderencia

import logging
logging.basicConfig(filename='./monitoring.log', level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

app = FastAPI(title='Monitoramento de modelos', version="1.0.0")

data = {'latest-performance': {'model': '', 'value': {}}, 'latest-aderencia': {}}


@app.get("/")
async def read_root():
    """Hello World message."""
    return {"Hello World": "from FastAPI"}


@app.get("/performance/", status_code=status.HTTP_200_OK)
async def read_performance():
    return data['latest-performance']


@app.post("/performance/{model_from}", response_model=dict, status_code=status.HTTP_201_CREATED)
async def write_performance(model_from: str, json_data: List[dict]):
    
    logging.info("Starting API post request to /performance/" + model_from + ".")
    performance = calc_performance(model_from, json_data)
    logging.info("Finished POST, performance metrics successfully calculated!")

    data['latest-performance']['model'] = model_from
    data['latest-performance']['value'] = performance

    return performance


@app.get("/aderencia/", status_code=status.HTTP_200_OK)
async def read_aderencia():
    return data['latest-aderencia']


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
