import cloudpickle
import os
import tensorflow as tf
from scraper import scrape_text
from fastapi import FastAPI, Response, Request
from typing import List, Dict
from pydantic import BaseModel, Field
from fastapi.exceptions import RequestValidationError
import uvicorn
import json
import logging
import multiprocessing
from news_classifier import predict_news_classes


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_USE_LEGACY_KERAS"] = "1"


def load_model():
    logging.warning('Entering load transformer')
    with open("classification_models/label_encoder.bin", "rb") as model_file_obj:
        label_encoder = cloudpickle.load(model_file_obj)

    with open("classification_models/calibrated_model.bin", "rb") as model_file_obj:
        calibrated_model = cloudpickle.load(model_file_obj)

    tflite_model_path = os.path.join("classification_models", "model.tflite")
    calibrated_model.estimator.tflite_model_path = tflite_model_path
    logging.warning('Exiting load transformer')
    return calibrated_model, label_encoder


async def scrape_urls(urls):
    logging.warning('Entering scrape_urls()')
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    
    results = []
    for url in urls:
        f = pool.apply_async(scrape_text, [url]) # asynchronously scraping text
        results.append(f) # appending result to results
        
    scraped_texts = []
    scrape_errors = []
    for f in results:
        t, e = f.get(timeout=120)
        scraped_texts.append(t)
        scrape_errors.append(e)
    pool.close()
    pool.join()
    logging.warning('Exiting scrape_urls()')
    return scraped_texts, scrape_errors


description = '''API to classify news articles into categories from their URLs.\n
Categories = ASTROLOGY, BUSINESS, EDUCATION, ENTERTAINMENT, HEALTH, NATION, SCIENCE, SPORTS, TECHNOLOGY, WEATHER, WORLD'''
app = FastAPI(title='News Classifier API',
              description=description, 
              version="0.0.1",
              contact={
                  "name": "Author: KSV Muralidhar",
                  "url": "https://ksvmuralidhar.in"
              }, 
             license_info={
                 "name": "License: MIT",
                 "identifier": "MIT"
             },
             swagger_ui_parameters={"defaultModelsExpandDepth": -1})


class URLList(BaseModel):
    urls: List[str] = Field(..., description="List of URLs of news articles to classify")
    key: str = Field(..., description="Authentication Key")

class Categories(BaseModel):
    label: str = Field(..., description="category label")
    calibrated_prediction_proba: float = Field(..., 
                                               description="calibrated prediction probability (confidence)")

class SuccessfulResponse(BaseModel):
    urls: List[str] = Field(..., description="List of URLs of news articles inputted by the user")
    scraped_texts: List[str] = Field(..., description="List of scraped text from input URLs")
    scrape_errors: List[str] = Field(..., description="List of errors raised during scraping. One item for corresponding URL")
    category: Categories = Field(..., description="Dict of category label of news articles along with calibrated prediction_proba")
    classifier_error: str = Field("", description="Empty string as the response code is 200")

class AuthenticationError(BaseModel):
    urls: List[str] = Field(..., description="List of URLs of news articles inputted by the user")
    scraped_texts: str = Field("", description="Empty string as authentication failed")
    scrape_errors: str = Field("", description="Empty string as authentication failed")
    category: str = Field("", description="Empty string as authentication failed")
    classifier_error: str = Field("Error: Authentication error: Invalid API key.")

class ClassifierError(BaseModel):
    urls: List[str] = Field(..., description="List of URLs of news articles inputted by the user")
    scraped_texts: List[str] = Field(..., description="List of scraped text from input URLs")
    scrape_errors: List[str] = Field(..., description="List of errors raised during scraping. One item for corresponding URL")
    category: str = Field("", description="Empty string as classifier encountered an error")
    classifier_error: str = Field("Error: Classifier Error with a message describing the error")

class InputValidationError(BaseModel):
    urls: List[str] = Field(..., description="List of URLs of news articles inputted by the user")
    scraped_texts: str = Field("", description="Empty string as validation failed")
    scrape_errors: str = Field("", description="Empty string as validation failed")
    category: str = Field("", description="Empty string as validation failed")
    classifier_error: str = Field("Validation Error with a message describing the error")


class NewsClassifierAPIAuthenticationError(Exception):
    pass 

class NewsClassifierAPIScrapingError(Exception):
    pass 


def authenticate_key(api_key: str):
    if api_key != os.getenv('API_KEY'):
        raise NewsClassifierAPIAuthenticationError("Authentication error: Invalid API key.")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    urls = request.query_params.getlist("urls")
    error_details = exc.errors()
    error_messages = []
    for error in error_details:
        loc = [*map(str, error['loc'])][-1]
        msg = error['msg']
        error_messages.append(f"{loc}: {msg}")
    error_message = "; ".join(error_messages) if error_messages else ""
    response_json = {'urls': urls, 'scraped_texts': '', 'scrape_errors': '', 'categories': "", 'classifier_error': f'Validation Error: {error_message}'}
    json_str = json.dumps(response_json, indent=5) # convert dict to JSON str
    return Response(content=json_str, media_type='application/json', status_code=422)


calibrated_model, label_encoder = load_model()

@app.post("/classify/", tags=["Classify"], response_model=List[SuccessfulResponse],
         responses={
        401: {"model": AuthenticationError, "description": "Authentication Error: Returned when the entered API key is incorrect"}, 
        500: {"model": ClassifierError, "description": "Classifier Error: Returned when the API couldn't classify even a single article"},
        422: {"model": InputValidationError, "description": "Validation Error: Returned when the payload data doesn't satisfy the data type requirements"}
         })
async def classify(q: URLList):
    """
    Get categories of news articles by passing the list of URLs as input.
    - **urls**: List of URLs (required)
    - **key**: Authentication key (required)
    """
    try:
        logging.warning("Entering classify()")
        urls = ""
        scraped_texts = ""
        scrape_errors = ""
        labels = ""
        probs = 0
        request_json = q.json()
        request_json = json.loads(request_json)
        urls = request_json['urls']
        api_key = request_json['key']
        _ = authenticate_key(api_key)
        scraped_texts, scrape_errors = await scrape_urls(urls)
        
        unique_scraped_texts = [*set(scraped_texts)]
        if (unique_scraped_texts[0] == "") and (len(unique_scraped_texts) == 1):
            raise NewsClassifierAPIScrapingError("Scrape Error: Couldn't scrape text from any of the URLs")
            
        labels, probs = await predict_news_classes(urls, scraped_texts, calibrated_model, label_encoder)
        label_prob = [{"label": "", "calibrated_prediction_proba": 0} 
                      if t == "" else {"label": l, "calibrated_prediction_proba": p} 
                      for l, p, t in zip(labels, probs, scraped_texts)]
        status_code = 200
        response_json = {'urls': urls, 'scraped_texts': scraped_texts, 'scrape_errors': scrape_errors, 'categories': label_prob, 'classifer_error': ''}
    except Exception as e:
        status_code = 500
        if e.__class__.__name__ == "NewsClassifierAPIAuthenticationError":
            status_code = 401
        response_json = {'urls': urls, 'scraped_texts': scraped_texts, 'scrape_errors': scrape_errors, 'categories': "", 'classifier_error': f'Error: {e}'}

    json_str = json.dumps(response_json, indent=5) # convert dict to JSON str
    return Response(content=json_str, media_type='application/json', status_code=status_code)


if __name__ == '__main__':
    uvicorn.run(app=app, host='0.0.0.0', port=7860, workers=3)
