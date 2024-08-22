import numpy as np
import tensorflow as tf 
import logging


def find_path(url):
    if url == '':
        return ''
    url = url.replace("-/-", "-")
    url_split = url.replace("https://", "")
    url_split = url_split.replace("www.", "")
    url_split = url_split.strip()
    url = url.replace("//", "/")
    url = url.replace("https/timesofindia-indiatimes-com", "")
    url_split = url_split.split("/")
    url_split = [u for u in url_split if (u != "") and
                                         (u != "articleshow") and
                                         (u.find(".cms")==-1) and
                                         (u.find(".ece")==-1) and
                                         (u.find(".htm")==-1) and
                                         (len(u.split('-')) <= 5) and
                                         (u.find(" ") == -1)
                ]
    if len(url_split) > 2:
        url_split = "/".join(url_split[1:])
    else:
        if len(url_split) > 0:
            url_split = url_split[-1]
        else:
            url_split = '-'
    return url_split


async def parse_prediction(tflite_pred, label_encoder):
    tflite_pred_argmax = np.argmax(tflite_pred, axis=1)
    tflite_pred_label = label_encoder.inverse_transform(tflite_pred_argmax)
    tflite_pred_prob = np.max(tflite_pred, axis=1)
    return tflite_pred_label, tflite_pred_prob
    

async def model_inference(text: list, calibrated_model, label_encoder):
    logging.info('Entering news_classifier.model_inference()')
    
    logging.info(f'Samples to predict: {len(text)}')
    if text != "":
        tflite_pred = calibrated_model.predict_proba(text)
        tflite_pred = await parse_prediction(tflite_pred, label_encoder)
    logging.info('Exiting news_classifier.model_inference()')
    return tflite_pred


async def predict_news_classes(urls: list, texts: list, calibrated_model, label_encoder):
    url_paths = [*map(find_path, urls)]
    paths_texts = [f"{p}. {t}" for p, t in zip(url_paths, texts)]
    label, prob = await model_inference(paths_texts, calibrated_model, label_encoder)
    return label, prob
