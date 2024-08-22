from sklearn.dummy import DummyClassifier
from tqdm import tqdm
import multiprocessing
import numpy as np
import tensorflow as tf
from transformers import DistilBertTokenizerFast



class PredictProba(DummyClassifier):
    def __init__(self, tflite_model_path: str, classes_: list, n_tokens: int):
        self.classes_ = classes_ # required attribute for an estimator to be used in calibration classifier
        self.n_tokens = n_tokens
        self.tflite_model_path = tflite_model_path


    def fit(self, x, y):
        print('called fit')
        return self # fit method is required for an estimator to be used in calibration classifier

    @staticmethod
    def get_token_batches(attention_mask, input_ids, batch_size: int=8):
        n_texts = len(attention_mask)
        n_batches = int(np.ceil(n_texts / batch_size))
        if n_texts <= batch_size:
            n_batches = 1

        attention_mask_batches = []
        input_ids_batches = []

        for i in range(n_batches):
            if i != n_batches-1:
                attention_mask_batches.append(attention_mask[i*batch_size: batch_size*(i+1)])
                input_ids_batches.append(input_ids[i*batch_size: batch_size*(i+1)])
            else:
                attention_mask_batches.append(attention_mask[i*batch_size:])
                input_ids_batches.append(input_ids[i*batch_size:])
        
        return attention_mask_batches, input_ids_batches
        
    
    def get_batch_inference(self, batch_size, attention_mask, input_ids):
        interpreter = tf.lite.Interpreter(model_path=self.tflite_model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()[0]
        interpreter.resize_tensor_input(input_details[0]['index'],[batch_size, self.n_tokens])
        interpreter.resize_tensor_input(input_details[1]['index'],[batch_size, self.n_tokens])
        interpreter.resize_tensor_input(output_details['index'],[batch_size, len(self.classes_)])
        interpreter.allocate_tensors()
        interpreter.set_tensor(input_details[0]["index"], attention_mask)
        interpreter.set_tensor(input_details[1]["index"], input_ids)
        interpreter.invoke()
        tflite_pred = interpreter.get_tensor(output_details["index"])
        return tflite_pred
    
    def inference(self, texts):
        model_checkpoint = "distilbert-base-uncased"
        tokenizer = DistilBertTokenizerFast.from_pretrained(model_checkpoint)
        tokens = tokenizer(texts, max_length=self.n_tokens, padding="max_length", 
                           truncation=True, return_tensors="tf")
        attention_mask, input_ids = tokens['attention_mask'], tokens['input_ids']
        attention_mask_batches, input_ids_batches = self.get_token_batches(attention_mask, input_ids)
        
        
        
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        results = []
        for attention_mask, input_ids in zip(attention_mask_batches, input_ids_batches):
            f = pool.apply_async(self.get_batch_inference, args=(len(attention_mask), attention_mask, input_ids))
            results.append(f)
        
        all_predictions = np.array([])
        for n_batch in tqdm(range(len(results))):
            tflite_pred = results[n_batch].get(timeout=360)
            if n_batch == 0:
                all_predictions = tflite_pred
            else:
                all_predictions = np.concatenate((all_predictions, tflite_pred), axis=0)
        return all_predictions

    def predict_proba(self, X, y=None):
        predict_prob = self.inference(X)
        return predict_prob
    