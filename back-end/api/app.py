# #LOGGING
# import subprocess
# import json
# from datetime import datetime


# def get_volume_creation_datetime(volume_name):
#     command = f"docker volume inspect {volume_name}"
#     output = subprocess.check_output(command, shell=True)
#     volume_info = json.loads(output)[0]
#     creation_datetime_str = volume_info["CreatedAt"]
#     creation_datetime = datetime.strptime(creation_datetime_str, "%Y-%m-%dT%H:%M:%SZ")
#     return creation_datetime

# volume_name = "0aa3831033443e68bfcb33017ec8c7122c0cd22cce87e3f0aa126c1c8367b1a1"
# volume_creation_datetitme = get_volume_creation_datetime(volume_name)
# print(f"Datetime of Volume: '{volume_name}': {volume_creation_datetitme}")

#MAIN
import torch
import numpy as np
from torch.nn import functional as F
from fastapi import FastAPI
from transformers import BertTokenizer, BertForSequenceClassification
from fastapi.encoders import jsonable_encoder

MODEL_NAME = 'bert-base-uncased'
app = FastAPI()
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME, force_download=True, resume_download=False)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, force_download=True, resume_download=False)

@app.get("/")
def root():
    return {"Hello": "World!"}

@app.post("/predict/")
def predict(text: str):
    input = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    
    with torch.no_grad():
        output = model(**input)
        probabilites = F.softmax(output.logits, dim=1)
        predicted_class = torch.argmax(probabilites).item()
        print("What are you? Probabilities", probabilites)
        print("And what are you? Predicted_Class", predicted_class)
    response = {
        "predicted_class": np.array(predicted_class),
        "probailities": np.array(probabilites)
    }
    print("Dir Response: ", dir(response))
    print("Response: ", response)
    return jsonable_encoder(response)
