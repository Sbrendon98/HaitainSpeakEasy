
import openai
import os
import json
from dotenv import load_dotenv
from fastapi import FastAPI

app = FastAPI()
load_dotenv()
openai.api_key = os.getenv('OPENAI_KEY')
instruct_prompt = "You are a Haitian Teacher, your goal is to have a conversation with me in Creole. The conversation could be about anything like we would normally do as ChatGPT. However, every response you make, most be in stringified json format where you send both the english and haitian translations."

@app.get("/")
def root():
    return {"Hello": "World!"}

@app.post("/predict/")
def get_haitain_predict(text: str):
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            {
                'role': "system",
                'content': instruct_prompt
            },
            {
                'role': 'user',
                'content': text,
                
            }
        ],
        temperature=0,
        #stream=True
    )
    response_content = response["choices"][0]["message"]["content"].splitlines()
    concat_content = ''
    for char in response_content:
        concat_content += char
    jsonified_content = json.loads(concat_content)
    return jsonified_content