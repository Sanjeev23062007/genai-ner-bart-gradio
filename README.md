## Development of a Named Entity Recognition (NER) Prototype Using a Fine-Tuned BART Model and Gradio Framework

### AIM:
To design and develop a prototype application for Named Entity Recognition (NER) by leveraging a fine-tuned BART model and deploying the application using the Gradio framework for user interaction and evaluation.

### PROBLEM STATEMENT:
### Design Steps

### Step 1: Import Required Libraries
First, we import the necessary Python libraries such as os, json, requests, gradio, and dotenv. These libraries help us handle environment variables, send API requests, and create the user interface. We also load the .env file to securely access the Hugging Face API key and model endpoint.

### Step 2: Create Helper Function for API Request
Next, we create a function called get_completion(). This function sends a POST request to the Hugging Face Inference API. It also includes the Authorization header using the API token so that the request can securely access the model.

### Step 3: Implement the Named Entity Recognition (NER) Function
Then we define the NER function which takes the input text from the user. This function calls the get_completion() helper function and sends the text to the NER model endpoint. After receiving the response, it processes the JSON data and extracts the named entities such as person names, locations, and organizations.

### Step 4: Merge Tokens (Optional Step)
Sometimes the model splits words into smaller parts called tokens (for example: “Cal” and “##ifornia”). To make the output easier to read, we implement a merge_tokens() function which combines these tokens into a single word like “California”.

### Step 5: Create the Gradio User Interface
Finally, we build a simple Gradio interface using gr.Interface().

The input is a textbox where users can enter the text.

The output is a highlighted text display that shows the detected entities.

We also add example sentences so users can quickly test the application.

After setting up the interface, we run the application using demo.launch(share=True), which generates a public link so others can access and test the application easily.

### PROGRAM:
```
import os
import io
from IPython.display import Image, display, HTML
from PIL import Image
import base64 
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
hf_api_key = os.environ['HF_API_KEY']
# Helper function
import requests, json

#Summarization endpoint
def get_completion(inputs, parameters=None,ENDPOINT_URL=os.environ['HF_API_SUMMARY_BASE']): 
    headers = {
      "Authorization": f"Bearer {hf_api_key}",
      "Content-Type": "application/json"
    }
    data = { "inputs": inputs }
    if parameters is not None:
        data.update({"parameters": parameters})
    response = requests.request("POST",
                                ENDPOINT_URL, headers=headers,
                                data=json.dumps(data)
                               )
    return json.loads(response.content.decode("utf-8"))
API_URL = os.environ['HF_API_NER_BASE'] #NER endpoint
text = "My name is Sanjeev, I'm building DeepLearningAI and I live in Tamilnadu"
get_completion(text, parameters=None, ENDPOINT_URL= API_URL)
import gradio as gr

def ner(input):
    output = get_completion(input, parameters=None, ENDPOINT_URL=API_URL)
    return {"text": input, "entities": output}

gr.close_all()
demo = gr.Interface(fn=ner,
                    inputs=[gr.Textbox(label="Text to find entities", lines=2)],
                    outputs=[gr.HighlightedText(label="Text with entities")],
                    title="NER with dslim/bert-base-NER",
                    description="Find entities using the `dslim/bert-base-NER` model under the hood!",
                    allow_flagging="never",
                    #Here we introduce a new tag, examples, easy to use examples for your application
                    examples=["My name is Sanjeev, I'm building DeepLearningAI and I live in Tamilnadu", "My name is Vijay and work at HuggingFace"])
demo.launch()
def merge_tokens(tokens):
    merged_tokens = []
    for token in tokens:
        if merged_tokens and token['entity'].startswith('I-') and merged_tokens[-1]['entity'].endswith(token['entity'][2:]):
            # If current token continues the entity of the last one, merge them
            last_token = merged_tokens[-1]
            last_token['word'] += token['word'].replace('##', '')
            last_token['end'] = token['end']
            last_token['score'] = (last_token['score'] + token['score']) / 2
        else:
            # Otherwise, add the token to the list
            merged_tokens.append(token)

    return merged_tokens

def ner(input):
    output = get_completion(input, parameters=None, ENDPOINT_URL=API_URL)
    merged_tokens = merge_tokens(output)
    return {"text": input, "entities": merged_tokens}

gr.close_all()
demo = gr.Interface(fn=ner,
                    inputs=[gr.Textbox(label="Text to find entities", lines=2)],
                    outputs=[gr.HighlightedText(label="Text with entities")],
                    title="NER with dslim/bert-base-NER",
                    description="Find entities using the `dslim/bert-base-NER` model under the hood!",
                    allow_flagging="never",
                    examples=["My name is Sanjeev, I'm building DeepLearningAI and I live in Tamilnadu", "My name is Vijay and work at HuggingFace"])

demo.launch(share=True, server_port=int(os.environ['PORT4']))
```
### OUTPUT:
![image alt](https://github.com/Sanjeev23062007/genai-ner-bart-gradio/blob/d7a08127634b29313b7dd663e6d9b5bf046ddabc/Screenshot%202026-03-13%20143701.png)


### RESULT:
The Named Entity Recognition (NER) prototype was successfully developed using the fine-tuned BERT model (dslim/bert-base-NER) and deployed through the Gradio interface.
The system efficiently identifies and highlights entities such as names, locations, and organizations from user-provided text input.


