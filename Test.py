from flask import Flask, request
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import RobertaConfig
from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig
import torch
from torch import cuda
import os
import re

app = Flask(__name__)

# config = RobertaConfig.from_pretrained("PirateXX/ChatGPT-Text-Detector", use_auth_token= ACCESS_TOKEN)
# model = RobertaForSequenceClassification.from_pretrained("PirateXX/ChatGPT-Text-Detector", use_auth_token= ACCESS_TOKEN, config = config)

device = 'cuda' if cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained("PirateXX/AI-Content-Detector")
model = AutoModelForSequenceClassification.from_pretrained("PirateXX/AI-Content-Detector")
model.to(device)


# model_name = "roberta-base"
# tokenizer = RobertaTokenizer.from_pretrained(model_name, map_location=torch.device('cpu'))

def text_to_sentences(text):
    clean_text = text.replace('\n', ' ')
    return re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', clean_text)


# function to concatenate sentences into chunks of size 900 or less
def chunks_of_900(text, chunk_size=900):
    sentences = text_to_sentences(text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk + sentence) <= chunk_size:
            if len(current_chunk) != 0:
                current_chunk += " " + sentence
            else:
                current_chunk += sentence
        else:
            chunks.append(current_chunk)
            current_chunk = sentence
    chunks.append(current_chunk)
    return chunks


def predict(query):
    tokens = tokenizer.encode(query)
    all_tokens = len(tokens)
    tokens = tokens[:tokenizer.model_max_length - 2]
    used_tokens = len(tokens)
    tokens = torch.tensor([tokenizer.bos_token_id] + tokens + [tokenizer.eos_token_id]).unsqueeze(0)
    mask = torch.ones_like(tokens)

    with torch.no_grad():
        logits = model(tokens.to(device), attention_mask=mask.to(device))[0]
        probs = logits.softmax(dim=-1)

    fake, real = probs.detach().cpu().flatten().numpy().tolist()
    return real


def findRealProb(text):
    chunksOfText = (chunks_of_900(text))
    results = []
    for chunk in chunksOfText:
        output = predict(chunk)
        results.append([output, len(chunk)])

    ans = 0
    cnt = 0
    for prob, length in results:
        cnt += length
        ans = ans + prob * length
    realProb = ans / cnt
    return {"Real": realProb, "Fake": 1 - realProb}, results
