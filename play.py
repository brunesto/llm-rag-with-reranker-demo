
import json  

from rag import *

#----------------------------------------------------------------
#uploaded_file = open("/home/bc2/bruno/work/rinkai/material/docs/201015 -- Rinkai Routing - EN.pdf", "rb")
#rag.splitAndStore(uploaded_file)


#results = query_collection("how to increase a vehicle capacity?")




import torch
from transformers import AutoModel, AutoTokenizer


class SeznamEmbeddings:

  def __init__(self,model_name:str):    
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.model = AutoModel.from_pretrained(model_name)


  def embeddings(self,texts):
     # Tokenize the input texts
     batch_dict = self.tokenizer(texts, max_length=128, padding=True, truncation=True, return_tensors='pt')

     outputs =self.model(**batch_dict)
     embeddings = outputs.last_hidden_state[:, 0] # CLS
     return embeddings

  #similarity = torch.nn.functional.cosine_similarity(embeddings[0], embeddings[1], dim=0)
  #print(embeddings)


  input_texts = [
    "Dnes je výborné počasí na procházku po parku.",
    "Večer si oblíbím dobrý film a uvařím si čaj."
]


seznam=SeznamEmbeddings("Seznam/retromae-small-cs")
