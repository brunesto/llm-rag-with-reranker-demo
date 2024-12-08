# benchmark.py
from dataclasses import dataclass
import json  
import re
from dbrag import *
from chatrag import *
from docparser import *

# @dataclass
# class BenchmarkedRag:
#     embedding:str
#     separators:list[str]
    
# rules=[
#     (r'^\d+\/70 ',"","remove page number"),
#     (r'\n *\n',"\n","simplify newlines"),
#     (r' +'," ","merge spacing"),
#     (r'\n[ #]*(\d+.)+ ([^\n]*)',"\nodstavec \\2:","make title explicit")
#     ]

dbrag=DbRag()
dbrag.config.embedding="ollama/nomic-embed-text:latest"
#dbrag.config.embedding=("seznam", "Seznam/retromae-small-cs")
#dbrag.config.embedding="seznam/Seznam/simcse-dist-mpnet-czeng-cs-en" # warning not for commercial use
dbrag.config.collection_name="ben"

chatrag=ChatRag()
chatrag.config.model="llama3.2:3b"
# chatrag.config.model="qwen2"
chatrag.config.original_system_prompt=SYSTEM_PROMPT_EN;

# ollama_ef = OllamaEmbeddingFunction(url="http://localhost:11434/api/embeddings", model_name='nomic-embed-text:latest')
# texts = ["Hello, world!", "How are you?"]
# embeddings = ollama_ef(texts)

def populate():

   

    # pdftk ~/bruno/work/rinkai/material/docs/201015\ --\ Rinkai\ Routing\ -\ CZ.pdf cat 6-10 output  rinkai-cz.p6-10.pdf
    #fileName="/home/bc2/bruno/work/github/brunesto/llm-rag-with-reranker-demo/rinkai-cz.p6-10.pdf"
    fileName="/home/bc2/bruno/work/rinkai/material/docs/201015 -- Rinkai Routing - EN.pdf"

    basename=sanitizedBaseName(fileName);
    filesInDb=list(dbrag.db_stats()["files"])
    if basename in map(lambda x:x[0],filesInDb):
        print ("already in db ",fileName)
        return


    # with open(fileName,"rb") as file:
    dbrag.splitAndStore(fileName)
        # chunks=parser.process_document(file,"doc")
        #    rag.add_to_vector_collection(all_splits,"doc")
    # with open(fileName,"rb") as file:
    #     chunks=parser.process_documentOLD(file,"doc")   
    #     dbrag.add_to_vector_collection(chunks,"doc")



def ask(prompt,chats):
    print("--user prompt --")
    print(prompt)
    print("-- query chromadb --")
    results = dbrag.query_collection(prompt)
    print(results)    
    print("-- rerank --")
    context = results.get("documents")[0]
    relevant_text, relevant_text_ids = dbrag.re_rank_cross_encoders(prompt,context)
    print(relevant_text_ids)    
    prompts=chatrag.format_prompts(context=relevant_text, prompt=prompt,system_prompt=chatrag.config.original_system_prompt)
    print("--- prompts --")
    print(prompts)
    for i in range(0,chats):
       print("--- answer--")
       generator = chatrag.call_llm(prompts)
       # grab all values from the generator and concatenate them
       retVal=''.join(list(generator))
       print(retVal)
    print("------------")
    return retVal


populate()
print("chunks in db:",dbrag.db_stats())

ask("how to save a car?",3)


# BenchmarkedRag("nomic-embed-text:latest",["."])

#rag=Rag()
# rag.chromadbpath="benchmark"
# rag.text_cleaner=lambda self,text:text
 




