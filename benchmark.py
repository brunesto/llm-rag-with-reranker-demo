# benchmark.py
from dataclasses import dataclass
import json
import re
from dbrag import *
from chatrag import *
from docparser import *
from app_choices import *

# @dataclass
# class BenchmarkedRag:
#     embedding:str
#     separators:list[str]


dbrag = DbRag()
# dbrag.config.embedding = "ollama/nomic-embed-text:latest"
dbrag.config.embedding=("seznam", "Seznam/retromae-small-cs")
#dbrag.config.embedding="seznam/Seznam/simcse-dist-mpnet-czeng-cs-en" # warning not for commercial use
dbrag.config.collection_name = "bcz"




def populate():

    # pdftk ~/bruno/work/rinkai/material/docs/201015\ --\ Rinkai\ Routing\ -\ CZ.pdf cat 6-10 output  rinkai-cz.p6-10.pdf
    fileName="/home/bc2/bruno/work/rinkai/material/docs/rinkai-cz.p6-10.pdf"
    # fileName = (
    #     "/home/bc2/bruno/work/rinkai/material/docs/201015 -- Rinkai Routing - CZ.pdf"
    # )
    dbrag.splitAndStore(fileName)


populate()
print("chunks in db:", dbrag.db_total_chunks())



def ask(prompt, chats):
    print("--user prompt --")
    print(prompt)
    print("-- query chromadb --")
    results = dbrag.query_collection(prompt)
    print(results)
    print("-- rerank --")
    context = results.get("documents")[0]
    relevant_text, relevant_text_scores = dbrag.re_rank_cross_encoders(prompt, context)
    print(relevant_text_scores)
    if (chats>0):
        prompts = chatrag.format_prompts(
            context=relevant_text,
            prompt=prompt,
            system_prompt=chatrag.config.original_system_prompt,
        )
        print("--- prompts --")
        print(prompts)
        for i in range(0, chats):
            print("--- answer--")
            generator = chatrag.call_llm(prompts)
            # grab all values from the generator and concatenate them
            retVal = "".join(list(generator))
            print(retVal)
        print("------------")
        return retVal
    else:
        return []




chatrag = ChatRag()
chatrag.config.model = "llama3.2:3b"
# chatrag.config.model="qwen2"
chatrag.config.original_system_prompt = SYSTEM_PROMPT_EN

# ollama_ef = OllamaEmbeddingFunction(url="http://localhost:11434/api/embeddings", model_name='nomic-embed-text:latest')
# texts = ["Hello, world!", "How are you?"]
# embeddings = ollama_ef(texts)

ask("jak uložit auta?", 0)


# BenchmarkedRag("nomic-embed-text:latest",["."])

# rag=Rag()
# rag.chromadbpath="benchmark"
# rag.text_cleaner=lambda self,text:text
