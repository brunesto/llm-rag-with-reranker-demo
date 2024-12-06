# benchmark.py
from dataclasses import dataclass
import json  
import re
from rag import Rag
from docparser import *

@dataclass(frozen=True)
class BenchmarkedRag:
    embedding:str
    separators:list[str]
    
rules=[
    (r'^\d+\/70 ',"","remove page number"),
    (r'\n *\n',"\n","simplify newlines"),
    (r' +'," ","merge spacing"),
    (r'\n[ #]*(\d+.)+ ([^\n]*)',"\nodstavec \\2:","make title explicit")
    ]

rag=Rag()
rag.chromadbpath="bcz"


promptcz="""


"""

def populate():
    # pdftk ~/bruno/work/rinkai/material/docs/201015\ --\ Rinkai\ Routing\ -\ CZ.pdf cat 6-10 output  rinkai-cz.p6-10.pdf
    #fileName="/hom e/bc2/bruno/work/github/brunesto/llm-rag-with-reranker-demo/rinkai-cz.p6-10.pdf"
    fileName="/home/bc2/bruno/work/rinkai/material/docs/201015 -- Rinkai Routing - CZ.pdf"


    parser=DocParser()
    parser.rules=rules
    with open(fileName,"rb") as file:
        chunks=parser.process_document(file,"doc")
        #    rag.add_to_vector_collection(all_splits,"doc")
    with open(fileName,"rb") as file:
        chunks=parser.process_documentOLD(file,"doc")   
        rag.add_to_vector_collection(chunks,"doc")



def ask(prompt):
    print("--user prompt --")
    print(prompt)
    print("-- query chromadb --")
    results = rag.query_collection(prompt)
    print(results)    
    print("-- rerank --")
    context = results.get("documents")[0]
    relevant_text, relevant_text_ids = rag.re_rank_cross_encoders(prompt,context)
    print(relevant_text_ids)    
    prompts=rag.format_prompts(context=relevant_text, prompt=prompt,system_prompt=rag.original_system_prompt)
    print("--- prompts --")
    print(prompts)
    for i in range(0,5):
       print("--- answer--")
       generator = rag.call_llm(prompts)
       # grab all values from the generator and concatenate them
       retVal=''.join(list(generator))
       print(retVal)
    print("------------")
    return retVal


#populate()
print("chunks in db:",rag.db_chunks_size())

ask("jak zalozit auto?")


# BenchmarkedRag("nomic-embed-text:latest",["."])

#rag=Rag()
# rag.chromadbpath="benchmark"
# rag.text_cleaner=lambda self,text:text
 




