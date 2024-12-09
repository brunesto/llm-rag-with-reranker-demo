# docparser.py
# in charge of parsing and splitting documents
#
# TODO probably langchain does this better


import pathlib
import os
import re
import json
import tempfile
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pymupdf4llm
from dataclasses import dataclass



def reformat_text(rules,text):
        """ each rule is a triplet: (regexp,replacement,rule name) """
        print("original:")
        print(text)
        text=text.strip()
        if len(text)>0:
            if text[0]!='\n':
                text='\n'+text
            for rule in rules:
               print("applying rule:",rule[2])
               text=re.sub(rule[0],rule[1],'\n'+text)
               print("text:",text)
        return text    
@dataclass
class Chunk:
   meta: str
   text: str

def dumpSplits(all_splits,normalizedName):
  # for debug: dump the splits as json
        strs=list(map(lambda x:x.text,all_splits))
        with open("/tmp/"+normalizedName+".json", "w") as save_file:  
            json.dump(strs, save_file, indent = 6)  

def transform_chunks(rules, chunks):
        """Split documents."""
        # texts, metadatas = [], []
        for doc in chunks:
            doc.text=reformat_text(rules,doc.text)           
  

class DocParserNew:
    rules=[]    

             
    def process_document(self,filename,normalizedName) -> list:       
        normalizedName=normalizedName+"pymupdf4llm"
        

        md_text = pymupdf4llm.to_markdown(filename)

        # # for debug
        pathlib.Path("/tmp/"+normalizedName+".md").write_bytes(md_text.encode())

        llama_reader = pymupdf4llm.LlamaMarkdownReader()
        llama_docs = list(llama_reader.load_data(filename))

        # convert to chunks and remove the metadata 
        chunks=list(map(lambda split:Chunk(meta=None,text=split.text),llama_docs))
        transform_chunks(self.rules,chunks)
        dumpSplits(chunks,normalizedName)
        return chunks
    
class DocParserOld:

    separators=["."]
    rules=[]  

    
    
        
    def process_document(self,temp_file,normalizedName) -> list:       
        normalizedName=normalizedName+"classic"
      

        loader = PyMuPDFLoader(temp_file)
        docs = loader.load()        
      
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=100,            
            separators=self.separators,
        )
        all_splits=text_splitter.split_documents(docs)
        chunks=list(map(lambda split:Chunk(meta=split.metadata,text=split.page_content),all_splits))
        transform_chunks(self.rules,chunks)
        dumpSplits(chunks,normalizedName)
        return chunks
    


    

      
