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

class DocParser:

    separators=["."]
    rules=[]    

    def process_document(self,uploaded_file,normalizedName) -> list:       
        normalizedName=normalizedName+"pymupdf4llm"
        # Store uploaded file as a temp file
        temp_file = tempfile.NamedTemporaryFile("wb", prefix=normalizedName,suffix=".pdf", delete=False)
        temp_file.write(uploaded_file.read())

        md_text = pymupdf4llm.to_markdown(temp_file)

        # # for debug
        pathlib.Path("/tmp/"+normalizedName+".md").write_bytes(md_text.encode())

        llama_reader = pymupdf4llm.LlamaMarkdownReader()
        llama_docs = list(llama_reader.load_data(temp_file.name))

        # convert to chunks and remove the metadata 
        chunks=list(map(lambda split:Chunk(meta=None,text=split.text),llama_docs))
        self.transform_chunks(chunks)
        self.dumpSplits(chunks,normalizedName)
        return chunks
    
    def dumpSplits(self,all_splits,normalizedName):
  # for debug: dump the splits as json
        strs=list(map(lambda x:x.text,all_splits))
        with open("/tmp/"+normalizedName+".json", "w") as save_file:  
            json.dump(strs, save_file, indent = 6)  
        
    def process_documentOLD(self,uploaded_file,normalizedName) -> list:       
        normalizedName=normalizedName+"classic"
        # Store uploaded file as a temp file
        temp_file = tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False)
        temp_file.write(uploaded_file.read())

        loader = PyMuPDFLoader(temp_file.name)
        docs = loader.load()        
        os.unlink(temp_file.name)  # Delete temp file

        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=100,            
            separators=self.separators,
        )
        all_splits=text_splitter.split_documents(docs)
        chunks=list(map(lambda split:Chunk(meta=split.metadata,text=split.page_content),all_splits))
        self.transform_chunks(chunks)
        self.dumpSplits(chunks,normalizedName)
        return chunks
    


    def transform_chunks(self, chunks):
        """Split documents."""
        # texts, metadatas = [], []
        for doc in chunks:
            doc.text=reformat_text(self.rules,doc.text)            

      
