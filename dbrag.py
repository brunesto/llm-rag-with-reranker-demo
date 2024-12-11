#
# document retrieval
#

import os
import tempfile
import time

# https://stackoverflow.com/questions/76958817/streamlit-your-system-has-an-unsupported-version-of-sqlite3-chroma-requires-sq
__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import pickle
import chromadb

import ollama
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)
from sentence_transformers import CrossEncoder
from chromadb import Documents, EmbeddingFunction, Embeddings
import torch
from transformers import AutoModel, AutoTokenizer
from docparser import *


def sanitizedName(uploaded_file):
    r = "-"
    return uploaded_file.translate(
        str.maketrans(
            {
                ",": r,
                ":": r,
                ".": r,
                "'": r,
                " ": r,
                "/": r,
                "(": r,
                ")": r,
                "[": r,
                "]": r,
            }
        )
    )


def sanitizedBaseName(uploaded_file):
    return sanitizedName(os.path.basename(uploaded_file))


class SeznamEmbeddings(EmbeddingFunction):
    """https://github.com/seznam/czech-semantic-embedding-models"""

    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def __call__(self, input: Documents) -> Embeddings:
        print(input[0])
        print("===============================================")
        # texts=list(map(lambda d:d.text,input))
        # embed the documents somehow
        return self.embeddings(input)

    def embeddings(self, texts):
        # Tokenize the input texts
        batch_dict = self.tokenizer(
            texts, max_length=128, padding=True, truncation=True, return_tensors="pt"
        )

        outputs = self.model(**batch_dict)
        embeddings = outputs.last_hidden_state[:, 0]  # CLS
        return embeddings.detach().numpy()


def get_embedding_func(config_str):
    retVal = None
    config = config_str.split("/", 1)
    print("embedding:", str(config))
    if config[0] == "ollama":
        retVal = OllamaEmbeddingFunction(
            url="http://localhost:11434/api/embeddings", model_name=config[1]
        )
    elif config[0] == "seznam":
        retVal = SeznamEmbeddings(config[1])
    else:
        raise "dunno embedding:" + str(config)

    # texts = ["Hello, world!", "How are you?"]
    # embeddings = retVal(texts)
    # print(embeddings)
    return retVal


def get_doc_parser(config):
    if config[0] == "DocParserOld":
        return DocParserOld()
    elif config[0] == "DocParserNew":
        retVal = DocParserNew()
        retVal.rules = config[1]["rules"]
    else:
        raise "dunno doc_parser config:" + str(config)
    return retVal


@dataclass
class DbConfig:
    # root directory of all dbs
    base_dir = "./chromadbs/"
    # the name of DB
    collection_name = "demo-rag"
    # embedding=("ollama/nomic-embed-text:latest")
    embedding = "ollama/nomic-embed-text:latest"

    docparser = ("DocParserOld", None)
    # docparser = (
    #     "DocParserNew",
    #     {
    #         "rules": [
    #             (r"^\d+\/70 ", "", "remove page number"),
    #             (r"\n *\n", "\n", "simplify newlines"),
    #             (r" +", " ", "merge spacing"),
    #             (r"\n[ #]*(\d+.)+ ([^\n]*)", "\nodstavec \\2:", "make title explicit"),
    #         ]
    #     },
    # )
    # https://www.sbert.net/docs/cross_encoder/pretrained_models.html
    cross_encoder = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # https://docs.trychroma.com/guides#changing-the-distance-function
    embedding_distance_function = "cosine"

    max_answer_chunks = 3


def is_DbConfig_compatible_with(db1: DbConfig, db2: DbConfig):
    """returns True if a runtime and saved db config are compatible"""
    if db1.embedding != db2.embedding:
        print("embedding mismatch")
        return False
    return True


class DbRag:

    config = DbConfig()

    def db_basedirname(self):
        return self.config.base_dir + "/" + self.config.collection_name

    def db_prefix(self):
        return self.db_basedirname() + "/" + sanitizedName(str(self.config.embedding))

    def get_vector_collection(self) -> chromadb.Collection:

        # ensure root db dir exists

        os.makedirs(self.db_basedirname(), exist_ok=True)

        # # if there is an existing conf, check that is compatible with self.config
        # conf_filename = self.config.base_dir + self.config.collection_name + ".pickle"
        # print("conf_filename:",conf_filename)
        # if os.path.isfile(conf_filename):
        #     with open(conf_filename, "rb") as f:
        #         existingconf = pickle.load(f)
        #         if not is_DbConfig_compatible_with(self.config, existingconf):
        #             raise (
        #                 "db config not compatible with whats on disk"
        #                 + str(existingconf)
        #             )
        # else:
        #     with open(conf_filename, "wb") as f:
        #         pickle.dump(self.config, f, protocol=pickle.HIGHEST_PROTOCOL)

        # # now create the chromadb collection
        # print("using embedding", self.config.embedding)

        embedding_function = get_embedding_func(self.config.embedding)

        chroma_client = chromadb.PersistentClient(path=self.db_prefix()+"-chromadb")
        
        return chroma_client.get_or_create_collection(
            name="rag_app",
            embedding_function=embedding_function,
            metadata={"hnsw:space": self.config.embedding_distance_function},
        )

   

    def db_upload_stats_filename(self):
        return self.db_prefix() + "-uploads.pickle"

    def db_read_upload_stats(self):
        """return the list of files uploaded to db"""
        path = self.db_upload_stats_filename()
        if os.path.isfile(path):
            with open(path, "rb") as fp:
                itemlist = pickle.load(fp)
            return itemlist
        else:
            return []
        

    def db_save_upload_stats(self,stats):
        path = self.db_upload_stats_filename()
        with open(path, 'wb') as fp:
            pickle.dump(stats, fp)


    def db_total_chunks(self):
        """return the number of chunks found in the db"""
        ids_only_result = self.get_vector_collection().get(include=["metadatas"])

        tuples = set()
        for metadata in ids_only_result["metadatas"]:
            # if metadata == None:
            tuples.add((None, None))
        # else:
        #     tuples.add((metadata["file"], metadata["time"]))

        return {"files": tuples, "chunks": len(ids_only_result["ids"])}

    # -- adding docs to DB -----------------------

    def splitAndStore(self, filename,normalizedFileName=None):
        print("splitAndStore()")
        stats=self.db_read_upload_stats()
        if normalizedFileName == None:
            normalizedFileName = sanitizedBaseName(filename)
        if normalizedFileName in map(lambda x: x["normalizedFileName"], stats):
            print("already in db ", normalizedFileName)
            return False
        
        
        docParser = get_doc_parser(self.config.docparser)
        all_splits = docParser.process_document(filename, normalizedFileName)
        self.add_to_vector_collection(all_splits, normalizedFileName)
        

        stats.append({"normalizedFileName":normalizedFileName,"chunks":len(all_splits)})
        self.db_save_upload_stats(stats)
        return True

    def add_to_vector_collection(self, all_splits: list[Document], file_name: str):
        print("add_to_vector_collection()")
        collection = self.get_vector_collection()
        documents, metadatas, ids = [], [], []

        current_time = time.time()

        for idx, split in enumerate(all_splits):
            #     documents.append(split.text)

            #     #split.meta[]
            #     metadatas.append(split.meta)
            #     # {"file": file_name, "time": current_time})
            #     ids.append(f"{file_name}_{idx}")

            # collection.upsert(
            #     documents=documents,
            #     metadatas=metadatas,
            #     ids=ids,
            # )
            id = f"{file_name}_{idx}"
            print(id, split.text, "...")
            collection.upsert(documents=[split.text], metadatas=[split.meta], ids=[id])

    # -- retrieving data from DB -----------------------

    def query_collection(self, prompt: str, n_results: int = 10):
        collection = self.get_vector_collection()
        results = collection.query(query_texts=[prompt], n_results=n_results)
        return results

    def re_rank_cross_encoders(
        self, prompt: str, documents: list[str]
    ) -> tuple[str, list[int]]:
        """Re-ranks documents using a cross-encoder model for more accurate relevance scoring."""

        relevant_text = []
        relevant_text_score = []
        if self.config.cross_encoder != "NONE":
            encoder_model = CrossEncoder(self.config.cross_encoder)
            if len(documents) > self.config.max_answer_chunks:
                ranks = encoder_model.rank(
                    prompt, documents, top_k=self.config.max_answer_chunks
                )
                for rank in ranks:
                    relevant_text.append(documents[rank["corpus_id"]])
                    relevant_text_score.append(rank["score"])
            else:
                for i in range(0, min(self.config.max_answer_chunks, len(documents))):
                    relevant_text.append(documents[i])
                    relevant_text_score.append(rank["score"])
        else:
            for i in range(0, min(self.config.max_answer_chunks, len(documents))):
                relevant_text.append(documents[i])
                relevant_text_score.append(0)

        return relevant_text, relevant_text_score

    # UI
