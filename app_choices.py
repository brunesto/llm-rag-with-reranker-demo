
EMBEDDINGS = [
    "ollama/nomic-embed-text:latest",
    "ollama/mxbai-embed-large",
    "ollama/snowflake-arctic-embed",
    "ollama/all-minilm",
    "ollama/bge-m3",
    "ollama/bge-large",
    "ollama/paraphrase-multilingual",
    "ollama/snowflake-arctic-embed2 ",    
    "seznam/Seznam/retromae-small-cs",
    "seznam/Seznam/dist-mpnet-paracrawl-cs-en",
    "seznam/Seznam/dist-mpnet-czeng-cs-en",
    "seznam/Seznam/simcse-retromae-small-cs",
    "seznam/Seznam/simcse-dist-mpnet-paracrawl-cs-en",
    "seznam/Seznam/simcse-dist-mpnet-czeng-cs-en",
    "seznam/Seznam/simcse-small-e-czech",
    ]
EMBEDDING_DISTANCE_FUNCTIONS=["cosine", "l2", "ip"]

LLM_MODEL=["llama3.2:3b","llama3.2:1b", #facebook, no cz
           "mistral", # mistral AI
           "gemma:2b","gemma:7b", # google, may be it supports cz
           "phi3","phi3:14b" # microsoft
           "qwq","qwen2","marco-o1", # alibaba
           "dbrx",
           "alfred",
           "aya-expanse", # cz support
           "nuextract"
           #"llama3.3",
            ]

CROSS_ENCODERS= (
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "NONE",
            "cross-encoder/ms-marco-TinyBERT-L-2-v2",
            "cross-encoder/ms-marco-MiniLM-L-2-v2",
            "cross-encoder/ms-marco-MiniLM-L-4-v2",
            #    "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "cross-encoder/ms-marco-MiniLM-L-12-v2",
            "cross-encoder/qnli-distilroberta-base",
            "cross-encoder/qnli-electra-base",
            "cross-encoder/stsb-TinyBERT-L-4",
            "cross-encoder/stsb-distilroberta-base",
            "cross-encoder/stsb-roberta-base",
            "cross-encoder/stsb-roberta-large",
            "cross-encoder/quora-distilroberta-base",
            "cross-encoder/quora-roberta-base",
            "cross-encoder/quora-roberta-large",
            "cross-encoder/nli-deberta-v3-base",
            "cross-encoder/nli-deberta-base",
            "cross-encoder/nli-deberta-v3-xsmall",
            "cross-encoder/nli-deberta-v3-small",
            "cross-encoder/nli-roberta-base",
            "cross-encoder/nli-MiniLM2-L6-H768",
            "cross-encoder/nli-distilroberta-base",
            "BAAI/bge-reranker-base",
            "BAAI/bge-reranker-large",
            "BAAI/bge-reranker-v2-m3",
            "BAAI/bge-reranker-v2-gemma",
            "BAAI/bge-reranker-v2-minicpm-layerwise",
            "jinaai/jina-reranker-v1-tiny-en",
            "jinaai/jina-reranker-v1-turbo-en",
            "mixedbread-ai/mxbai-rerank-xsmall-v1",
            "mixedbread-ai/mxbai-rerank-base-v1",
            "mixedbread-ai/mxbai-rerank-large-v1",
            "maidalun1020/bce-reranker-base_v1",
        )
