
# This is just the UI 
from dbrag import *
from chatrag import *
import streamlit as st

st.set_page_config(page_title="RAG Question Answer")

print("new Rag...")
dbrag=DbRag()
chatrag=ChatRag()



# Question and Answer Area
st.header("🗣️ RAG Question Answer")


# Document Upload Area
with st.sidebar:
   

    dbrag.chromadbpath = st.text_input("db (this let you experiment with loading different documents):","demo-rag")
    
    st.write("db stats: ",dbrag.db_stats())

    dbrag.config.embedding = st.selectbox(
    "embedding (https://ollama.com/search?c=embedding)",
    ("ollama/nomic-embed-text:latest",
     "ollama/mxbai-embed-large"
    ))


   
    uploaded_file = st.file_uploader(
        "Upload PDF ", type=["pdf"], accept_multiple_files=False
    )

    process = st.button(
        "⚡️ Process",
    )
    if uploaded_file and process:
        # Store uploaded file as a temp file
        temp_file = tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False)
        temp_file.write(uploaded_file.read())
        dbrag.splitAndStore(temp_file.name)
        os.unlink(temp_file)  # Delete temp file

        st.success("Data added to the vector store!")






prompt = st.text_area("prompt:")
ask = st.button(
    "🔥 Ask",
)

dbrag.config.embedding_distance_function = st.selectbox(
    "embeddings distance function (https://docs.trychroma.com/guides#changing-the-distance-function)",
    ("cosine", "l2", "ip"),
    )


dbrag.config.cross_encoder = st.selectbox(
    """cross encoder is used to rerank documents before feeding to LLM  (https://www.sbert.net/docs/cross_encoder/pretrained_models.html)""",
   ("cross-encoder/ms-marco-MiniLM-L-6-v2",
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
    "maidalun1020/bce-reranker-base_v1"

     )
)


system_prompt = st.text_area("system prompt:",value=chatrag.config.original_system_prompt)



chatrag.config.model=st.selectbox(
    "LLM model (https://ollama.com/search)",(
    "llama3.2:3b",
    "qwen2"
    ))

st.divider()

if ask and prompt:
    results = dbrag.query_collection(prompt)
    
    with st.expander("See retrieved documents based on embeddings"):
        st.write(results)

    context = results.get("documents")[0]
    relevant_text, relevant_text_ids = dbrag.re_rank_cross_encoders(prompt,context)

    with st.expander("Most relevant documents (reranking)"):
        st.write(relevant_text_ids)
        st.write(relevant_text)

    st.write("Response:")
    response = chatrag.call_llm(chatrag.format_prompts(context=relevant_text,prompt=prompt,system_prompt=system_prompt))
    st.write_stream(response)
    

    
