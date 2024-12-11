# This is just the UI
from dbrag import *
from chatrag import *
import streamlit as st
import numpy as np
import pandas as pd
from app_choices import *

st.set_page_config(layout="wide", page_title="RAG Question Answer")


def trans(M):
    return [[M[j][i] for j in range(len(M))] for i in range(len(M[0]))]


# init
if "dbrag" not in st.session_state:
    print("new session")
    st.session_state.dbrag = DbRag()
    st.session_state.chatRag = ChatRag()

dbrag = st.session_state.dbrag
chatrag = st.session_state.chatRag


# Question and Answer Area
st.header("🗣️ RAG Question Answer")

@st.dialog("Db settings")
def change_deb_settings():
    st.write(f"The db is specified thru basename + embedding")
    base_dir = st.text_input(
        "base name (This let you experiment with loading different documents:)",
        value=st.session_state.dbrag.config.base_dir,
    )
    # dbencoding=st.text_input("encoding:",value=st.session_state.dbrag.config.embedding)
    dbembedding = st.selectbox(
        "prefix embedding by either\n\n * ollama/ (for https://ollama.com/search?c=embedding), \n * seznam (for https://github.com/seznam/czech-semantic-embedding-models)",
        EMBEDDINGS,
        index=EMBEDDINGS.index(st.session_state.dbrag.config.embedding),
    )
    if st.button("Submit"):
        st.session_state.dbrag.config.base_dir = base_dir
        st.session_state.dbrag.config.embedding = dbembedding
        st.rerun()


# Document Upload Area

col1, col2, col3 = st.columns([1, 3, 3])

col1.subheader("Chroma DB")
col2.subheader("Retrieval")
col3.subheader("LLM")


chunks = dbrag.db_total_chunks()["chunks"]

with col1:

    st.markdown(
        f"\n###### db basedir\n{dbrag.config.base_dir}"
        + f"\n###### embedding\n{dbrag.config.embedding}"        
    )

    with st.expander("🗎 upload"):

        uploaded_file = st.file_uploader(
            "Upload PDF ", type=["pdf"], accept_multiple_files=False
        )

        process = st.button(
            "⚡️ Process",
        )
        if uploaded_file and process:
            normalizedFileName = sanitizedBaseName(uploaded_file.name)
            # Store uploaded file as a temp file
            temp_file = tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False)
            print("temp file:", temp_file.name)
            temp_file.write(uploaded_file.read())
            processed=dbrag.splitAndStore(temp_file.name,normalizedFileName)
            os.unlink(temp_file.name)  # Delete temp file
            if processed:
                st.success("Data added to the db!")
            else:
                st.warning("document already in db")
            chunks = dbrag.db_total_chunks()["chunks"]


    # st.header('Chroma DB')

    st.markdown(f"\n###### chunks\n{chunks}")

    stats=dbrag.db_read_upload_stats()
    with st.expander("uploaded"):
            data = []
            for row in stats:
                data.append(row)
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=False)

    if st.button("Change Db settings"):
        change_deb_settings()

    # dbrag.chromadbpath = st.text_input("db (this let you experiment with loading different documents):","demo-rag")

   
with col2:

    if chunks == 0:
        st.warning("⚠️ no chunks in db. plz upload some pdf or change db")

    prompt = st.text_area("prompt:")
    ask = st.button(
        "▶️ Ask",
    )

    dbrag.config.embedding_distance_function = st.selectbox(
        "embeddings distance function (https://docs.trychroma.com/guides#changing-the-distance-function)",
        EMBEDDING_DISTANCE_FUNCTIONS,
    )

    dbrag.config.cross_encoder = st.selectbox(
        """cross encoder is used to rerank documents before feeding to LLM  (https://www.sbert.net/docs/cross_encoder/pretrained_models.html)""",
        CROSS_ENCODERS      
    )

    st.divider()

    with col3:

        system_prompt = st.text_area(
            "system prompt:", value=chatrag.config.original_system_prompt
        )

        chatrag.config.model = st.selectbox(
            "LLM model (https://ollama.com/search)", LLM_MODEL
        )

        st.divider()

    if ask and prompt:
        results = dbrag.query_collection(prompt)
        distances = results.get("distances")[0]
        context = results.get("documents")[0]

        with st.expander("Matching chunks"):
            data = []
            for idx, text in enumerate(context):
                data.append({"distance": distances[idx], "text": context[idx]})
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)

        relevant_text, relevant_text_scores = dbrag.re_rank_cross_encoders(
            prompt, context
        )

        with st.expander("Most relevant documents (reranking)"):
            data = []
            for idx, text in enumerate(relevant_text):
                data.append(
                    {"distance": relevant_text_scores[idx], "text": relevant_text[idx]}
                )
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)

        # print('========================')
        # print(relevant_text)
        # print('========================')
        # with st.expander("Most relevant documents (reranking)"):
        #    for idx,text in enumerate(relevant_text):
        #         # st.write(relevant_text_ids)
        #         st.write(text)

        with col3:
            with st.expander("full prompts:"):
                prompts = chatrag.format_prompts(
                    context=relevant_text, prompt=prompt, system_prompt=system_prompt
                )
                st.write(prompts)
            st.write("Response:")
            response = chatrag.call_llm(prompts)
            st.write_stream(response)
