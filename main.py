import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain import hub
import pandas as pd
import os
import json
import pdb

st.sidebar.title("Navigation")
selected_option = st.sidebar.radio("Mode:", ["QA Pairs", "Answer Generator"])


# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"

# Path to your CSV file (update as necessary)
csv_file_path = 'df_covid_qa_kg.csv'

# Load the dataset
df = pd.read_csv(csv_file_path)
df = df.drop(df.columns[0], axis=1)

# Function to format JSON data with HTML/CSS styling


def format_json(data):
    # Convert dictionary to JSON string
    json_str = json.dumps(data, indent=4)

    # HTML/CSS styling
    styled_json = f"""
    <style>
        .json-wrapper {{
            font-family: Arial, sans-serif;
            border: 1px solid #ccc;
            border-radius: 5px;
            padding: 10px;
            background-color: #f9f9f9;
        }}
        .json-code {{
            white-space: pre-wrap;
            word-break: break-all;
            overflow-x: auto;
        }}
    </style>
    <div class="json-wrapper">
        <code class="json-code">{json_str}</code>
    </div>
    """
    return styled_json


if selected_option =='QA Pairs':
    # Title of your app
    st.title('Question-Answering pairs')
    st.write(df)

else:
    st.title("Answer Generator")
    embeddings = OpenAIEmbeddings()
    db = FAISS.load_local('faiss_db_covid_abstracts/', index_name= "index", embeddings = embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 6})

    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)


    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)


    rag_chain_from_docs = (
            RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
            | prompt
            | llm
            | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)

    # User input
    user_question = st.text_input("Ask a question:", "")

    if user_question:
        try:
            # Invoke the RAG chain with the user's question
            answer = rag_chain_with_source.invoke(user_question)
            st.write("Answer:", answer)
        except Exception as e:
            st.write("An error occurred while processing your question.")
            st.error(e)