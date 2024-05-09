import os

import streamlit as st
from duckduckgo_search import DDGS
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

st.title("Chat with Google Search")

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
llm = genai.GenerativeModel("gemini-pro")

query = st.text_input("Enter your query, e.g., 'Tell me about the new iPad Pro'")
if query:
    results = DDGS().text(
        query,
        region="wt-wt",
        safesearch="moderate",
        timelimit="y",
        max_results=5,
        backend="api",
    )

    with st.expander("Sources"):
        for result in results:
            st.write(result)

    context = "\n\n".join([result["title"] + result["body"] for result in results])
    prompt = f"""You are an AI assistant that answers the user's question. Please answer based on the provided context below. If there is no information in the context, reply 'I do not have enough information to answer the question.'
            
    Context: {results}
    
    Question: {query}

    Answer:
    """
    with st.spinner("Generating response..."):
        response = llm.generate_content(prompt)
        st.write(response.text)
