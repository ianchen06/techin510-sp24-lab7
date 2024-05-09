import os

import fitz
import google.generativeai as genai
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search
from utils import split_large_text, displayPDF

st.set_page_config(layout="wide")
st.title("Chat with PDF")

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
llm = genai.GenerativeModel("gemini-pro")
model = SentenceTransformer("Supabase/gte-small")

col1, col2 = st.columns(2)
with col2:
    pdf_display = st.empty()

with col1:
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file:
        f = uploaded_file.read()
        displayPDF(pdf_display, uploaded_file.getvalue())

        doc = fitz.open(stream=f, filetype="pdf")
        text = ""
        for page in doc:  # iterate the document pages
            text += page.get_text()
        split_text_list = split_large_text(text, 512)
        docs_embeddings = model.encode(split_text_list, convert_to_tensor=True)

        query = st.text_input("Enter your query")
        if query:
            query_embedding = model.encode(query, convert_to_tensor=True)
            hits = semantic_search(query_embedding, docs_embeddings, top_k=2)

            context = "\n\n".join(
                [split_text_list[hit["corpus_id"]] for hit in hits[0]]
            )

            with st.expander("Sources"):
                for hit in hits[0]:
                    st.write(
                        split_text_list[hit["corpus_id"]],
                        "(Score: %.4f)" % hit["score"],
                    )

            prompt = f"""You are an AI assistant that answers the user's question. Please answer based on the provided context below. If there is no information in the context, reply 'I do not have enough information to answer the question.'
            
            Context: {context}
            
            Question: {query}

            Answer:
            """
            with st.spinner("Generating response..."):
                response = llm.generate_content(prompt)
                st.write(response.text)
