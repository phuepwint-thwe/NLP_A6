import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.vectorstores import FAISS
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceInstructEmbeddings

# Set up API keys
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your_key"
os.environ["GROQ_API_KEY"] = "your_key"

# Load FAISS-based vector store
def load_faiss():
    vector_path = "../vector-store"
    db_file_name = "phue"
    embedding_model = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base")
    vectordb = FAISS.load_local(
        folder_path=os.path.join(vector_path, db_file_name),
        embeddings=embedding_model,
        index_name="ppt",
        allow_dangerous_deserialization=True
    )
    return vectordb.as_retriever()


retriever = load_faiss()
groq_model = ChatGroq(model_name="llama-3.3-70b-specdec", temperature=0.7)

# Define prompt template
prompt_template = """
You are a helpful AI assistant. Answer the question based on the provided context.

Context:
{context}

Question: {question}

Gentle & Informative Answer:
""".strip()

PROMPT = PromptTemplate.from_template(prompt_template)

def get_response(question):
    input_documents = retriever.get_relevant_documents(question)
    context = "\n".join([doc.page_content for doc in input_documents])
    response = groq_model.invoke([{"role": "user", "content": PROMPT.format(context=context, question=question)}])
    return response.content, input_documents

# Streamlit UI Setup
st.set_page_config(page_title="Chatbot with RAG", layout="centered")

# Title & Description
st.title("💬 RAG-Based Chatbot")
st.write("Ask a question and get AI-powered responses based on retrieved documents.")

# User Input Section
user_input = st.text_area("🔍 Enter your question below:", height=100)

if st.button("Get Answer"):
    if user_input.strip():
        with st.spinner("Generating response..."):
            response, documents = get_response(user_input)

        # Display chatbot response
        st.subheader("🤖 AI Response")
        st.write(response)

        # Display Supporting Documents (Collapsible)
        if documents:
            st.subheader("📄 Supporting Documents")
            for idx, doc in enumerate(documents):
                with st.expander(f"Document {idx + 1}"):
                    st.write(doc.page_content[:1000])  # Show first 1000 characters

    else:
        st.warning("⚠️ Please enter a question.")

# Footer
st.markdown("---")
st.text("Powered by Groq & FAISS | Built with ❤️ using Streamlit")

