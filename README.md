# NLP_A6: Let's Talk with Yourself

## Project Details

- **Name:** Phe Pwint Thwe  
- **ID:** st124748  

This project implements **Retrieval-Augmented Generation (RAG)** using the **LangChain framework**. The chatbot is designed to answer personal questions based on provided **documents and resumes**.

The implementation is divided into three key tasks ; **Source Discovery (Documents)** , **Analysis and Problem Solving**, and **Chatbot Development (Web Application)**

---

## Task 1: Source Discovery

### Data Sources
- **Documents:**
  - `PHUE PWINT THWE Academic CV.pdf`

### Key Steps:
- **Load and Process Documents:** Using `PyMuPDFLoader` to extract text.
- **Text Chunking:** Splitting content into manageable sizes (`chunk_size=100`, `chunk_overlap=20`).
- **Embedding Generation:** Using `fastchat-t5-3b-v1.0` with `FAISS` for efficient search.
- **Retriever Setup:** `FAISS` is used to store and retrieve document embeddings efficiently.

---

## Task 2: Analysis and Problem Solving

### Models Used:

| Component    | Model |
|-------------|---------------------------------|
| **Retriever** | FAISS (`fastchat-t5-3b-v1.0`) |
| **Generator** | Groq LLaMA-3.1-8B (`https://python.langchain.com/docs/integrations/chat/groq/`) |

### Identified Issues & Solutions:
1. **Data Irregularities:** Ensured proper preprocessing and text cleaning.
2. **Model Hallucinations:** Improved prompt engineering for accurate answers.
3. **Performance Optimization:** Stored FAISS embeddings locally for faster retrieval.

---

## Task 3: Chatbot Development (Web Application)

A **Streamlit-based web application** was built to enable interaction with the chatbot.

### Features:
- **Chat Interface:** Users can enter messages in a web-based chatbox.
- **Intelligent Responses:** The model generates personalized responses using `qa_chain`.

### Screenshots:
- **Chatbot Interface**  
  ![Chatbot](A6.png)

## Example Questions for this Chatbot
1. How old are you?
2. What is your highest level of education?
3. What major or field of study did you pursue during your education?
4. How many years of work experience do you have?
5. What type of work or industry have you been involved in?
6. Can you describe your current role or job responsibilities?
7. What are your core beliefs regarding the role of technology in shaping society?
8. How do you think cultural values should influence technological advancements?
9. As a master’s student, what is the most challenging aspect of your studies so far?
10. What specific research interests or academic goals do you hope to achieve during your time as a master’s student?
---

## How to Run the Project

### **Step 1: Install Dependencies**
```bash
pip install streamlit langchain langchain-community torch sentence-transformers faiss-cpu

