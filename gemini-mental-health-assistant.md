
# 🧠 Gemini-Powered Mental Health Assistant: A Thoughtful Guide to Building Empathetic AI with LangGraph + RAG

Mental health is deeply human — and while AI should never replace real care, it can support people with timely, grounded, and compassionate responses. That’s what this project is about.

In this tutorial, we’ll walk through how to build a **Gemini-powered mental health assistant** using [LangGraph](https://www.langgraph.dev) for structured conversation flows and **RAG (Retrieval-Augmented Generation)** to provide real, document-backed answers.

This assistant isn’t just smart — it’s mindful.

---

## 💡 Why This Project?

We wanted to create an AI assistant that can:

- Chat empathetically, without being generic
- Provide helpful, evidence-based mental health insights
- Stay grounded in reliable documents (no hallucinations!)
- Be flexible, modular, and easy to extend

To achieve this, we used:

- 🧠 **Google Gemini** for both chat and embeddings
- 📄 A publicly available **Mental Health Guide PDF**
- 📦 **FAISS** for vector search
- 🧩 **LangGraph** for structuring the conversation flow

---

## 🔧 Setting Up the Environment

```bash
pip install -U langgraph langchain google-generativeai
pip install -U faiss-cpu tiktoken langchain-community
```

---

## 📥 Loading the Mental Health Guide

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("Mental Health Guide.pdf")
docs = loader.load()
```

---

## ✂️ Splitting and Embedding

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeEmbeddings
from langchain.vectorstores import FAISS

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = text_splitter.split_documents(docs)

embeddings = GoogleGenerativeEmbeddings(model="models/embedding-001")
vectorstore = FAISS.from_documents(split_docs, embeddings)
```

---

## 🔍 Retrieval-Augmented Generation (RAG)

```python
retriever = vectorstore.as_retriever()

def retrieve_context(user_input):
    docs = retriever.get_relevant_documents(user_input)
    return "\n\n".join(doc.page_content for doc in docs)
```

---

## 🧾 Grounding the Model with Context

```python
from langchain.prompts import PromptTemplate

prompt = PromptTemplate.from_template("""
You are a helpful and compassionate mental health assistant.
Use the following context to answer the user's question:

{context}

User: {question}
""")
```

---

## 🔁 LangGraph to Orchestrate the Flow

```text
user_input → retrieve_context → generate_response
```

Here’s a visual representation:

![LangGraph Flow](A_blog_post_image_titled_"Building_a_Compassionate.png")

---

## 🚀 Try It Yourself

📥 **[Download the Notebook](gemini-powered-mental-health-assistant.ipynb)**

---

## 💬 Final Thoughts

This isn’t just another chatbot — it’s a prototype for what empathetic AI could be.

By blending Gemini’s power with the structure of LangGraph and the grounding of RAG, we get an assistant that listens, reasons, and supports — all while staying rooted in real-world knowledge.

And the best part? It’s completely customizable for your own use cases.
