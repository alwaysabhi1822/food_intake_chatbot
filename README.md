# food_intake_chatbot
# 🍲 Food Intake Chatbot

An AI-powered chatbot that provides personalized **Indian diet recommendations** based on user preferences, health conditions, and BMI report data.

Built using **Streamlit**, **LangChain**, and the **Groq API (LLaMA3-70B)** for real-time conversational intelligence.

---

## 🚀 Features

- 🧑‍⚕️ Personalized Indian diet advice
- 📊 BMI report-based food suggestions
- 📱 Responsive Streamlit UI
- 💡 Health issue-specific recommendations (e.g. Diabetes, BP, Cholesterol)
- 🧠 FAISS-based semantic search for food/nutrition data
- 🔐 Secure via `.env` environment variables

---

## 🏗️ Architecture

- **Frontend**: Streamlit
- **Backend/LLM**: LangChain + Groq (LLaMA3-70B)
- **Embeddings**: HuggingFace (`all-MiniLM-L6-v2`)
- **Vector Store**: FAISS (`vector.pkl`)

---

---

## ⚙️ Setup Instructions

### 1. 🔧 Create `food_data.txt`

Prepare a plain text file named `food_data.txt` that contains detailed, categorized content on:

- Indian foods & nutrients
- Diet plans for specific diseases
- Weight loss/gain strategies
- Balanced Indian meal suggestions


---

### 2. 🔁 Generate `vector.pkl`

```python
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import pickle

# Load and split text
with open("food_data.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_text(raw_text)
documents = [Document(page_content=chunk) for chunk in chunks]

# Embed
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents, embeddings)

# Save
with open("vector.pkl", "wb") as f:
    pickle.dump(vectorstore, f)
```
```python
3. 🌐 Set up Environment
Create a .env file and add your Groq API key:
```
4. 💻 Run the Application
```python
   # Create virtual environment
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Launch Streamlit app
streamlit run main.py
```

