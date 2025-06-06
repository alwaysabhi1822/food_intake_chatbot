import streamlit as st
import os
import pickle
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('Groq_Api_Key')

# Title
st.title("🍲 Food Recommendation Chatbot")

# Check for missing API Key
if not groq_api_key:
    st.error("🚨 GROQ API key not found. Please check your .env file.")
    st.stop()

# Load LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")

# Embeddings
huggingface_embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-l6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# Prompt Template
prompt = ChatPromptTemplate.from_template(
    """
    You are an expert in Indian food and nutrition. Your responsibilities include:

    1. **Personalized Diet Recommendations**:
       - Based on the user's health goals (e.g., weight loss, managing diabetes, cholesterol), suggest appropriate Indian diet plans.
       - Recommend specific Indian foods that are commonly available in India to consume or avoid, tailored to the user's dietary preferences.
       - Provide meal suggestions for breakfast, lunch, and dinner using Indian dishes.

    2. **Interpreting BMI Reports**:
       - If a BMI report is provided, analyze and give personalized advice based on the user's BMI and other body metrics.

    3. **General Nutrition Advice**:
       - Offer tips on healthy Indian eating habits, nutrient-rich foods, and strategies for balanced diet.

    **User Preferences**:
    Dietary preference: {diet_preference}
    Health problems: {health_problem}
    BMI Report: {bmi_report}

    **User Query**:
    {input}

    **Context**:
    {context}
    """
)

# Sidebar Preferences Form
st.sidebar.subheader("Set Your Preferences")

with st.sidebar.form(key="preferences_form"):
    diet_preference = st.selectbox("Dietary Preference", ["Veg", "Non-Veg", "Both"])
    health_problem = st.selectbox("Do you have any health issues?", ["No", "Yes"])

    if health_problem == "Yes":
        health_issue = st.multiselect("Select health issues", ["Diabetes", "Blood Pressure", "Cholesterol"])
    else:
        health_issue = []

    bmi_report = st.selectbox("Do you have BMI report data?", ["No", "Yes"])

    if bmi_report == "Yes":
        bmi_data = {
            "Body Composition Analysis": st.text_input("Body Composition Analysis"),
            "Intracellular Water": st.text_input("Intracellular Water"),
            "Extracellular Water": st.text_input("Extracellular Water"),
            "Total Body Water": st.text_input("Total Body Water"),
            "Lean Body Mass": st.text_input("Lean Body Mass"),
            "Weight": st.text_input("Weight"),
            "Dry Lean Mass": st.text_input("Dry Lean Mass"),
            "Body Fat Mass": st.text_input("Body Fat Mass"),
            "SMM": st.text_input("Skeletal Muscle Mass (SMM)"),
            "BMI": st.text_input("Body Mass Index (BMI)"),
            "PBF": st.text_input("Percent Body Fat (PBF)"),
        }
    else:
        bmi_data = {}

    submit_button = st.form_submit_button(label="OK")
    reset_button = st.form_submit_button(label="Reset Preferences")

# Handle preferences state
if reset_button:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.experimental_rerun()

if submit_button:
    st.session_state["diet_preference"] = diet_preference
    st.session_state["health_problem"] = health_problem
    st.session_state["health_issue"] = health_issue
    st.session_state["bmi_report"] = bmi_report
    st.session_state["bmi_data"] = bmi_data
    st.sidebar.success("✅ Preferences saved.")

# Load FAISS Vector Store
if "vectors" not in st.session_state:
    try:
        with open("vector.pkl", "rb") as f:
            st.session_state.vectors = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading vector store: {e}")
        st.stop()

# User Query
user_query = st.text_input("Enter Your Query Here:")

if not user_query:
    st.info("💡 Try asking something like: *Suggest a veg Indian diet for weight loss*")

# Run Chatbot
if user_query:
    diet_preference = st.session_state.get("diet_preference", "Both")
    health_problem = ", ".join(st.session_state.get("health_issue", [])) if st.session_state.get("health_problem") == "Yes" else "None"
    bmi_report = st.session_state.get("bmi_data", "Not provided")

    # Retrieval Chain
    doc_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, doc_chain)

    response = retrieval_chain.invoke({
        'input': user_query,
        'diet_preference': diet_preference,
        'health_problem': health_problem,
        'bmi_report': bmi_report,
        'context': '',
    })

    # Output response
    st.markdown("### 🤖 Chatbot Response")
    st.write(response["answer"])

    # Display retrieved documents (if any)
    if "context" in response and isinstance(response["context"], list):
        with st.expander("📄 Relevant Documents"):
            for doc in response["context"]:
                st.write(doc.page_content)
                st.write("---")
