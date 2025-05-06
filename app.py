import os
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("❌ OpenAI API key not found. Please add it to the .env file.")
    st.stop()

openai.api_key = OPENAI_API_KEY

# Streamlit page setup
st.set_page_config(page_title="Student Performance Chatbot", layout="wide")
st.title("📊 Student Performance EDA Chatbot")
st.sidebar.header("🔍 User Input")

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Upload CSV file
uploaded_file = st.file_uploader("📂 Upload your dataset (CSV format)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Clean column names for reference
    column_mapping = {col.lower().replace(" ", "_"): col for col in df.columns}

    def preprocess_query(query):
        query = query.lower()
        for clean_col, original_col in column_mapping.items():
            if clean_col in query:
                query = query.replace(clean_col, original_col)
        return query

    dataset_overview = f"""
    ### Dataset Overview:
    - **Rows**: {df.shape[0]}, **Columns**: {df.shape[1]}
    - **Columns & Types**:
    {df.dtypes.to_string()}

    ### Summary Statistics:
    {df.describe().to_string()}

    ### Sample Data:
    {df.head(3).to_string()}
    """

    option = st.sidebar.selectbox("📊 Choose an analysis:", [
        "Select an option",
        "Basic Info",
        "Gender Distribution",
        "Race/Ethnicity Distribution",
        "Parental Education Impact",
        "Maximum Scores",
        "Grade Distribution",
    ])

    def plot_pie_bar(data, title):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sns.barplot(x=data.index, y=data.values, ax=axes[0], palette="viridis")
        axes[0].set_title(f"{title} (Bar Chart)")
        axes[0].set_ylabel("Count")
        axes[0].tick_params(axis='x', rotation=45)

        axes[1].pie(data, labels=data.index, autopct='%1.1f%%',
                    colors=sns.color_palette("viridis", len(data)))
        axes[1].set_title(f"{title} (Pie Chart)")
        st.pyplot(fig)

    def plot_max_scores():
        max_scores = {
            'Math': df['math score'].max(),
            'Reading': df['reading score'].max(),
            'Writing': df['writing score'].max()
        }
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=list(max_scores.keys()), y=list(max_scores.values()), palette="coolwarm", ax=ax)
        ax.set_title("📊 Maximum Scores in Each Subject")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 100)
        st.pyplot(fig)

    # Handle each option
    if option == "Basic Info":
        st.subheader("📄 Dataset Overview")
        st.write(df.describe())
        st.write("### 🔹 Missing Values:", df.isnull().sum())
        st.write("### 🔹 Duplicate Rows:", df.duplicated().sum())

    elif option == "Gender Distribution":
        st.subheader("👩‍🎓 Gender Distribution")
        gender_counts = df['gender'].value_counts()
        plot_pie_bar(gender_counts, "Gender Distribution")

    elif option == "Race/Ethnicity Distribution":
        st.subheader("🌍 Race/Ethnicity Distribution")
        race_counts = df['race/ethnicity'].value_counts()
        plot_pie_bar(race_counts, "Race/Ethnicity Distribution")

    elif option == "Parental Education Impact":
        st.subheader("🎓 Parental Education Distribution")
        education_counts = df['parental level of education'].value_counts()
        plot_pie_bar(education_counts, "Parental Education Distribution")

    elif option == "Maximum Scores":
        st.subheader("📊 Maximum Scores in Subjects")
        plot_max_scores()

    elif option == "Grade Distribution":
        st.subheader("📚 Grade Distribution")
        df['Grade'] = pd.cut(df[['math score', 'reading score', 'writing score']].mean(axis=1),
                             bins=[0, 50, 70, 85, 100],
                             labels=["D", "C", "B", "A"])
        grade_counts = df['Grade'].value_counts()
        plot_pie_bar(grade_counts, "Grade Distribution")

    # AI Chatbot section
    st.subheader("💬 Ask ChatGPT about the dataset")

    def get_chatgpt_response(query):
        prompt = f"""
You are a data analyst helping a user interpret a dataset. Here's the dataset info:

{dataset_overview}

User Query:
{query}

Respond with clear insights, suggestions, or visual analysis ideas. If the query is unrelated, politely guide the user.
"""
        response = openai.ChatCompletion.create(
              model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful data analyst."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message["content"]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("🔍 Ask a question...")

    if user_input:
        processed_query = preprocess_query(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        answer = get_chatgpt_response(processed_query)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)
