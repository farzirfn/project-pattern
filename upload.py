import streamlit as st
import pandas as pd
import numpy as np
import nltk
import re
import hashlib
from typing import Tuple, Dict
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import plotly.express as px

# ================================
# SETUP NLTK (CACHE)
# ================================
@st.cache_resource
def setup_nltk():
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)

    return set(stopwords.words("english")), PorterStemmer()

stop_words, stemmer = setup_nltk()

# ================================
# DATA VALIDATION
# ================================
def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, str]:
    required_columns = ["title", "text", "subject", "status"]

    if df.empty:
        return False, "DataFrame is empty"

    df.columns = df.columns.str.strip().str.lower()
    missing = [c for c in required_columns if c not in df.columns]

    if missing:
        return False, f"Missing required columns: {', '.join(missing)}"

    return True, "OK"

def validate_column_data(df: pd.DataFrame) -> Dict:
    stats = {
        "empty_titles": df["title"].isna().sum(),
        "empty_texts": df["text"].isna().sum(),
        "invalid_status": (~df["status"].astype(str).isin(
            ["real", "fake", "true", "false", "0", "1", "Real", "Fake"]
        )).sum()
    }
    return stats

# ================================
# PREPROCESSING
# ================================
def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    original = len(df)

    df.columns = df.columns.str.strip().str.lower()
    df = df.dropna(subset=["title", "text", "status"])
    df["subject"] = df["subject"].fillna("Unknown")
    df = df.drop_duplicates(subset=["title", "text"])

    removed = original - len(df)
    return df, removed

def clean_and_stem(text: str) -> str:
    if not isinstance(text, str) or text.strip() == "":
        return ""

    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [stemmer.stem(t) for t in tokens]

    return " ".join(tokens)

# ================================
# MAIN APP
# ================================
def upload_page():
    st.title("📤 Dataset Upload & Preprocessing")
    st.caption("Upload CSV or Excel files and automatically clean the text data")

    st.subheader("📋 Upload Instructions")
    st.markdown("""
    - Supported formats: **CSV, XLS, XLSX**
    - Required columns: `title`, `text`, `subject`, `status`
    - Automatic cleaning: lowercase, punctuation removal, stopwords, stemming
    - Rows with missing values are removed
    """)

    uploaded_file = st.file_uploader(
        "Choose a dataset file",
        type=["csv", "xls", "xlsx"]
    )

    if uploaded_file is None:
        st.info("Please upload a file to begin.")
        return

    try:
        with st.spinner("Loading file..."):
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file, encoding="utf-8", on_bad_lines="skip")
            else:
                df = pd.read_excel(uploaded_file)

        st.success(f"Loaded file: {uploaded_file.name}")

        # Validate structure
        valid, msg = validate_dataframe(df)
        if not valid:
            st.error(msg)
            st.write("Detected columns:", df.columns.tolist())
            return

        # Validate content
        validation_stats = validate_column_data(df)
        if any(validation_stats.values()):
            st.warning(f"""
            Data quality issues detected:
            - Empty titles: {validation_stats['empty_titles']}
            - Empty texts: {validation_stats['empty_texts']}
            - Invalid status values: {validation_stats['invalid_status']}
            """)

        # Stats before
        st.subheader("📊 Dataset Statistics")
        col1, col2 = st.columns(2)
        col1.metric("Rows Before Cleaning", df.shape[0])

        # Preprocess
        df_clean, removed = preprocess_data(df)
        col2.metric("Rows After Cleaning", df_clean.shape[0])

        if df_clean.empty:
            st.error("No valid data left after preprocessing.")
            return

        if removed > 0:
            st.warning(f"{removed} rows were removed during preprocessing.")

        # Preview
        st.subheader("🔍 Preprocessing Preview")

        preview = df_clean.head(5).copy()
        preview["title_clean"] = preview["title"].apply(clean_and_stem)
        preview["text_clean"] = preview["text"].apply(clean_and_stem)

        tab1, tab2 = st.tabs(["Original", "Cleaned"])

        with tab1:
            st.dataframe(preview[["title", "text", "subject", "status"]])

        with tab2:
            st.dataframe(preview[["title_clean", "text_clean", "subject", "status"]])

        # Visualization
        st.subheader("📊 Data Distribution")

        col1, col2 = st.columns(2)

        with col1:
            status_dist = df_clean["status"].value_counts()
            fig1 = px.pie(
                values=status_dist.values,
                names=status_dist.index,
                title="Status Distribution"
            )
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            subject_dist = df_clean["subject"].value_counts().head(10)
            fig2 = px.bar(
                x=subject_dist.index,
                y=subject_dist.values,
                title="Top 10 Subjects",
                labels={"x": "Subject", "y": "Count"}
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Download
        st.subheader("⬇️ Download Cleaned Dataset")
        st.download_button(
            label="Download CSV",
            data=df_clean.to_csv(index=False),
            file_name="cleaned_dataset.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.exception(e)

# ================================
# RUN
# ================================
if __name__ == "__main__":
    upload_page()
