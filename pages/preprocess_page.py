import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from typing import Dict

# Initialize NLTK
@st.cache_resource
def setup_nltk():
    """Download NLTK data if not already present"""
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download("stopwords", quiet=True)
    
    return set(stopwords.words("english")), PorterStemmer()

stop_words, stemmer = setup_nltk()


def show():
    """Preprocess Data Page"""
    
    st.markdown('<h1 class="preprocess-title">⚙️ Preprocess Data</h1>', unsafe_allow_html=True)
    st.markdown('<p class="preprocess-subtitle">Clean and prepare your dataset for training</p>', unsafe_allow_html=True)
    
    if st.session_state.dataset is None:
        st.warning("⚠️ No dataset uploaded. Please upload a dataset first.")
        if st.button("📤 Go to Upload Page"):
            st.session_state.page = 'upload'
            st.rerun()
        return
    
    df = st.session_state.dataset.copy()
    
    # Show current dataset info
    st.markdown("### 📊 Current Dataset")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-label">Total Rows</div>
            <div class="stats-value">{df.shape[0]:,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        missing_count = df.isnull().sum().sum()
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-label">Missing Values</div>
            <div class="stats-value">{missing_count:,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        duplicate_count = df.duplicated().sum()
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-label">Duplicates</div>
            <div class="stats-value">{duplicate_count:,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        if 'status' in df.columns:
            unique_status = df['status'].nunique()
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-label">Unique Status</div>
                <div class="stats-value">{unique_status}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Preprocessing Options
    st.markdown("### ⚙️ Preprocessing Options")
    st.markdown("Choose which cleaning operations to apply to your data:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📝 Text Cleaning")
        
        lowercase = st.checkbox(
            "Convert to lowercase",
            value=True,
            help="Convert all text to lowercase (e.g., 'Hello World' → 'hello world')"
        )
        
        remove_punctuation = st.checkbox(
            "Remove punctuation",
            value=True,
            help="Remove punctuation and special characters (e.g., 'Hello, World!' → 'Hello World')"
        )
        
        remove_stopwords = st.checkbox(
            "Remove stopwords",
            value=True,
            help="Remove common words like 'the', 'is', 'and', etc."
        )
        
        stemming = st.checkbox(
            "Apply stemming",
            value=True,
            help="Reduce words to their root form (e.g., 'running' → 'run')"
        )
        
        remove_numbers = st.checkbox(
            "Remove numbers",
            value=False,
            help="Remove all numeric characters"
        )
        
        remove_urls = st.checkbox(
            "Remove URLs",
            value=True,
            help="Remove web URLs from text"
        )
    
    with col2:
        st.markdown("#### 🧹 Data Cleaning")
        
        remove_missing = st.checkbox(
            "Remove rows with missing values",
            value=True,
            help="Remove rows that have empty title, text, or status fields"
        )
        
        remove_duplicates = st.checkbox(
            "Remove duplicate entries",
            value=True,
            help="Remove rows with duplicate title and text combinations"
        )
        
        normalize_status = st.checkbox(
            "Normalize status values",
            value=True,
            help="Convert status to 0 (fake) and 1 (real)"
        )
        
        # Quick preset buttons
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Quick Presets:**")
        
        preset_col1, preset_col2, preset_col3 = st.columns(3)
        
        with preset_col1:
            if st.button("🔥 Full Clean", use_container_width=True):
                lowercase = remove_punctuation = remove_stopwords = stemming = True
                remove_urls = normalize_status = True
                remove_missing = remove_duplicates = True
                st.session_state['preset_applied'] = 'full'
        
        with preset_col2:
            if st.button("⚡ Standard", use_container_width=True):
                lowercase = remove_punctuation = remove_stopwords = True
                stemming = remove_urls = normalize_status = True
                remove_missing = remove_duplicates = True
                remove_numbers = False
                st.session_state['preset_applied'] = 'standard'
        
        with preset_col3:
            if st.button("🍃 Minimal", use_container_width=True):
                lowercase = remove_punctuation = remove_stopwords = stemming = False
                remove_numbers = remove_urls = False
                remove_missing = remove_duplicates = normalize_status = False
                st.session_state['preset_applied'] = 'minimal'
    
    # Store cleaning options
    cleaning_options = {
        'lowercase': lowercase,
        'remove_punctuation': remove_punctuation,
        'remove_stopwords': remove_stopwords,
        'stemming': stemming,
        'remove_numbers': remove_numbers,
        'remove_urls': remove_urls
    }
    
    # Show selected options summary
    selected_text_options = [k.replace('_', ' ').title() for k, v in cleaning_options.items() if v]
    data_options = []
    if remove_missing:
        data_options.append("Remove missing values")
    if remove_duplicates:
        data_options.append("Remove duplicates")
    if normalize_status:
        data_options.append("Normalize status")
    
    all_options = selected_text_options + data_options
    
    if all_options:
        st.markdown(f"""
        <div class="info-box">
            <strong>✅ Active preprocessing:</strong> {', '.join(all_options)}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="warning-box">
            <strong>⚠️ No preprocessing selected:</strong> Data will be saved as-is
        </div>
        """, unsafe_allow_html=True)
    
    # Preview preprocessing
    st.markdown("### 🔍 Preview Preprocessing")
    
    if st.button("👁️ Preview Changes", type="secondary"):
        with st.spinner("Processing preview..."):
            # Apply preprocessing
            df_preview = preprocess_dataframe(
                df.copy(), 
                cleaning_options, 
                remove_missing, 
                remove_duplicates,
                normalize_status
            )
            
            # Show comparison
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Original Data (First 3 rows)**")
                st.dataframe(df.head(3), use_container_width=True)
            
            with col2:
                st.markdown("**Preprocessed Data (First 3 rows)**")
                st.dataframe(df_preview.head(3), use_container_width=True)
            
            # Show statistics
            st.markdown("### 📊 Preprocessing Statistics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="stats-card">
                    <div class="stats-label">Before</div>
                    <div class="stats-value">{len(df):,}</div>
                    <div class="stats-change">Rows</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="stats-card" style="border-left-color: #28a745;">
                    <div class="stats-label">After</div>
                    <div class="stats-value" style="color: #28a745;">{len(df_preview):,}</div>
                    <div class="stats-change">Rows</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                removed = len(df) - len(df_preview)
                removed_pct = (removed / len(df) * 100) if len(df) > 0 else 0
                color = "#e45756" if removed > 0 else "#28a745"
                st.markdown(f"""
                <div class="stats-card" style="border-left-color: {color};">
                    <div class="stats-label">Removed</div>
                    <div class="stats-value" style="color: {color};">{removed:,}</div>
                    <div class="stats-change">{removed_pct:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Show example transformations
            st.markdown("### 📝 Text Transformation Examples")
            
            if len(df_preview) > 0 and 'text' in df.columns:
                for i in range(min(2, len(df_preview))):
                    with st.expander(f"Example {i+1}: {df['title'].iloc[i][:50]}..."):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Original Text:**")
                            st.text_area("", df['text'].iloc[i][:300], height=150, key=f"orig_{i}", disabled=True)
                        with col2:
                            st.markdown("**Cleaned Text:**")
                            st.text_area("", df_preview['text'].iloc[i][:300], height=150, key=f"clean_{i}", disabled=True)
    
    # Apply preprocessing button
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 💾 Apply Preprocessing")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if st.button("🚀 Apply and Save", use_container_width=True, type="primary"):
            with st.spinner("Preprocessing data..."):
                # Apply preprocessing
                df_clean = preprocess_dataframe(
                    df.copy(), 
                    cleaning_options, 
                    remove_missing, 
                    remove_duplicates,
                    normalize_status
                )
                
                if len(df_clean) == 0:
                    st.error("❌ No data remaining after preprocessing! Please adjust your options.")
                    return
                
                # Save to session state
                st.session_state.preprocessed_data = df_clean
                st.session_state.cleaning_options = cleaning_options
                st.session_state.trained_model = None  # Reset model
                st.session_state.vectorizer = None
                st.session_state.model_metrics = None
                
                st.success("✅ Data preprocessed successfully!")
                st.balloons()
                
                # Show final statistics
                st.markdown("### 📊 Final Statistics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Rows", f"{len(df_clean):,}")
                
                with col2:
                    st.metric("Missing Values", f"{df_clean.isnull().sum().sum():,}")
                
                with col3:
                    st.metric("Duplicates", f"{df_clean.duplicated().sum():,}")
                
                with col4:
                    if 'status' in df_clean.columns:
                        st.metric("Unique Status", f"{df_clean['status'].nunique()}")
                
                st.info("👉 Next step: Go to **Train Model** page to train your classifier")


def clean_text(text: str, options: Dict[str, bool]) -> str:
    """Clean text based on selected options"""
    if not isinstance(text, str) or text.strip() == "":
        return ""
    
    original_text = text
    
    # Remove URLs
    if options.get('remove_urls', False):
        text = re.sub(r'http\S+|www.\S+', '', text)
    
    # Remove numbers
    if options.get('remove_numbers', False):
        text = re.sub(r'\d+', '', text)
    
    # Lowercase
    if options.get('lowercase', True):
        text = text.lower()
    
    # Remove punctuation
    if options.get('remove_punctuation', True):
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Tokenize
    tokens = text.split()
    
    # Remove stopwords
    if options.get('remove_stopwords', True):
        tokens = [t for t in tokens if t and t.lower() not in stop_words]
    
    # Stemming
    if options.get('stemming', True):
        try:
            tokens = [stemmer.stem(token) for token in tokens]
        except Exception:
            pass
    
    # Rejoin
    result = " ".join(tokens)
    
    # Return original if cleaning resulted in empty string
    return result.strip() if result.strip() else original_text[:100]


def preprocess_dataframe(df: pd.DataFrame, cleaning_options: Dict, 
                        remove_missing: bool, remove_duplicates: bool,
                        normalize_status: bool) -> pd.DataFrame:
    """Apply all preprocessing steps to dataframe"""
    
    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()
    
    # Remove missing values
    if remove_missing:
        df = df.dropna(subset=['title', 'text', 'status'], how='any')
        if 'subject' in df.columns:
            df['subject'] = df['subject'].fillna('Unknown')
    else:
        df['title'] = df['title'].fillna('Untitled')
        df['text'] = df['text'].fillna('')
        df['status'] = df['status'].fillna('Unknown')
        if 'subject' in df.columns:
            df['subject'] = df['subject'].fillna('Unknown')
    
    # Remove duplicates
    if remove_duplicates:
        df = df.drop_duplicates(subset=['title', 'text'], keep='first')
    
    # Clean text columns
    df['title'] = df['title'].apply(lambda x: clean_text(str(x), cleaning_options))
    df['text'] = df['text'].apply(lambda x: clean_text(str(x), cleaning_options))
    if 'subject' in df.columns:
        df['subject'] = df['subject'].apply(lambda x: clean_text(str(x), cleaning_options))
    
    # Normalize status
    if normalize_status:
        status_mapping = {
            'fake': 0, 'false': 0, '0': 0, 'Fake': 0, 'False': 0,
            'real': 1, 'true': 1, '1': 1, 'Real': 1, 'True': 1
        }
        df['status'] = df['status'].astype(str).map(status_mapping)
        df = df.dropna(subset=['status'])  # Remove unmapped status values
        df['status'] = df['status'].astype(int)
    
    # Remove rows with empty text after cleaning
    df = df[df['text'].str.len() > 10]
    df = df[df['title'].str.len() > 3]
    
    # Reset index
    df = df.reset_index(drop=True)
    
    return df


# Custom CSS
st.markdown("""
<style>
    .preprocess-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .preprocess-subtitle {
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    .stats-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
        border-left: 4px solid #667eea;
    }
    
    .stats-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
        margin: 0.5rem 0;
    }
    
    .stats-label {
        color: #666;
        font-size: 0.9rem;
        font-weight: 500;
        text-transform: uppercase;
    }
    
    .stats-change {
        color: #28a745;
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }
    
    .info-box {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #ffc10715 0%, #fb6f9215 100%);
        border-left: 4px solid #ffc107;
        padding: 1.5rem;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)