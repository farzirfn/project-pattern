import streamlit as st
import pandas as pd
import mysql.connector
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import plotly.graph_objects as go
import plotly.express as px
from typing import Tuple, Dict
import hashlib

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(
    page_title="Upload Dataset",
    page_icon="📤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================
# CUSTOM CSS
# ================================
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Main container */
    .main {
        padding: 2rem;
    }
    
    /* Title styling */
    .upload-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .upload-subtitle {
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Upload area */
    .upload-zone {
        border: 2px dashed #667eea;
        border-radius: 16px;
        padding: 3rem;
        text-align: center;
        background: linear-gradient(135deg, #667eea10 0%, #764ba210 100%);
        margin: 2rem 0;
        transition: all 0.3s ease;
    }
    
    .upload-zone:hover {
        border-color: #764ba2;
        background: linear-gradient(135deg, #667eea20 0%, #764ba220 100%);
    }
    
    /* Stats cards */
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
    
    /* Info boxes */
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
    }
    
    .success-box {
        background: linear-gradient(135deg, #28a74515 0%, #20c99715 100%);
        border-left: 4px solid #28a745;
    }
    
    .error-box {
        background: linear-gradient(135deg, #e4575615 0%, #dc354515 100%);
        border-left: 4px solid #e45756;
    }
    
    /* Preprocessing options card */
    .preprocessing-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    /* Process steps */
    .step-container {
        display: flex;
        align-items: center;
        margin: 1rem 0;
        padding: 1rem;
        background: white;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .step-number {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 1.2rem;
        margin-right: 1rem;
    }
    
    .step-content {
        flex: 1;
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# ================================
# NLTK SETUP
# ================================
@st.cache_resource
def setup_nltk():
    """Download NLTK data if not already present"""
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download("stopwords", quiet=True)
    
    return set(stopwords.words("english")), PorterStemmer()

stop_words, stemmer = setup_nltk()

# ================================
# DATABASE CONNECTION
# ================================
def create_connection():
    """Create database connection with error handling"""
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="fyp",
            autocommit=False
        )
        return conn
    except mysql.connector.Error as e:
        st.error(f"❌ Database connection failed: {str(e)}")
        return None

# ================================
# DATA VALIDATION
# ================================
def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate dataframe has required columns
    Returns: (is_valid, error_message)
    """
    required_columns = ['title', 'text', 'subject', 'status']
    
    if df.empty:
        return False, "❌ DataFrame is empty"
    
    df.columns = df.columns.str.strip().str.lower()
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    
    if missing_cols:
        return False, f"❌ Missing required columns: {', '.join(missing_cols)}"
    
    return True, "✅ Validation passed"

def validate_column_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Validate and clean column data
    Returns: (cleaned_df, stats_dict)
    """
    stats = {
        'empty_titles': 0,
        'empty_texts': 0,
        'invalid_status': 0
    }
    
    stats['empty_titles'] = df['title'].isna().sum()
    stats['empty_texts'] = df['text'].isna().sum()
    
    valid_statuses = ['real', 'fake', 'true', 'false', '0', '1', 'Real', 'Fake', 'True', 'False']
    stats['invalid_status'] = (~df['status'].astype(str).isin(valid_statuses)).sum()
    
    return df, stats

# ================================
# DATA PREPROCESSING WITH OPTIONS
# ================================
def preprocess_data(df: pd.DataFrame, remove_missing: bool = True, remove_duplicates: bool = True) -> Tuple[pd.DataFrame, int]:
    """Remove rows with missing values and duplicates based on user options"""
    original_count = len(df)
    
    df.columns = df.columns.str.strip().str.lower()
    
    # Remove rows with missing values (optional)
    if remove_missing:
        df = df.dropna(subset=['title', 'text', 'status'], how='any')
        df['subject'] = df['subject'].fillna('Unknown')
    else:
        # Just fill missing values but keep the rows
        df['title'] = df['title'].fillna('Untitled')
        df['text'] = df['text'].fillna('')
        df['subject'] = df['subject'].fillna('Unknown')
        df['status'] = df['status'].fillna('Pending')
    
    # Remove duplicates (optional)
    if remove_duplicates:
        df = df.drop_duplicates(subset=['title', 'text'], keep='first')
    
    removed_count = original_count - len(df)
    
    return df, removed_count

def clean_and_stem(text: str, options: Dict[str, bool]) -> str:
    """
    Clean text based on selected options
    
    Options:
    - lowercase: Convert to lowercase
    - remove_punctuation: Remove punctuation and special characters
    - remove_stopwords: Remove common stopwords
    - stemming: Apply Porter stemming
    """
    if not isinstance(text, str) or text.strip() == "":
        return ""
    
    original_text = text
    
    # Lowercase (optional)
    if options.get('lowercase', True):
        text = text.lower()
    
    # Remove punctuation (optional)
    if options.get('remove_punctuation', True):
        text = re.sub(r"[^a-zA-Z\s]", "", text)
    
    # Tokenize
    tokens = text.split()
    
    # Remove stopwords (optional)
    if options.get('remove_stopwords', True):
        tokens = [t for t in tokens if t and t.lower() not in stop_words]
    
    # Stem each token (optional)
    if options.get('stemming', True):
        try:
            tokens = [stemmer.stem(token) for token in tokens]
        except Exception:
            pass  # Fallback if stemming fails
    
    # Rejoin
    result = " ".join(tokens)
    
    # Return original if cleaning resulted in empty string
    return result if result.strip() else original_text[:100]

def generate_content_hash(title: str, text: str) -> str:
    """Generate hash for duplicate detection"""
    content = f"{title}|{text}".lower().strip()
    return hashlib.md5(content.encode()).hexdigest()

# ================================
# DATABASE OPERATIONS
# ================================
def get_database_stats() -> Dict:
    """Get current database statistics"""
    try:
        conn = create_connection()
        if not conn:
            return {'total': 0, 'by_status': []}
        
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("SELECT COUNT(*) as total FROM dataset")
        total = cursor.fetchone()['total']
        
        cursor.execute("SELECT status, COUNT(*) as count FROM dataset GROUP BY status")
        by_status = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return {'total': total, 'by_status': by_status}
    except Exception as e:
        st.warning(f"⚠️ Could not fetch database stats: {str(e)}")
        return {'total': 0, 'by_status': []}

def save_to_database(df: pd.DataFrame, cleaning_options: Dict[str, bool]) -> Tuple[bool, str, Dict]:
    """
    Save dataframe to database with transaction support and custom cleaning options
    Returns: (success, message, stats)
    """
    conn = create_connection()
    if not conn:
        return False, "❌ Database connection failed", {}
    
    try:
        cursor = conn.cursor()
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        stats = {
            'attempted': len(df),
            'inserted': 0,
            'duplicates': 0,
            'errors': 0
        }
        
        status_text.text("🔄 Checking for duplicates...")
        
        for i, (_, row) in enumerate(df.iterrows(), start=1):
            try:
                # Clean data with user-selected options
                title_clean = clean_and_stem(str(row.get("title", "Untitled")), cleaning_options)
                text_clean = clean_and_stem(str(row.get("text", "")), cleaning_options)
                subject_clean = clean_and_stem(str(row.get("subject", "Unknown")), cleaning_options)
                status_val = str(row.get("status", "Pending")).strip()
                
                # Validate cleaned data
                if not title_clean or len(title_clean) < 3:
                    stats['errors'] += 1
                    continue
                
                if not text_clean or len(text_clean) < 10:
                    stats['errors'] += 1
                    continue
                
                # Check for duplicates
                content_hash = generate_content_hash(title_clean, text_clean)
                
                cursor.execute(
                    "SELECT COUNT(*) FROM dataset WHERE MD5(CONCAT(title, '|', text)) = %s",
                    (content_hash,)
                )
                
                if cursor.fetchone()[0] > 0:
                    stats['duplicates'] += 1
                else:
                    # Insert
                    cursor.execute(
                        """INSERT INTO dataset (title, text, subject, status) 
                           VALUES (%s, %s, %s, %s)""",
                        (title_clean, text_clean, subject_clean, status_val)
                    )
                    stats['inserted'] += 1
                
            except Exception as e:
                stats['errors'] += 1
                st.warning(f"⚠️ Error on row {i}: {str(e)}")
            
            # Update progress
            progress = i / len(df)
            progress_bar.progress(progress)
            
            if i % 10 == 0:
                status_text.text(
                    f"🔄 Processing... {i}/{len(df)} rows "
                    f"(Inserted: {stats['inserted']}, Duplicates: {stats['duplicates']}, Errors: {stats['errors']})"
                )
        
        # Commit transaction
        conn.commit()
        
        # Get final count
        cursor.execute("SELECT COUNT(*) FROM dataset")
        final_count = cursor.fetchone()[0]
        stats['total_in_db'] = final_count
        
        cursor.close()
        conn.close()
        
        progress_bar.empty()
        status_text.empty()
        
        if stats['inserted'] > 0:
            return True, "✅ Upload completed successfully", stats
        else:
            return False, "❌ No new records inserted", stats
        
    except mysql.connector.Error as e:
        conn.rollback()
        conn.close()
        return False, f"❌ Database error: {str(e)}", {}
    except Exception as e:
        conn.rollback()
        conn.close()
        return False, f"❌ Unexpected error: {str(e)}", {}

# ================================
# MAIN UPLOAD PAGE
# ================================
def upload_page():
    # Header
    st.markdown('<h1 class="upload-title">📤 Upload Dataset</h1>', unsafe_allow_html=True)
    st.markdown('<p class="upload-subtitle">Upload and preprocess your dataset with customizable cleaning options</p>', unsafe_allow_html=True)
    
    # Current database stats
    db_stats = get_database_stats()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h4>📋 Upload Instructions</h4>
            <ul>
                <li>✅ Supported formats: CSV, XLS, XLSX</li>
                <li>✅ Required columns: <code>title</code>, <code>text</code>, <code>subject</code>, <code>status</code></li>
                <li>✅ Customize preprocessing options below</li>
                <li>✅ Preview changes before saving</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-label">Current Database</div>
            <div class="stats-value">{db_stats['total']:,}</div>
            <div class="stats-change">📊 Total Records</div>
        </div>
        """, unsafe_allow_html=True)
    
    # File uploader
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📁 Select File to Upload")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV, XLS, or XLSX file",
        type=["csv", "xls", "xlsx"],
        help="Upload your dataset file. Supported formats: CSV, Excel (XLS, XLSX)"
    )
    
    if uploaded_file is not None:
        try:
            # Load file based on extension
            with st.spinner("📂 Loading file..."):
                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file, encoding='utf-8', on_bad_lines='skip')
                elif uploaded_file.name.endswith((".xls", ".xlsx")):
                    df = pd.read_excel(uploaded_file)
                else:
                    st.error("❌ Unsupported file type.")
                    return
            
            st.success(f"✅ File loaded: **{uploaded_file.name}**")
            
            # Validate dataframe structure
            is_valid, validation_msg = validate_dataframe(df)
            
            if not is_valid:
                st.markdown(f"""
                <div class="error-box">
                    {validation_msg}
                    <br><br>
                    <strong>Your columns:</strong> {', '.join(df.columns.tolist())}
                    <br>
                    <strong>Required columns:</strong> title, text, subject, status
                </div>
                """, unsafe_allow_html=True)
                return
            
            # Validate column data
            df_validated, validation_stats = validate_column_data(df)
            
            # Show validation warnings
            if any(validation_stats.values()):
                warning_messages = []
                if validation_stats['empty_titles'] > 0:
                    warning_messages.append(f"{validation_stats['empty_titles']} empty titles")
                if validation_stats['empty_texts'] > 0:
                    warning_messages.append(f"{validation_stats['empty_texts']} empty texts")
                if validation_stats['invalid_status'] > 0:
                    warning_messages.append(f"{validation_stats['invalid_status']} invalid status values")
                
                if warning_messages:
                    st.markdown(f"""
                    <div class="warning-box">
                        ⚠️ Data quality issues detected: {', '.join(warning_messages)}
                    </div>
                    """, unsafe_allow_html=True)
            
            # ================================
            # PREPROCESSING OPTIONS (NEW)
            # ================================
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
                
                # Quick preset buttons
                st.markdown("**Quick Presets:**")
                preset_col1, preset_col2 = st.columns(2)
                
                with preset_col1:
                    if st.button("🔥 Full Clean", use_container_width=True):
                        st.session_state['preset'] = 'full'
                        st.rerun()
                
                with preset_col2:
                    if st.button("🍃 Minimal Clean", use_container_width=True):
                        st.session_state['preset'] = 'minimal'
                        st.rerun()
            
            # Apply presets if selected
            if 'preset' in st.session_state:
                if st.session_state['preset'] == 'full':
                    lowercase = remove_punctuation = remove_stopwords = stemming = True
                    remove_missing = remove_duplicates = True
                elif st.session_state['preset'] == 'minimal':
                    lowercase = remove_punctuation = remove_stopwords = stemming = False
                    remove_missing = remove_duplicates = False
                del st.session_state['preset']
            
            # Store cleaning options
            cleaning_options = {
                'lowercase': lowercase,
                'remove_punctuation': remove_punctuation,
                'remove_stopwords': remove_stopwords,
                'stemming': stemming
            }
            
            # Show selected options summary
            selected_options = [k.replace('_', ' ').title() for k, v in cleaning_options.items() if v]
            data_options = []
            if remove_missing:
                data_options.append("Remove missing values")
            if remove_duplicates:
                data_options.append("Remove duplicates")
            
            if selected_options or data_options:
                st.markdown(f"""
                <div class="info-box">
                    <strong>✅ Active preprocessing:</strong> {', '.join(selected_options + data_options)}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="warning-box">
                    <strong>⚠️ No preprocessing selected:</strong> Data will be saved as-is
                </div>
                """, unsafe_allow_html=True)
            
            # Preprocess data with selected options
            df_clean, removed_count = preprocess_data(df, remove_missing, remove_duplicates)
            
            # Show stats AFTER preprocessing
            st.markdown("### 📊 Dataset Statistics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="stats-card">
                    <div class="stats-label">Before Preprocessing</div>
                    <div class="stats-value">{df.shape[0]:,}</div>
                    <div class="stats-change">Rows</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="stats-card" style="border-left-color: #28a745;">
                    <div class="stats-label">After Preprocessing</div>
                    <div class="stats-value" style="color: #28a745;">{df_clean.shape[0]:,}</div>
                    <div class="stats-change">Rows</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                removed_pct = (removed_count / df.shape[0] * 100) if df.shape[0] > 0 else 0
                color = "#e45756" if removed_count > 0 else "#28a745"
                st.markdown(f"""
                <div class="stats-card" style="border-left-color: {color};">
                    <div class="stats-label">Removed</div>
                    <div class="stats-value" style="color: {color};">{removed_count:,}</div>
                    <div class="stats-change">{removed_pct:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Check if data remains after cleaning
            if len(df_clean) == 0:
                st.markdown("""
                <div class="error-box">
                    ❌ <strong>No valid data remaining after preprocessing!</strong>
                    <br>All rows were removed. Try adjusting your preprocessing options.
                </div>
                """, unsafe_allow_html=True)
                return
            
            # Show warning if too many rows were removed
            if removed_count > 0:
                st.markdown(f"""
                <div class="warning-box">
                    ⚠️ <strong>{removed_count:,} rows</strong> ({removed_pct:.1f}%) were removed
                </div>
                """, unsafe_allow_html=True)
            
            # Preprocessing preview
            st.markdown("### 🔍 Preprocessing Preview")
            st.markdown("See how your data will be cleaned before saving:")
            
            # Create preview with before/after
            preview_size = min(5, len(df_clean))
            preview_df = df_clean.head(preview_size).copy()
            preview_df["title_clean"] = preview_df["title"].apply(lambda x: clean_and_stem(x, cleaning_options))
            preview_df["text_clean"] = preview_df["text"].apply(lambda x: clean_and_stem(x, cleaning_options))
            preview_df["subject_clean"] = preview_df["subject"].apply(lambda x: clean_and_stem(x, cleaning_options))
            
            # Show in tabs
            tab1, tab2, tab3 = st.tabs(["📝 Original Data", "✨ Cleaned Data", "🔄 Side-by-Side Comparison"])
            
            with tab1:
                st.dataframe(
                    df_clean[["title", "text", "subject", "status"]].head(10),
                    use_container_width=True
                )
            
            with tab2:
                cleaned_preview = pd.DataFrame({
                    "title": preview_df["title_clean"],
                    "text": preview_df["text_clean"],
                    "subject": preview_df["subject_clean"],
                    "status": preview_df["status"]
                })
                st.dataframe(cleaned_preview, use_container_width=True)
            
            with tab3:
                if len(preview_df) > 0:
                    for idx in range(min(2, len(preview_df))):  # Show 2 examples
                        st.markdown(f"**Example {idx + 1}:**")
                        comparison = pd.DataFrame({
                            "Field": ["Title", "Text", "Subject"],
                            "Original": [
                                str(preview_df["title"].iloc[idx])[:80] + ("..." if len(str(preview_df["title"].iloc[idx])) > 80 else ""),
                                str(preview_df["text"].iloc[idx])[:80] + ("..." if len(str(preview_df["text"].iloc[idx])) > 80 else ""),
                                str(preview_df["subject"].iloc[idx])[:80] + ("..." if len(str(preview_df["subject"].iloc[idx])) > 80 else "")
                            ],
                            "Cleaned": [
                                str(preview_df["title_clean"].iloc[idx])[:80] + ("..." if len(str(preview_df["title_clean"].iloc[idx])) > 80 else ""),
                                str(preview_df["text_clean"].iloc[idx])[:80] + ("..." if len(str(preview_df["text_clean"].iloc[idx])) > 80 else ""),
                                str(preview_df["subject_clean"].iloc[idx])[:80] + ("..." if len(str(preview_df["subject_clean"].iloc[idx])) > 80 else "")
                            ]
                        })
                        st.dataframe(comparison, use_container_width=True, hide_index=True)
                        st.markdown("---")
            
            # Data distribution visualization
            st.markdown("### 📊 Data Distribution")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Distribution by status
                status_dist = df_clean['status'].value_counts()
                fig_status = px.pie(
                    values=status_dist.values,
                    names=status_dist.index,
                    title='Distribution by Status',
                    color_discrete_sequence=px.colors.sequential.RdBu,
                    hole=0.4
                )
                fig_status.update_traces(textposition='inside', textinfo='percent+label')
                fig_status.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_status, use_container_width=True)
            
            with col2:
                # Distribution by subject
                subject_dist = df_clean['subject'].value_counts().head(10)
                fig_subject = px.bar(
                    x=subject_dist.index,
                    y=subject_dist.values,
                    title='Top 10 Subjects',
                    labels={'x': 'Subject', 'y': 'Count'},
                    color=subject_dist.values,
                    color_continuous_scale='Viridis'
                )
                fig_subject.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
                fig_subject.update_xaxes(tickangle=45)
                st.plotly_chart(fig_subject, use_container_width=True)
            
            # Save to database button
            st.markdown("### 💾 Save to Database")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🚀 Save to Database", use_container_width=True):
                    success, message, stats = save_to_database(df_clean, cleaning_options)
                    
                    if success:
                        # Success message with detailed stats
                        st.markdown(f"""
                        <div class="success-box">
                            <h3>✅ Upload Successful!</h3>
                            <p><strong>{stats['inserted']:,}</strong> new records have been processed and saved to the database.</p>
                            <hr>
                            <p><strong>📊 Upload Summary:</strong></p>
                            <ul>
                                <li>Total attempted: {stats['attempted']:,}</li>
                                <li>Successfully inserted: {stats['inserted']:,}</li>
                                <li>Duplicates skipped: {stats['duplicates']:,}</li>
                                <li>Errors: {stats['errors']:,}</li>
                                <li>Total records in database: {stats['total_in_db']:,}</li>
                            </ul>
                            <p><strong>🔧 Applied preprocessing:</strong> {', '.join(selected_options + data_options) if selected_options or data_options else 'None'}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.balloons()
                    else:
                        # Error or warning message
                        if stats:
                            st.markdown(f"""
                            <div class="warning-box">
                                <h3>⚠️ Upload Completed with Warnings</h3>
                                <p>{message}</p>
                                <hr>
                                <p><strong>📊 Upload Summary:</strong></p>
                                <ul>
                                    <li>Total attempted: {stats.get('attempted', 0):,}</li>
                                    <li>Successfully inserted: {stats.get('inserted', 0):,}</li>
                                    <li>Duplicates skipped: {stats.get('duplicates', 0):,}</li>
                                    <li>Errors: {stats.get('errors', 0):,}</li>
                                </ul>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.error(message)
        
        except pd.errors.EmptyDataError:
            st.error("❌ The file is empty or corrupted.")
        except pd.errors.ParserError:
            st.error("❌ Error parsing file. Please check the file format.")
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            st.exception(e)
    
    else:
        # Show empty state
        st.markdown("""
        <div class="upload-zone">
            <h2>📁 No File Selected</h2>
            <p>Click "Browse files" above to upload your dataset</p>
            <p style="color: #999; font-size: 0.9rem;">Supported formats: CSV, XLS, XLSX</p>
            <p style="color: #999; font-size: 0.9rem;">Required columns: title, text, subject, status</p>
        </div>
        """, unsafe_allow_html=True)

# ================================
# RUN PAGE
# ================================
if __name__ == "__main__":
    upload_page()