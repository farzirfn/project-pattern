import streamlit as st
import pandas as pd
from typing import Tuple

def show():
    """Upload Dataset Page"""
    
    st.markdown('<h1 class="upload-title">📤 Upload Dataset</h1>', unsafe_allow_html=True)
    st.markdown('<p class="upload-subtitle">Upload your dataset to begin the fake news detection process</p>', unsafe_allow_html=True)
    
    # Instructions
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h4>📋 Upload Instructions</h4>
            <ul>
                <li>✅ Supported formats: CSV, XLS, XLSX</li>
                <li>✅ Required columns: <code>title</code>, <code>text</code>, <code>status</code></li>
                <li>✅ Optional column: <code>subject</code></li>
                <li>✅ Status should contain: 'real', 'fake', 'true', 'false', '0', '1'</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if st.session_state.dataset is not None:
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-label">Current Dataset</div>
                <div class="stats-value">{len(st.session_state.dataset):,}</div>
                <div class="stats-change">📊 Total Rows</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # File uploader
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
                    <strong>Required columns:</strong> title, text, status
                </div>
                """, unsafe_allow_html=True)
                return
            
            # Show dataset info
            st.markdown("### 📊 Dataset Information")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="stats-card">
                    <div class="stats-label">Total Rows</div>
                    <div class="stats-value">{df.shape[0]:,}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="stats-card">
                    <div class="stats-label">Columns</div>
                    <div class="stats-value">{df.shape[1]}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                missing_count = df.isnull().sum().sum()
                st.markdown(f"""
                <div class="stats-card">
                    <div class="stats-label">Missing Values</div>
                    <div class="stats-value">{missing_count:,}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                duplicate_count = df.duplicated().sum()
                st.markdown(f"""
                <div class="stats-card">
                    <div class="stats-label">Duplicates</div>
                    <div class="stats-value">{duplicate_count:,}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Data preview
            st.markdown("### 🔍 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Column information
            st.markdown("### 📋 Column Information")
            
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.values,
                'Non-Null Count': df.count().values,
                'Null Count': df.isnull().sum().values,
                'Unique Values': [df[col].nunique() for col in df.columns]
            })
            
            st.dataframe(col_info, use_container_width=True, hide_index=True)
            
            # Data distribution
            if 'status' in df.columns:
                st.markdown("### 📊 Status Distribution")
                
                status_counts = df['status'].value_counts()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.bar_chart(status_counts)
                
                with col2:
                    for status, count in status_counts.items():
                        percentage = (count / len(df)) * 100
                        st.markdown(f"**{status}:** {count:,} ({percentage:.1f}%)")
            
            # Save button
            st.markdown("<br>", unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col2:
                if st.button("💾 Save Dataset", use_container_width=True, type="primary"):
                    # Store in session state
                    st.session_state.dataset = df
                    st.session_state.preprocessed_data = None  # Reset preprocessing
                    st.session_state.trained_model = None  # Reset model
                    st.session_state.vectorizer = None
                    st.session_state.model_metrics = None
                    
                    st.success("✅ Dataset saved successfully!")
                    st.balloons()
                    
                    st.info("👉 Next step: Go to **Preprocess Data** page to clean your dataset")
        
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
            <p style="color: #999; font-size: 0.9rem;">Required columns: title, text, status</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show current dataset if exists
        if st.session_state.dataset is not None:
            st.markdown("### 📊 Current Dataset")
            st.dataframe(st.session_state.dataset.head(10), use_container_width=True)
            
            if st.button("🗑️ Clear Current Dataset", type="secondary"):
                st.session_state.dataset = None
                st.session_state.preprocessed_data = None
                st.session_state.trained_model = None
                st.session_state.vectorizer = None
                st.session_state.model_metrics = None
                st.rerun()


def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate dataframe has required columns
    Returns: (is_valid, error_message)
    """
    required_columns = ['title', 'text', 'status']
    
    # Check if dataframe is empty
    if df.empty:
        return False, "❌ DataFrame is empty"
    
    # Strip whitespace from column names
    df.columns = df.columns.str.strip().str.lower()
    
    # Check required columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    
    if missing_cols:
        return False, f"❌ Missing required columns: {', '.join(missing_cols)}"
    
    return True, "✅ Validation passed"


# Add custom CSS
st.markdown("""
<style>
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
    
    .upload-zone {
        border: 2px dashed #667eea;
        border-radius: 16px;
        padding: 3rem;
        text-align: center;
        background: linear-gradient(135deg, #667eea10 0%, #764ba210 100%);
        margin: 2rem 0;
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
    
    .info-box {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .error-box {
        background: linear-gradient(135deg, #e4575615 0%, #dc354515 100%);
        border-left: 4px solid #e45756;
        padding: 1.5rem;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)