import streamlit as st

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="🔍",
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
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] .css-1d391kg {
        color: white;
    }
    
    /* Title styling */
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .main-subtitle {
        color: #666;
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 3rem;
    }
    
    /* Navigation cards */
    .nav-card {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
        transition: all 0.3s ease;
        border: 2px solid transparent;
        height: 100%;
        cursor: pointer;
    }
    
    .nav-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        border-color: #667eea;
    }
    
    .nav-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
    }
    
    .nav-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #333;
        margin-bottom: 0.5rem;
    }
    
    .nav-description {
        color: #666;
        font-size: 0.95rem;
    }
    
    /* Feature box */
    .feature-box {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ================================
# SESSION STATE INITIALIZATION
# ================================
if 'page' not in st.session_state:
    st.session_state.page = 'home'

if 'dataset' not in st.session_state:
    st.session_state.dataset = None

if 'preprocessed_data' not in st.session_state:
    st.session_state.preprocessed_data = None

if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None

if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None

if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = None

# ================================
# NAVIGATION
# ================================
def home_page():
    st.markdown('<h1 class="main-title">🔍 Fake News Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="main-subtitle">Upload, preprocess, train, and predict - all in one platform</p>', unsafe_allow_html=True)
    
    # System workflow
    st.markdown("### 📋 System Workflow")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="nav-card">
            <div class="nav-icon">📤</div>
            <div class="nav-title">1. Upload</div>
            <div class="nav-description">Upload your dataset (CSV/Excel)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="nav-card">
            <div class="nav-icon">⚙️</div>
            <div class="nav-title">2. Preprocess</div>
            <div class="nav-description">Clean and prepare your data</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="nav-card">
            <div class="nav-icon">🤖</div>
            <div class="nav-title">3. Train</div>
            <div class="nav-description">Train machine learning models</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="nav-card">
            <div class="nav-icon">🎯</div>
            <div class="nav-title">4. Predict</div>
            <div class="nav-description">Make predictions on new data</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Quick start guide
    st.markdown("### 🚀 Quick Start Guide")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="feature-box">
            <h4>📚 How to Use This System:</h4>
            <ol>
                <li><strong>Upload Dataset:</strong> Start by uploading your dataset with news articles</li>
                <li><strong>Preprocess Data:</strong> Choose preprocessing options and clean your data</li>
                <li><strong>Train Model:</strong> Select algorithm and train your model</li>
                <li><strong>Make Predictions:</strong> Use the trained model to detect fake news</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <h4>📋 Dataset Requirements:</h4>
            <ul>
                <li>✅ Required columns: <code>title</code>, <code>text</code>, <code>status</code></li>
                <li>✅ Optional column: <code>subject</code></li>
                <li>✅ Supported formats: CSV, XLSX, XLS</li>
                <li>✅ Status values: 'real', 'fake', 'true', 'false', '0', '1'</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # System status
        st.markdown("### 📊 System Status")
        
        status_items = [
            ("Dataset", st.session_state.dataset is not None),
            ("Preprocessed", st.session_state.preprocessed_data is not None),
            ("Model Trained", st.session_state.trained_model is not None),
            ("Ready to Predict", st.session_state.trained_model is not None)
        ]
        
        for item, status in status_items:
            icon = "✅" if status else "⏳"
            color = "#28a745" if status else "#999"
            st.markdown(f"**{icon} {item}** <span style='color: {color};'>{'Complete' if status else 'Pending'}</span>", unsafe_allow_html=True)
    
    # Navigation buttons
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🎯 Get Started")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("📤 Upload Dataset", use_container_width=True):
            st.session_state.page = 'upload'
            st.rerun()
    
    with col2:
        if st.button("⚙️ Preprocess Data", use_container_width=True, disabled=st.session_state.dataset is None):
            st.session_state.page = 'preprocess'
            st.rerun()
    
    with col3:
        if st.button("🤖 Train Model", use_container_width=True, disabled=st.session_state.preprocessed_data is None):
            st.session_state.page = 'train'
            st.rerun()

# ================================
# SIDEBAR NAVIGATION
# ================================
with st.sidebar:
    st.markdown("## 🧭 Navigation")
    
    if st.button("🏠 Home", use_container_width=True, type="primary" if st.session_state.page == 'home' else "secondary"):
        st.session_state.page = 'home'
        st.rerun()
    
    st.markdown("---")
    
    if st.button("📤 Upload Dataset", use_container_width=True, type="primary" if st.session_state.page == 'upload' else "secondary"):
        st.session_state.page = 'upload'
        st.rerun()
    
    if st.button("⚙️ Preprocess Data", use_container_width=True, type="primary" if st.session_state.page == 'preprocess' else "secondary", disabled=st.session_state.dataset is None):
        st.session_state.page = 'preprocess'
        st.rerun()
    
    if st.button("🤖 Train Model", use_container_width=True, type="primary" if st.session_state.page == 'train' else "secondary", disabled=st.session_state.preprocessed_data is None):
        st.session_state.page = 'train'
        st.rerun()
    
    if st.button("🎯 Make Prediction", use_container_width=True, type="primary" if st.session_state.page == 'predict' else "secondary", disabled=st.session_state.trained_model is None):
        st.session_state.page = 'predict'
        st.rerun()
    
    st.markdown("---")
    
    # System status in sidebar
    st.markdown("### 📊 Status")
    st.markdown(f"**Dataset:** {'✅' if st.session_state.dataset is not None else '⏳'}")
    st.markdown(f"**Preprocessed:** {'✅' if st.session_state.preprocessed_data is not None else '⏳'}")
    st.markdown(f"**Model:** {'✅' if st.session_state.trained_model is not None else '⏳'}")
    
    if st.session_state.dataset is not None:
        st.markdown(f"**Rows:** {len(st.session_state.dataset):,}")
    
    if st.session_state.model_metrics is not None:
        st.markdown(f"**Accuracy:** {st.session_state.model_metrics.get('accuracy', 0):.2%}")

# ================================
# PAGE ROUTING
# ================================
if st.session_state.page == 'home':
    home_page()
elif st.session_state.page == 'upload':
    from pages import upload_page
    upload_page.show()
elif st.session_state.page == 'preprocess':
    from pages import preprocess_page
    preprocess_page.show()
elif st.session_state.page == 'train':
    from pages import train_page
    train_page.show()
elif st.session_state.page == 'predict':
    from pages import predict_page
    predict_page.show()