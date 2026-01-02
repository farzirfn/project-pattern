import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import plotly.graph_objects as go
import time

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


def clean_text(text: str, options: dict, stop_words_set, stemmer_obj) -> str:
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
        tokens = [t for t in tokens if t and t.lower() not in stop_words_set]
    
    # Stemming
    if options.get('stemming', True):
        try:
            tokens = [stemmer_obj.stem(token) for token in tokens]
        except Exception:
            pass
    
    # Rejoin
    result = " ".join(tokens)
    
    # Return original if cleaning resulted in empty string
    return result.strip() if result.strip() else original_text[:100]


def show():
    """Make Prediction Page"""
    
    st.markdown('<h1 class="predict-title">🎯 Make Prediction</h1>', unsafe_allow_html=True)
    st.markdown('<p class="predict-subtitle">Use your trained model to detect fake news</p>', unsafe_allow_html=True)
    
    if st.session_state.trained_model is None:
        st.warning("⚠️ No trained model available. Please train a model first.")
        if st.button("🤖 Go to Train Model Page"):
            st.session_state.page = 'train'
            st.rerun()
        return
    
    # Show model info
    st.markdown("### 🤖 Current Model")
    
    metrics = st.session_state.model_metrics
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stats-card" style="border-left-color: #28a745;">
            <div class="stats-label">Model</div>
            <div class="stats-value" style="font-size: 1.2rem;">{metrics.get('model_name', 'N/A')}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-label">Accuracy</div>
            <div class="stats-value">{metrics['accuracy']:.2%}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-label">F1-Score</div>
            <div class="stats-value">{metrics['f1']:.2%}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-label">Features</div>
            <div class="stats-value">{metrics.get('n_features', 0):,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Single prediction only
    single_prediction()


def single_prediction():
    """Single text prediction interface"""
    
    st.markdown("### 📝 Enter News Article to Analyze")
    st.markdown("Enter the title and text of a news article to check if it's real or fake.")
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Initialize session state for examples if not exists
    if 'example_title' not in st.session_state:
        st.session_state.example_title = ""
    if 'example_text' not in st.session_state:
        st.session_state.example_text = ""
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        title = st.text_input(
            "Title",
            value=st.session_state.example_title,  # Bind to session state
            placeholder="Enter the news title here...",
            help="Enter the headline or title of the news article"
        )
        
        text = st.text_area(
            "Text",
            value=st.session_state.example_text,  # Bind to session state
            placeholder="Enter the news article text here...",
            height=200,
            help="Enter the full text or body of the news article"
        )
    
    with col2:
        st.markdown("#### 📋 Quick Examples")
        
        if st.button("📰 Example 1: Real News", use_container_width=True):
            st.session_state.example_title = "Climate Change Report Released"
            st.session_state.example_text = "Scientists have released a comprehensive report on climate change impacts. The study shows significant temperature increases over the past decade."
            st.rerun()
        
        if st.button("❌ Example 2: Fake News", use_container_width=True):
            st.session_state.example_title = "Miracle Cure Discovered!"
            st.session_state.example_text = "Amazing new treatment cures all diseases instantly! Doctors hate this one simple trick! Click here to learn more about this incredible discovery!"
            st.rerun()
        
        if st.button("🔄 Clear", use_container_width=True):
            st.session_state.example_title = ""
            st.session_state.example_text = ""
            st.rerun()
    
    # Predict button
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if st.button("🎯 Predict", use_container_width=True, type="primary"):
            if not title or not text:
                st.error("❌ Please enter both title and text!")
                return
            
            # Make prediction
            result = predict_single(title, text)
            
            if result:
                # Show result with animation
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### 📊 Prediction Result")
                
                prediction = result['prediction']
                confidence = result['confidence']
                
                # Result card with color coding
                if prediction == 1:  # Real news
                    color = "#28a745"
                    icon = "✅"
                    label = "REAL NEWS"
                    bg_color = "#28a74515"
                else:  # Fake news
                    color = "#e45756"
                    icon = "❌"
                    label = "FAKE NEWS"
                    bg_color = "#e4575615"
                
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, {bg_color}, {bg_color});
                    padding: 2rem;
                    border-radius: 16px;
                    border-left: 6px solid {color};
                    text-align: center;
                    margin: 2rem 0;
                ">
                    <div style="font-size: 4rem; margin-bottom: 1rem;">{icon}</div>
                    <div style="font-size: 2rem; font-weight: 700; color: {color}; margin-bottom: 0.5rem;">
                        {label}
                    </div>
                    <div style="font-size: 1.5rem; color: #666;">
                        Confidence: {confidence:.1%}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence meter
                st.markdown("#### 📈 Confidence Score")
                
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=confidence * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Prediction Confidence", 'font': {'size': 20}},
                    number={'suffix': "%"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': color},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 75], 'color': "gray"},
                            {'range': [75, 100], 'color': "darkgray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # Interpretation
                st.markdown("#### 💡 Interpretation")
                
                if confidence >= 0.8:
                    confidence_level = "Very High"
                    interpretation = "The model is very confident about this prediction."
                elif confidence >= 0.6:
                    confidence_level = "High"
                    interpretation = "The model is fairly confident about this prediction."
                elif confidence >= 0.5:
                    confidence_level = "Moderate"
                    interpretation = "The model has moderate confidence. Consider additional verification."
                else:
                    confidence_level = "Low"
                    interpretation = "The model has low confidence. This prediction should be verified."
                
                st.markdown(f"""
                <div class="info-box">
                    <strong>Confidence Level:</strong> {confidence_level}<br>
                    <strong>Interpretation:</strong> {interpretation}
                </div>
                """, unsafe_allow_html=True)
                
                # Show processed text
                with st.expander("🔍 View Processed Text"):
                    st.markdown("**Original Title:**")
                    st.text(title)
                    st.markdown("**Original Text:**")
                    st.text(text[:500] + "..." if len(text) > 500 else text)
                    st.markdown("**Processed Input:**")
                    st.text(result['processed_text'][:500] + "..." if len(result['processed_text']) > 500 else result['processed_text'])


def predict_single(title: str, text: str):
    """Make prediction for single text"""
    try:
        # Get preprocessing options
        cleaning_options = st.session_state.get('cleaning_options', {
            'lowercase': True,
            'remove_punctuation': True,
            'remove_stopwords': True,
            'stemming': True,
            'remove_numbers': False,
            'remove_urls': True
        })
        
        # Preprocess text
        title_clean = clean_text(title, cleaning_options, stop_words, stemmer)
        text_clean = clean_text(text, cleaning_options, stop_words, stemmer)
        
        # Combine based on training configuration
        text_feature = st.session_state.get('text_feature', 'Text + Title Combined')
        
        if text_feature == "Text Only":
            input_text = text_clean
        elif text_feature == "Title Only":
            input_text = title_clean
        else:
            input_text = title_clean + ' ' + text_clean
        
        # Vectorize
        vectorizer = st.session_state.vectorizer
        input_vec = vectorizer.transform([input_text])
        
        # Predict
        model = st.session_state.trained_model
        prediction = model.predict(input_vec)[0]
        
        # Get confidence (probability)
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(input_vec)[0]
            confidence = proba[prediction]
        elif hasattr(model, 'decision_function'):
            decision = model.decision_function(input_vec)[0]
            confidence = 1 / (1 + np.exp(-abs(decision)))  # Sigmoid approximation
        else:
            confidence = 0.75  # Default for models without probability
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'processed_text': input_text
        }
    
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        return None


# Custom CSS
st.markdown("""
<style>
    .predict-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .predict-subtitle {
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
    
    .info-box {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)