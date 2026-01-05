import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report)
import plotly.graph_objects as go
import plotly.express as px
import time


def show():
    """Train Model Page - SVM and Naive Bayes"""
    
    st.markdown('<h1 class="train-title">🤖 Train Model</h1>', unsafe_allow_html=True)
    st.markdown('<p class="train-subtitle">Train machine learning models for fake news detection</p>', unsafe_allow_html=True)
    
    if st.session_state.preprocessed_data is None:
        st.warning("⚠️ No preprocessed data available. Please preprocess your dataset first.")
        if st.button("⚙️ Go to Preprocess Page"):
            st.session_state.page = 'preprocess'
            st.rerun()
        return
    
    df = st.session_state.preprocessed_data.copy()
    
    # Show dataset info
    st.markdown("### 📊 Dataset Information")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-label">Total Samples</div>
            <div class="stats-value">{len(df):,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if 'status' in df.columns:
            class_dist = df['status'].value_counts()
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-label">Class Distribution</div>
                <div class="stats-value">{len(class_dist)}</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        avg_text_len = df['text'].str.len().mean()
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-label">Avg Text Length</div>
            <div class="stats-value">{avg_text_len:.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        if st.session_state.trained_model is not None:
            st.markdown(f"""
            <div class="stats-card" style="border-left-color: #28a745;">
                <div class="stats-label">Model Status</div>
                <div class="stats-value" style="color: #28a745; font-size: 1.2rem;">✅ Trained</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="stats-card" style="border-left-color: #ffc107;">
                <div class="stats-label">Model Status</div>
                <div class="stats-value" style="color: #ffc107; font-size: 1.2rem;">⏳ Not Trained</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Training Configuration
    st.markdown("### ⚙️ Training Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Model Selection")
        
        # Only 2 models
        model_choice = st.selectbox(
            "Choose Algorithm",
            [
                "Support Vector Machine (SVM)",
                "Naive Bayes"
            ],
            help="Select the machine learning algorithm to train"
        )
        
        # Show model description
        if model_choice == "Support Vector Machine (SVM)":
            st.success("🎯 **Best for text classification!** Finds optimal decision boundary. Excellent accuracy but slower training.")
            with st.expander("ℹ️ View Details"):
                st.markdown("**✅ Pros:** High accuracy, works well with high-dimensional data, robust")
                st.markdown("**❌ Cons:** Slower training, requires more memory")
        else:
            st.info("⚡ **Fastest training!** Probabilistic classifier based on Bayes theorem. Great for large datasets.")
            with st.expander("ℹ️ View Details"):
                st.markdown("**✅ Pros:** Very fast, works well with text data, requires less memory")
                st.markdown("**❌ Cons:** Assumes feature independence, may be less accurate")
        
        # Vectorizer selection
        vectorizer_choice = st.selectbox(
            "Feature Extraction Method",
            ["TF-IDF", "Count Vectorizer (Bag of Words)"],
            help="TF-IDF is recommended for text classification"
        )
        
        # Text feature selection
        text_feature = st.radio(
            "Text to Use",
            ["Text + Title Combined", "Text Only", "Title Only"],
            help="Which text field to use for training"
        )
        
    with col2:
        st.markdown("#### 🔧 Basic Settings")
        
        # Test size
        test_size = st.slider(
            "Test Set Size (%)",
            min_value=10,
            max_value=40,
            value=20,
            step=5,
            help="Percentage of data for testing (20% is recommended)"
        )
        
        # Max features
        max_features = st.number_input(
            "Max Features",
            min_value=1000,
            max_value=10000,
            value=5000,
            step=500,
            help="Maximum words to use (5000 is recommended)"
        )
        
        # Advanced settings toggle
        show_advanced = st.checkbox("⚙️ Show Advanced Settings", value=False)
        
        if show_advanced:
            st.markdown("**Advanced Hyperparameters:**")
            
            # Model-specific parameters
            if model_choice == "Support Vector Machine (SVM)":
                C = st.slider(
                    "C (Regularization)",
                    min_value=0.1,
                    max_value=10.0,
                    value=1.0,
                    step=0.1,
                    help="Higher = stricter classification"
                )
                
                kernel = st.selectbox(
                    "Kernel",
                    ["linear", "rbf"],
                    help="linear is best for text"
                )
                
                hyperparams = {'C': C, 'kernel': kernel}
                
            else:  # Naive Bayes
                alpha = st.slider(
                    "Alpha (Smoothing)",
                    min_value=0.1,
                    max_value=2.0,
                    value=1.0,
                    step=0.1,
                    help="Laplace smoothing parameter"
                )
                
                hyperparams = {'alpha': alpha}
        else:
            # Use defaults
            hyperparams = {}
    
    # Show class distribution
    if 'status' in df.columns:
        st.markdown("### 📊 Class Distribution")
        
        class_counts = df['status'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(
                values=class_counts.values,
                names=class_counts.index,
                title='Class Distribution',
                color_discrete_sequence=['#e45756', '#28a745'],
                hole=0.4
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("<br><br>", unsafe_allow_html=True)
            for label, count in class_counts.items():
                percentage = (count / len(df)) * 100
                st.markdown(f"""
                <div class="info-box">
                    <strong>Class {label}:</strong> {count:,} samples ({percentage:.1f}%)
                </div>
                """, unsafe_allow_html=True)
    
    # Train button
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🚀 Train Model")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if st.button("🎯 Start Training", use_container_width=True, type="primary"):
            train_model(
                df, 
                model_choice,
                vectorizer_choice,
                text_feature,
                test_size / 100,
                max_features,
                hyperparams
            )
    
    # Show previous model results if available
    if st.session_state.model_metrics is not None:
        st.markdown("---")
        display_model_results()


def train_model(df, model_choice, vectorizer_choice, text_feature, test_size, max_features, hyperparams):
    """Train selected machine learning model"""
    
    with st.spinner("🔄 Preparing data..."):
        # Prepare text data based on selection
        if text_feature == "Text Only":
            X = df['text']
        elif text_feature == "Title Only":
            X = df['title']
        else:  # Combined
            X = df['title'] + ' ' + df['text']
        
        y = df['status']
        
        # Split data with fixed random state for consistency
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        st.success(f"✅ Data split: {len(X_train):,} training, {len(X_test):,} testing")
    
    with st.spinner("🔄 Vectorizing text..."):
        # Create vectorizer
        if vectorizer_choice == "TF-IDF":
            vectorizer = TfidfVectorizer(max_features=max_features)
        else:
            vectorizer = CountVectorizer(max_features=max_features)
        
        # Fit and transform
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        st.success(f"✅ Vectorization complete: {X_train_vec.shape[1]:,} features")
    
    with st.spinner(f"🔄 Training {model_choice}..."):
        # Create model based on selection
        if model_choice == "Support Vector Machine (SVM)":
            model = SVC(
                kernel=hyperparams.get('kernel', 'linear'),
                C=hyperparams.get('C', 1.0),
                probability=True,
                random_state=42
            )
        else:  # Naive Bayes
            model = MultinomialNB(
                alpha=hyperparams.get('alpha', 1.0)
            )
        
        # Train model
        start_time = time.time()
        model.fit(X_train_vec, y_train)
        training_time = time.time() - start_time
        
        st.success(f"✅ Model trained in {training_time:.2f} seconds")
    
    with st.spinner("🔄 Evaluating model..."):
        # Make predictions
        y_pred = model.predict(X_test_vec)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        # Store in session state
        st.session_state.trained_model = model
        st.session_state.vectorizer = vectorizer
        st.session_state.text_feature = text_feature
        st.session_state.model_name = model_choice
        st.session_state.model_metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'classification_report': report,
            'model_name': model_choice,
            'vectorizer_name': vectorizer_choice,
            'training_time': training_time,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'n_features': X_train_vec.shape[1],
            'hyperparams': hyperparams
        }
        
        st.success("✅ Model evaluation complete!")
        st.balloons()
        st.rerun()


def display_model_results():
    """Display comprehensive model results"""
    
    st.markdown("### 📊 Model Performance")
    
    metrics = st.session_state.model_metrics
    
    # Performance indicator
    accuracy = metrics['accuracy']
    if accuracy >= 0.90:
        performance_color = "#28a745"
        performance_label = "Excellent"
        performance_emoji = "🌟"
    elif accuracy >= 0.80:
        performance_color = "#17a2b8"
        performance_label = "Good"
        performance_emoji = "👍"
    elif accuracy >= 0.70:
        performance_color = "#ffc107"
        performance_label = "Fair"
        performance_emoji = "⚠️"
    else:
        performance_color = "#dc3545"
        performance_label = "Needs Improvement"
        performance_emoji = "❌"
    
    st.markdown(f"""
    <div class="performance-banner" style="background: linear-gradient(135deg, {performance_color}15 0%, {performance_color}25 100%); 
                                          border-left: 4px solid {performance_color};">
        <span style="font-size: 1.5rem;">{performance_emoji}</span>
        <strong style="color: {performance_color};">Overall Performance: {performance_label}</strong>
        <span style="float: right; color: {performance_color}; font-weight: bold;">{accuracy:.1%}</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-label">Accuracy</div>
            <div class="stats-value">{metrics['accuracy']:.2%}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-label">Precision</div>
            <div class="stats-value">{metrics['precision']:.2%}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-label">Recall</div>
            <div class="stats-value">{metrics['recall']:.2%}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-label">F1-Score</div>
            <div class="stats-value">{metrics['f1']:.2%}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Confusion Matrix
    st.markdown("### 🎯 Confusion Matrix")
    
    cm = metrics['confusion_matrix']
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted Fake', 'Predicted Real'],
        y=['Actual Fake', 'Actual Real'],
        colorscale='RdYlGn',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 20},
        showscale=True
    ))
    
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Classification Report
    st.markdown("### 📋 Classification Report")
    
    report_df = pd.DataFrame(metrics['classification_report']).transpose()
    st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)
    
    # Model Comparison (if multiple models trained)
    if 'model_history' not in st.session_state:
        st.session_state.model_history = []
    
    # Add current model to history
    current_model_info = {
        'model': metrics['model_name'],
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1'],
        'time': metrics['training_time']
    }
    
    # Check if this model already exists in history
    model_exists = False
    for i, hist_model in enumerate(st.session_state.model_history):
        if hist_model['model'] == current_model_info['model']:
            st.session_state.model_history[i] = current_model_info
            model_exists = True
            break
    
    if not model_exists:
        st.session_state.model_history.append(current_model_info)
    
    # Display comparison if both models trained
    if len(st.session_state.model_history) == 2:
        st.markdown("### 📊 Model Comparison")
        
        comparison_df = pd.DataFrame(st.session_state.model_history)
        
        # Metrics comparison chart
        fig = go.Figure()
        
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
        for metric in metrics_to_plot:
            fig.add_trace(go.Bar(
                name=metric.capitalize(),
                x=comparison_df['model'],
                y=comparison_df[metric],
                text=comparison_df[metric].apply(lambda x: f'{x:.1%}'),
                textposition='auto',
            ))
        
        fig.update_layout(
            title='SVM vs Naive Bayes - Performance Comparison',
            xaxis_title='Model',
            yaxis_title='Score',
            barmode='group',
            height=400,
            yaxis=dict(tickformat='.0%')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display comparison table
        st.dataframe(
            comparison_df.style.format({
                'accuracy': '{:.2%}',
                'precision': '{:.2%}',
                'recall': '{:.2%}',
                'f1': '{:.2%}',
                'time': '{:.2f}s'
            }).highlight_max(subset=['accuracy', 'precision', 'recall', 'f1'], color='lightgreen'),
            use_container_width=True
        )
        
        # Winner announcement
        best_model = comparison_df.loc[comparison_df['accuracy'].idxmax()]
        st.success(f"🏆 **Winner:** {best_model['model']} with {best_model['accuracy']:.2%} accuracy!")
    
    # Model info
    st.markdown("### ℹ️ Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        hyperparams_str = ', '.join([f"{k}: {v}" for k, v in metrics.get('hyperparams', {}).items()])
        if not hyperparams_str:
            hyperparams_str = "Default"
            
        st.markdown(f"""
        <div class="info-box">
            <strong>Algorithm:</strong> {metrics.get('model_name', 'N/A')}<br>
            <strong>Hyperparameters:</strong> {hyperparams_str}<br>
            <strong>Vectorizer:</strong> {metrics.get('vectorizer_name', 'N/A')}<br>
            <strong>Training Time:</strong> {metrics.get('training_time', 0):.2f} seconds
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="info-box">
            <strong>Training Samples:</strong> {metrics.get('train_size', 0):,}<br>
            <strong>Testing Samples:</strong> {metrics.get('test_size', 0):,}<br>
            <strong>Features Used:</strong> {metrics.get('n_features', 0):,}<br>
            <strong>Test Set Size:</strong> {metrics.get('test_size', 0) / (metrics.get('train_size', 1) + metrics.get('test_size', 1)) * 100:.0f}%
        </div>
        """, unsafe_allow_html=True)
    
    st.success("✅ Model is ready for predictions!")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("👉 Go to **Make Prediction** page to test your model")
    with col2:
        if len(st.session_state.model_history) < 2:
            if st.button("🔄 Train the Other Model", use_container_width=True):
                st.rerun()
    with col3:
        if st.button("🗑️ Clear Model History", use_container_width=True):
            st.session_state.model_history = []
            st.rerun()


# Custom CSS
st.markdown("""
<style>
    .train-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .train-subtitle {
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
    
    .performance-banner {
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0 2rem 0;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)