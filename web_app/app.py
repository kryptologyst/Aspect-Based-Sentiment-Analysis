"""
Streamlit web interface for Aspect-Based Sentiment Analysis.

This module provides an interactive web interface for demonstrating
and using the aspect-based sentiment analysis capabilities.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from aspect_sentiment_analyzer import AspectBasedSentimentAnalyzer, AspectSentimentResult
from data_processor import SyntheticDatasetGenerator, DataProcessor


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = []
    if 'dataset' not in st.session_state:
        st.session_state.dataset = []


def load_analyzer():
    """Load the sentiment analyzer."""
    if st.session_state.analyzer is None:
        with st.spinner("Loading sentiment analyzer..."):
            st.session_state.analyzer = AspectBasedSentimentAnalyzer()
    return st.session_state.analyzer


def create_sentiment_chart(results: list[AspectSentimentResult]):
    """Create interactive charts for sentiment analysis results."""
    if not results:
        return None
    
    # Prepare data
    aspects = [r.aspect for r in results]
    sentiments = [r.sentiment for r in results]
    confidences = [r.confidence for r in results]
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Sentiment Distribution', 'Confidence by Aspect'),
        specs=[[{"type": "pie"}, {"type": "bar"}]]
    )
    
    # Pie chart for sentiment distribution
    sentiment_counts = pd.Series(sentiments).value_counts()
    fig.add_trace(
        go.Pie(
            labels=sentiment_counts.index,
            values=sentiment_counts.values,
            name="Sentiment Distribution"
        ),
        row=1, col=1
    )
    
    # Bar chart for confidence scores
    df = pd.DataFrame({
        'Aspect': aspects,
        'Confidence': confidences,
        'Sentiment': sentiments
    })
    
    for sentiment in df['Sentiment'].unique():
        sentiment_data = df[df['Sentiment'] == sentiment]
        fig.add_trace(
            go.Bar(
                x=sentiment_data['Aspect'],
                y=sentiment_data['Confidence'],
                name=sentiment,
                text=sentiment_data['Confidence'].round(3),
                textposition='auto'
            ),
            row=1, col=2
        )
    
    fig.update_layout(
        height=500,
        showlegend=True,
        title_text="Aspect-Based Sentiment Analysis Results"
    )
    
    return fig


def display_results_table(results: list[AspectSentimentResult]):
    """Display results in a formatted table."""
    if not results:
        st.warning("No results to display.")
        return
    
    data = []
    for result in results:
        data.append({
            'Aspect': result.aspect,
            'Sentiment': result.sentiment,
            'Confidence': f"{result.confidence:.3f}",
            'Text Span': result.text_span[:100] + "..." if len(result.text_span) > 100 else result.text_span
        })
    
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Aspect-Based Sentiment Analysis",
        page_icon="📊",
        layout="wide"
    )
    
    initialize_session_state()
    
    # Header
    st.title("📊 Aspect-Based Sentiment Analysis")
    st.markdown("Analyze sentiment for specific aspects in text using state-of-the-art transformer models.")
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Model selection
    model_options = {
        "Twitter RoBERTa": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "BERT Base": "bert-base-uncased",
        "DistilBERT": "distilbert-base-uncased-finetuned-sst-2-english"
    }
    
    selected_model = st.sidebar.selectbox(
        "Select Model",
        list(model_options.keys()),
        index=0
    )
    
    # Aspect selection
    st.sidebar.subheader("Aspect Configuration")
    auto_detect = st.sidebar.checkbox("Auto-detect aspects", value=True)
    
    if not auto_detect:
        aspects_input = st.sidebar.text_input(
            "Specify aspects (comma-separated)",
            value="food,service,ambiance,price"
        )
        custom_aspects = [aspect.strip() for aspect in aspects_input.split(",")]
    else:
        custom_aspects = None
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Text Analysis", "📊 Batch Analysis", "🎲 Demo Dataset", "📈 Model Performance"])
    
    with tab1:
        st.header("Single Text Analysis")
        
        # Text input
        text_input = st.text_area(
            "Enter text to analyze:",
            height=150,
            value="The food at the restaurant was absolutely delicious and well-prepared. However, the service was quite slow and the staff seemed overwhelmed. The ambiance was cozy and romantic, perfect for a date night. The prices were reasonable for the quality of food, though the portions could have been larger."
        )
        
        col1, col2 = st.columns([1, 4])
        
        with col1:
            analyze_button = st.button("Analyze", type="primary")
        
        if analyze_button and text_input.strip():
            analyzer = load_analyzer()
            
            with st.spinner("Analyzing text..."):
                results = analyzer.analyze_text(text_input, custom_aspects)
                st.session_state.analysis_results = results
            
            # Display results
            st.success(f"Analysis complete! Found {len(results)} aspects.")
            
            # Results table
            st.subheader("Analysis Results")
            display_results_table(results)
            
            # Visualizations
            st.subheader("Visualizations")
            chart = create_sentiment_chart(results)
            if chart:
                st.plotly_chart(chart, use_container_width=True)
            
            # Detailed report
            st.subheader("Detailed Report")
            report = analyzer.generate_report(results)
            st.text(report)
    
    with tab2:
        st.header("Batch Text Analysis")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload a JSON file with texts",
            type=['json'],
            help="JSON file should contain a list of text strings"
        )
        
        if uploaded_file:
            try:
                data = json.load(uploaded_file)
                if isinstance(data, list):
                    texts = data
                else:
                    st.error("JSON file should contain a list of text strings.")
                    texts = []
            except Exception as e:
                st.error(f"Error reading file: {e}")
                texts = []
        else:
            # Sample texts
            sample_texts = [
                "The food was amazing but the service was terrible.",
                "Great ambiance and reasonable prices, but the food was mediocre.",
                "Excellent service and delicious food, though it's quite expensive.",
                "The delivery was fast but the product quality was poor.",
                "Good value for money, but the customer service needs improvement."
            ]
            texts = sample_texts
            st.info("Using sample texts. Upload a JSON file for custom analysis.")
        
        if st.button("Analyze Batch", type="primary") and texts:
            analyzer = load_analyzer()
            
            progress_bar = st.progress(0)
            batch_results = []
            
            for i, text in enumerate(texts):
                progress_bar.progress((i + 1) / len(texts))
                results = analyzer.analyze_text(text, custom_aspects)
                batch_results.append({
                    'text': text,
                    'results': results
                })
            
            st.success(f"Batch analysis complete! Processed {len(texts)} texts.")
            
            # Display batch results
            for i, batch_result in enumerate(batch_results):
                with st.expander(f"Text {i+1}: {batch_result['text'][:50]}..."):
                    display_results_table(batch_result['results'])
    
    with tab3:
        st.header("Demo Dataset")
        
        col1, col2 = st.columns(2)
        
        with col1:
            dataset_size = st.slider("Dataset Size", 10, 100, 50)
            domains = st.multiselect(
                "Domains",
                ["restaurant", "ecommerce"],
                default=["restaurant"]
            )
        
        with col2:
            if st.button("Generate Dataset", type="primary"):
                generator = SyntheticDatasetGenerator()
                dataset = generator.generate_dataset(size=dataset_size, domains=domains)
                st.session_state.dataset = dataset
                st.success(f"Generated {len(dataset)} samples!")
        
        if st.session_state.dataset:
            st.subheader("Dataset Preview")
            
            # Show sample
            sample_idx = st.selectbox("Select sample to view", range(len(st.session_state.dataset)))
            sample = st.session_state.dataset[sample_idx]
            
            st.write("**Text:**", sample.text)
            st.write("**Aspects:**", ", ".join(sample.aspects))
            st.write("**Aspect Sentiments:**", sample.aspect_sentiments)
            st.write("**Overall Sentiment:**", sample.overall_sentiment)
            st.write("**Domain:**", sample.domain)
            
            # Analyze sample
            if st.button("Analyze This Sample"):
                analyzer = load_analyzer()
                results = analyzer.analyze_text(sample.text)
                
                st.subheader("Analysis Results")
                display_results_table(results)
                
                chart = create_sentiment_chart(results)
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
    
    with tab4:
        st.header("Model Performance")
        
        st.info("This section would show model performance metrics, confusion matrices, and evaluation results.")
        
        # Placeholder for performance metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Accuracy", "0.87", "0.02")
        
        with col2:
            st.metric("F1 Score", "0.85", "0.01")
        
        with col3:
            st.metric("Precision", "0.83", "-0.01")
        
        # Performance chart placeholder
        st.subheader("Performance Over Time")
        
        # Create sample performance data
        performance_data = pd.DataFrame({
            'Epoch': range(1, 11),
            'Accuracy': [0.75, 0.78, 0.81, 0.83, 0.85, 0.86, 0.87, 0.87, 0.87, 0.87],
            'F1 Score': [0.72, 0.75, 0.78, 0.81, 0.83, 0.84, 0.85, 0.85, 0.85, 0.85]
        })
        
        fig = px.line(performance_data, x='Epoch', y=['Accuracy', 'F1 Score'], 
                     title='Model Performance During Training')
        st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit and Transformers")


if __name__ == "__main__":
    main()
