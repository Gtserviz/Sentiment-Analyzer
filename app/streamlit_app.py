import streamlit as st
import sys
import time
from pathlib import Path

# ===== SETUP =====
def setup_paths():
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    src_path = project_root / "src"
    
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    return project_root

project_root = setup_paths()

# ===== IMPORTS =====
try:
    from models.sentiment_predictor import SentimentPredictor
    from models.emoji_predictor import EmojiPredictor
    sentiment_available = True
    emoji_available = True
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# ===== HELPER FUNCTIONS =====
def get_sentiment_color(sentiment: str) -> str:
    colors = {'positive': '#10b981', 'negative': '#ef4444', 'neutral': '#f59e0b'}
    return colors.get(sentiment.lower(), '#6b7280')

def get_sentiment_icon(sentiment: str) -> str:
    icons = {'positive': '📈', 'negative': '📉', 'neutral': '➖'}
    return icons.get(sentiment.lower(), '❓')

def create_sentiment_chart(probabilities):
    try:
        import plotly.graph_objects as go
        labels = list(probabilities.keys())
        values = list(probabilities.values())
        colors = [get_sentiment_color(label) for label in labels]
        
        fig = go.Figure(data=[go.Bar(x=labels, y=values, marker_color=colors)])
        fig.update_layout(
            title="Sentiment Confidence",
            yaxis=dict(title="Confidence", tickformat='.1%'),
            height=400
        )
        return fig
    except:
        return None

def display_emoji_grid(emojis):
    if not emojis:
        st.info("No emojis available")
        return
    
    cols = st.columns(min(len(emojis), 5))
    for i, emoji_data in enumerate(emojis):
        with cols[i % len(cols)]:
            st.markdown(f"""
                <div style="text-align: center; padding: 1rem; border: 2px solid #8b5cf6; 
                     border-radius: 12px; background: #f8fafc; margin: 0.25rem;">
                    <div style="font-size: 2.5rem;">{emoji_data['emoji']}</div>
                    <div style="font-weight: bold; color: #8b5cf6;">{emoji_data['confidence']:.1%}</div>
                </div>
            """, unsafe_allow_html=True)

# ===== MODEL LOADING =====
@st.cache_resource
def load_models():
    model_path = project_root / "data" / "models" / "sentiment"
    
    # For GitHub deployment, load model from Hugging Face Hub
    try:
        predictor = SentimentPredictor(str(model_path))
        if predictor.load_model():
            emoji_predictor = EmojiPredictor(project_root / "data" / "processed")
            emoji_predictor.load_emoji_data()
            return predictor, emoji_predictor
    except:
        st.error("⚠️ Model not found locally. Please train the model first.")
        return None, None
    
    return None, None

# ===== MAIN APP =====
def main():
    st.set_page_config(page_title="AI Sentiment Analyzer", page_icon="🎭", layout="wide")
    
    # Load models
    sentiment_predictor, emoji_predictor = load_models()
    
    # Sidebar
    st.sidebar.header("🏥 Model Status")
    if sentiment_predictor:
        st.sidebar.success("✅ Model Loaded")
        info = sentiment_predictor.get_model_info()
        st.sidebar.write(f"**Device:** {info.get('device_type', 'CPU')}")
        st.sidebar.write(f"**Parameters:** {info.get('num_parameters', 0):,}")
    else:
        st.sidebar.error("❌ Model Not Loaded")
        st.sidebar.info("Run training script to create model files")
    
    # Header
    st.markdown("""
        <div style="text-align: center; margin: 2rem 0;">
            <h1 style="font-size: 3.5rem; color: #8b5cf6; margin-bottom: 0.5rem;">
                🎭 AI Sentiment Analyzer
            </h1>
            <p style="font-size: 1.2rem; color: #666;">
                Powered by DistilBERT with Smart Emoji Predictions
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    if not sentiment_predictor:
        st.error("❌ Cannot proceed without loaded model")
        st.info("**Setup Instructions:**")
        st.code("1. Run: python src/training/train_distilbert.py")
        st.code("2. Restart the app")
        return
    
    # Input Section
    st.subheader("📝 Analyze Your Text")
    
    user_text = st.text_area(
        "Enter text to analyze:",
        placeholder="Type your message here...",
        height=120,
        max_chars=512
    )
    
    if st.button("🚀 Analyze Sentiment", type="primary", use_container_width=True):
        if not user_text.strip():
            st.warning("⚠️ Please enter some text!")
        else:
            analyze_text(user_text, sentiment_predictor, emoji_predictor)

def analyze_text(text, sentiment_predictor, emoji_predictor):
    with st.spinner("Analyzing..."):
        try:
            start_time = time.time()
            result = sentiment_predictor.predict(text)
            
            # Get emojis if available
            emojis = []
            if emoji_predictor and emoji_predictor.is_loaded():
                emojis = emoji_predictor.predict_emojis(text, result, max_emojis=5)
            
            processing_time = time.time() - start_time
            
            # Display results
            sentiment = result['label']
            confidence = result['confidence']
            color = get_sentiment_color(sentiment)
            
            # Main result card
            st.markdown(f"""
                <div style="background: {color}15; border: 2px solid {color}; 
                     border-radius: 15px; padding: 2rem; text-align: center; margin: 1rem 0;">
                    <div style="font-size: 3rem;">{get_sentiment_icon(sentiment)}</div>
                    <h2 style="color: {color};">{sentiment.upper()}</h2>
                    <h3 style="color: #666;">{confidence:.1%} Confidence</h3>
                </div>
            """, unsafe_allow_html=True)
            
            # Show emojis first
            if emojis:
                st.subheader("😊 Recommended Emojis")
                display_emoji_grid(emojis)
            
            # Tabs for detailed results
            tab1, tab2 = st.tabs(["📈 Probabilities", "⚡ Performance"])
            
            with tab1:
                st.write("**Sentiment Probabilities:**")
                for label, prob in result['probabilities'].items():
                    st.write(f"- **{label.title()}**: {prob:.1%}")
                
                fig = create_sentiment_chart(result['probabilities'])
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                col1, col2 = st.columns(2)
                col1.metric("Processing Time", f"{processing_time*1000:.0f} ms")
                col2.metric("Text Length", f"{len(text)} chars")
                
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")

if __name__ == "__main__":
    main()
