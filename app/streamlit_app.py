import streamlit as st
import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
import json

# ===== CONFIGURATION =====
# Replace with your actual Hugging Face model repository
HUGGINGFACE_MODEL_REPO = "Arskye/Sentiment-Model"

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
    except Exception:
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

# ===== EMOJI PREDICTOR =====
class SimpleEmojiPredictor:
    def __init__(self):
        self.sentiment_emojis = {
            'positive': [
                {'emoji': '😊', 'confidence': 0.95},
                {'emoji': '😃', 'confidence': 0.92},
                {'emoji': '🥰', 'confidence': 0.90},
                {'emoji': '😍', 'confidence': 0.88},
                {'emoji': '🤗', 'confidence': 0.85}
            ],
            'negative': [
                {'emoji': '😢', 'confidence': 0.95},
                {'emoji': '😞', 'confidence': 0.92},
                {'emoji': '😔', 'confidence': 0.90},
                {'emoji': '💔', 'confidence': 0.93},
                {'emoji': '😤', 'confidence': 0.88}
            ],
            'neutral': [
                {'emoji': '😐', 'confidence': 0.95},
                {'emoji': '🤔', 'confidence': 0.93},
                {'emoji': '😑', 'confidence': 0.90},
                {'emoji': '🙄', 'confidence': 0.85},
                {'emoji': '😶', 'confidence': 0.88}
            ]
        }
    
    def predict_emojis(self, text, sentiment_result, max_emojis=5):
        sentiment = sentiment_result['label']
        confidence = sentiment_result['confidence']
        
        emojis = self.sentiment_emojis.get(sentiment, [])
        # Adjust confidence scores
        adjusted_emojis = []
        for emoji_data in emojis[:max_emojis]:
            adjusted_emojis.append({
                'emoji': emoji_data['emoji'],
                'confidence': emoji_data['confidence'] * confidence
            })
        return adjusted_emojis
    
    def is_loaded(self):
        return True

# ===== MODEL LOADING =====
@st.cache_resource
def load_models():
    """Load models from Hugging Face Hub"""
    try:
        with st.spinner("Loading model from Hugging Face Hub..."):
            tokenizer = AutoTokenizer.from_pretrained(HUGGINGFACE_MODEL_REPO)
            model = AutoModelForSequenceClassification.from_pretrained(HUGGINGFACE_MODEL_REPO)
            
            # Move to appropriate device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            
            # Initialize emoji predictor
            emoji_predictor = SimpleEmojiPredictor()
            
            return tokenizer, model, device, emoji_predictor
    except Exception as e:
        st.error(f"❌ Failed to load model from Hugging Face Hub: {str(e)}")
        st.info("Please make sure you've uploaded your model to Hugging Face Hub and updated the HUGGINGFACE_MODEL_REPO variable.")
        return None, None, None, None

def predict_sentiment(tokenizer, model, device, text):
    """Predict sentiment for given text"""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()
    
    # Get label mapping from model config
    labels = model.config.id2label or {0: "negative", 1: "neutral", 2: "positive"}
    
    # Create result dictionary
    result = {
        'label': labels[predicted_class],
        'confidence': probabilities[0][predicted_class].item(),
        'probabilities': {
            labels[i]: probabilities[0][i].item() 
            for i in range(len(labels))
        }
    }
    
    return result

# ===== MAIN APP =====
def main():
    st.set_page_config(
        page_title="AI Sentiment Analyzer", 
        page_icon="🎭", 
        layout="wide"
    )
    
    # Load models
    tokenizer, model, device, emoji_predictor = load_models()
    
    # Sidebar
    st.sidebar.header("🏥 Model Status")
    if model is not None:
        st.sidebar.success("✅ Model Loaded from Hugging Face Hub")
        st.sidebar.write(f"**Device:** {device}")
        st.sidebar.write(f"**Model:** {HUGGINGFACE_MODEL_REPO}")
        
        # Model info
        try:
            num_params = sum(p.numel() for p in model.parameters())
            st.sidebar.write(f"**Parameters:** {num_params:,}")
        except:
            pass
    else:
        st.sidebar.error("❌ Model Not Loaded")
        st.sidebar.info("Check Hugging Face Hub connection")
    
    # Header
    st.markdown("""
        <div style="text-align: center; margin: 2rem 0;">
            <h1 style="font-size: 3.5rem; color: #8b5cf6; margin-bottom: 0.5rem;">
                🎭 AI Sentiment Analyzer
            </h1>
            <p style="font-size: 1.2rem; color: #666;">
                Powered by Fine-tuned DistilBERT from Hugging Face Hub
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    if model is None:
        st.error("❌ Cannot proceed without loaded model")
        st.info(f"**Please ensure your model is uploaded to:** {HUGGINGFACE_MODEL_REPO}")
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
            analyze_text(user_text, tokenizer, model, device, emoji_predictor)

def analyze_text(text, tokenizer, model, device, emoji_predictor):
    """Analyze text and display results"""
    with st.spinner("Analyzing..."):
        try:
            start_time = time.time()
            result = predict_sentiment(tokenizer, model, device, text)
            
            # Get emojis
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
