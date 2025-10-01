# 🎭 AI Sentiment Analyzer

A modern sentiment analysis application powered by DistilBERT with smart emoji predictions.

---

## 🚀 Live Demo

Try the live application: [[Click Here](https://arskye-sentiment-analyzer.streamlit.app/)]

## Features

- **3-Class Sentiment Analysis**: Positive, Negative, Neutral
- **Smart Emoji Predictions**: Context-aware emoji suggestions  
- **Real-time Analysis**: Instant results with confidence scores
- **Modern UI**: Clean, responsive Streamlit interface

## Quick Start

### 1. Installation

git clone https://github.com/yourusername/sentiment-analysis.git
cd sentiment-analysis
pip install -r requirements.txt

### 2. Train Model

python src/training/train_distilbert.py

### 3. Run App

streamlit run app/streamlit_app.py


## Project Structure

├── app/streamlit_app.py # Streamlit web application
├── src/models/ # Model classes
├── src/training/ # Training scripts
├── data/processed/ # Processed data files
└── requirements.txt # Dependencies

## Model Details

- **Architecture**: DistilBERT (66M parameters)
- **Classes**: Negative, Neutral, Positive
- **Performance**: ~85% accuracy on test data

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License

MIT License - see LICENSE file for details.
