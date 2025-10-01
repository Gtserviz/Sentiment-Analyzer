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

## 🏁 Quick Start

### 1️⃣ **Clone & Install**
```bash
git clone https://github.com/Abhay-Rudatala/Sentiment-Analyzer.git
cd Sentiment-Analyzer
pip install -r requirements.txt
```

### 2️⃣ **Train Models**
```bash
python src/training/train_distilbert.py
```

### 4️⃣ **Run App**
```bash
streamlit run app/streamlit_app.py
```

🌐 Open your browser to `http://localhost:8501`

## 📁 Project Structure

```
├── 🎨 app/streamlit_app.py          # Streamlit web application
├── 📁 src/models/                   # Model classes
├── 🤖 src/training/                 # Training scripts
├── 📊 data/processed/               # Processed data files
├── 📦 requirements.txt              # Dependencies
└── 📖 README.md                     # This file
```

## 📈 Model Details

- **Architecture**: DistilBERT (66M parameters)
- **Classes**: Negative, Neutral, Positive
- **Performance**: ~85% accuracy on test data

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## 🌟 Show Your Support

If this project helped you, please ⭐ star this repository!

---

*Ready to analyze your resume? Let's get started! 🚀*
