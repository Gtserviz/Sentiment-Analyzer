<div align="center">
  <h1 align="center">SENTIMENT-ANALYZER</h1>
  <h3><em>Real-Time Sentiment Classification with Emoji Recommendations</em></h3>
  
  ![Last Commit](https://img.shields.io/github/last-commit/Abhay-Rudatala/Sentiment-Analyzer?label=last%20commit&color=blue)
  ![Python](https://img.shields.io/badge/python-100.0%25-blue)
  ![Languages](https://img.shields.io/badge/languages-1-green)

  <br>
  
  <h4><strong>Built with the tools and technologies:</strong></h4>

  ![Markdown](https://img.shields.io/badge/markdown-%23000000.svg?style=for-the-badge&logo=markdown&logoColor=white)
  ![Streamlit](https://img.shields.io/badge/streamlit-%23FE4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)
  ![transformers](https://img.shields.io/badge/transformers-%23FF6F61.svg?style=for-the-badge&logo=transformers&logoColor=white)
  ![PyTorch](https://img.shields.io/badge/pytorch-%23EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white)

  ![HuggingFace](https://img.shields.io/badge/huggingface-%23FF6F00.svg?style=for-the-badge&logo=huggingface&logoColor=white)
  ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
  ![Plotly](https://img.shields.io/badge/plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)
  ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
</div>

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
