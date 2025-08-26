<<<<<<< HEAD
# WhatsApp Chat Analyser

## Overview

This project performs sentiment analysis on WhatsApp chat data using the DistilBERT model and provides insightful data visualizations, including word clouds, weekly activity heatmaps, and identification of the most active users.

## Features

1. Sentiment Analysis:
   - Utilizes the DistilBERT model (distillBERT base uncased finetuned sst english) for sentiment analysis.
   - Analyzes the sentiment of messages to determine if they are positive or negative by finding out the sentiment scores.

2. Data Visualization:
   - Generates various visualizations to provide a comprehensive overview of the chat data.
   - Includes word clouds, weekly activity heatmaps, identification of the most active users, weekly, monthly and daily activity graphs.
   - Inlcudes most commonly used emojis
   - Sentiment Scores are plotted against each message

## Requirements

- Python 3.x
- Required Python packages are listed in the `requirements.txt` file.

## Setup

1. Clone the repository:

    bash
    git clone https://github.com/your-username/whatsapp-chat-analysis.git

2. Install the required packages:

    bash
    pip install -r requirements.txt

3. Run the main script:

    bash
    python main.py

## Usage

1. Sentiment Analysis:
   - The `sentiment_analysis.py` script performs sentiment analysis on WhatsApp chat data.
   - Adjust the parameters, such as positive and negative thresholds, as needed by ###the user.

2. Data Visualization:
   - The `data_visualization.py` script generates various plots for better understanding of the chat data.
   - Customize the script to include or exclude specific visualizations based on your preferences.
=======
# 📊 WhatsApp Chat Analyzer

This project is a **Streamlit web app** that analyzes WhatsApp chat exports and provides insightful visualizations, statistics, and sentiment analysis of conversations.  

---

## 🚀 Features
- **Basic Statistics**
  - Total messages  
  - Total words  
  - Media shared  
  - Links shared  

- **Timelines & Activity Maps**
  - Monthly timeline of chat activity  
  - Daily activity trends  
  - Weekly heatmap of activity  

- **User Insights**
  - Most active users  
  - WordCloud of frequently used words  
  - Most common words  

- **Emoji Analysis**
  - Emoji usage statistics  
  - Emoji distribution pie chart  

- **Sentiment Analysis**
  - Sentiment (positive/negative) of each message  
  - Sentiment score tracking over time  

---

## 🛠️ Tech Stack
- **Frontend & UI**: [Streamlit](https://streamlit.io/)  
- **Data Processing**: `pandas`, `re`  
- **Visualization**: `matplotlib`, `seaborn`, `wordcloud`  
- **NLP & Sentiment Analysis**: Hugging Face `transformers`, `torch`  
- **Utilities**: `urlextract`, `emoji`

---

## 📂 Project Structure
```

.
├── app.py                 # Main Streamlit application
├── helper.py              # Helper functions for stats, plots, sentiment trends
├── preprocessor.py        # Chat preprocessing (dates, users, messages, etc.)
├── sentiment\_analysis.py  # Sentiment analysis using Hugging Face models
├── requirements.txt       # Python dependencies
└── stop\_hinglish.txt      # Stopwords for text cleaning (hinglish)

````

---

## ⚙️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/whatsapp-chat-analyzer.git
   cd whatsapp-chat-analyzer
   ```

2. **Create a virtual environment (recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Mac/Linux
   venv\Scripts\activate      # On Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app**

   ```bash
   streamlit run app.py
   ```

---

## 📥 How to Use

1. Export a WhatsApp chat (without media) from your phone.
2. Upload the exported `.txt` file in the sidebar of the app.
3. Explore the different analyses and visualizations!

---

## 📊 Example Output

* 📈 Top statistics of chat activity
* ☁️ WordCloud of common words
* 😀 Emoji usage distribution
* 📉 Sentiment over time graph

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
Feel free to fork the repo and submit a pull request.

---


💡 *Built with ❤️ using Streamlit & Hugging Face Transformers*

>>>>>>> fcfbd584046b32005c908f931ae5d9ff4a42871a
