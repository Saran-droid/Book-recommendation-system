
# 📚 Semantic Book Recommender System

## Overview

This project is an intelligent book recommendation system that suggests books based on user-provided descriptions, genres, and emotional tones. It leverages semantic similarity using embeddings, a vector store (Chroma with LangChain), and a visually rich Gradio dashboard to provide highly personalized book recommendations.

---

## Features

- 🔍 **Semantic Search**: Uses sentence-transformers for deep semantic matching.
- 🎨 **Gradio Dashboard**: Interactive and stylish frontend for end users.
- 🎭 **Emotional Filtering**: Recommends books by tone (e.g., Happy, Sad, Surprising).
- 🧠 **Vector Database**: Efficient similarity search with Chroma and FAISS.
- 📜 **Custom Styling**: Beautiful UI enhancements using CSS.

---

## Project Structure

```
├── gradio-dashboard.py          # Main Gradio application
├── tagged_description.txt       # Book descriptions used for semantic matching
├── df_cleaned.csv               # Cleaned book metadata (optional, not referenced)
├── *.ipynb                      # Supporting notebooks (exploration, sentiment analysis, etc.)
```

---

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/book-recommender.git
   cd book-recommender
   ```

2. **Install Requirements**
   Create a virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   Requirements may include:
   ```
   gradio
   pandas
   numpy
   langchain
   chromadb
   sentence-transformers
   flask
   ```

3. **Add `.env` File**
   Place any API keys or environment variables here (if needed for embeddings, etc.)

4. **Run the App**
   ```bash
   python gradio-dashboard.py
   ```
   This launches the dashboard locally at `http://127.0.0.1:7860`.

---

## How It Works

1. Loads book metadata and large thumbnails from a CSV file.
2. Loads descriptions from `tagged_description.txt`, splits into chunks.
3. Embeds descriptions using `all-MiniLM-L6-v2`.
4. Stores in Chroma vector database.
5. Retrieves most similar books based on user input using semantic similarity.
6. Filters results based on tone and category preferences.
7. Presents recommendations with styled UI in Gradio.

---

## Example Queries

- *"A science fiction tale about alien friendship and space travel"*
- *"A romance set in post-war France with a tragic ending"*
- *"A philosophical book exploring the nature of love and pain"*

---

## Notebooks

These provide extra context and exploration:
- `data-exploration.ipynb`: EDA on the books dataset.
- `text-classification.ipynb`: Model training for tone classification.
- `sentiment-analysis.ipynb`: Sentiment tagging experiments.
- `vector-search.ipynb`: Embedding and retrieval benchmarking.

---

## Credits

- Book metadata from OpenLibrary
- Embeddings by HuggingFace Transformers
- UI with Gradio
- Vector search via LangChain + Chroma

---

## License

MIT License
