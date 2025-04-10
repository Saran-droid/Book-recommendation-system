import pandas as pd
import numpy as np
from flask.cli import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import gradio as gr
import time

# Load environment variables
load_dotenv()

# Load book data
books = pd.read_csv("books_emo.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"].fillna("cover.jpg", inplace=True)

# Load and process document embeddings
raw_doc = TextLoader("tagged_description.txt", encoding="utf-8").load()
text_splitter = CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator="\n")
doc = text_splitter.split_documents(raw_doc)

db_books = Chroma.from_documents(
    doc, embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
)


def retrieve_semantic_recommendations(query: str, category: str = None, tone: str = None, initial_topk: int = 50,
                                      final_topk: int = 16) -> pd.DataFrame:
    recs = db_books.similarity_search(query, k=initial_topk)
    book_list = [int(rec.page_content[:13].strip().replace('"', '')) for rec in recs]
    book_recs = books[books["isbn13"].isin(book_list)]

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category][:final_topk]
    else:
        book_recs = book_recs[:final_topk]

    if tone == "Happy":
        book_recs = book_recs.sort_values(by="joy", ascending=False)
    elif tone == "Surprising":
        book_recs = book_recs.sort_values(by="surprise", ascending=False)
    elif tone == "Angry":
        book_recs = book_recs.sort_values(by="anger", ascending=False)
    elif tone == "Suspenseful":
        book_recs = book_recs.sort_values(by="fear", ascending=False)
    elif tone == "Sad":
        book_recs = book_recs.sort_values(by="sadness", ascending=False)

    return book_recs


def recommend_books(query: str, category: str, tone: str):
    if not query.strip():
        gr.Warning("Please enter a book description or theme")
        return []

    # Show loading animation
    with gr.Row():
        loading = gr.Markdown(
            "<div style='text-align: center; margin: 20px;'><div class='loader'></div><p>Finding your perfect books...</p></div>")

    # Simulate processing time for better UX
    time.sleep(1.5)

    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        authors_split = row["authors"].split(";")
        authors_str = " and ".join(authors_split) if len(
            authors_split) <= 2 else f"{', '.join(authors_split[:-1])} and {authors_split[-1]}"

        # Create the image path and caption
        image_path = row["large_thumbnail"]
        caption = f"""
        <div style='padding: 10px;'>
            <h3 style='margin: 0 0 5px 0; color: #2a2a2a;'>{row['title']}</h3>
            <p style='margin: 0 0 5px 0; color: #555; font-style: italic;'>by {authors_str}</p>
            <p style='margin: 0; color: #444; font-size: 0.9em;'>{row['description'][:150]}...</p>
            <div style='margin-top: 8px;'>
                <span style='display: inline-block; padding: 3px 8px; background: #f0f0f0; border-radius: 12px; 
                font-size: 0.8em; margin-right: 5px;'>{row['simple_categories']}</span>
            </div>
        </div>
        """
        results.append((image_path, caption))

    # Return properly formatted list of (image_path, caption) tuples
    return results


# Define dropdown choices
categories = ["All"] + sorted(books["simple_categories"].unique())
tone = ["All", "Happy", "Surprising", "Suspenseful", "Angry", "Sad"]

# Custom CSS for enhanced styling
custom_css = """
:root {
    --primary: #4f46e5;
    --secondary: #f9fafb;
    --accent: #f59e0b;
}

.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto;
}

h1 {
    text-align: center;
    color: var(--primary) !important;
    margin-bottom: 20px !important;
}

.dark h1 {
    color: white !important;
}

.loader {
    border: 4px solid #f3f3f3;
    border-top: 4px solid var(--primary);
    border-radius: 50%;
    width: 40px;
    height: 40px;
    animation: spin 1s linear infinite;
    margin: 0 auto 10px auto;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.gallery-item {
    border-radius: 12px !important;
    overflow: hidden !important;
    transition: transform 0.2s !important;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
}

.gallery-item:hover {
    transform: scale(1.03) !important;
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15) !important;
}

button {
    background: var(--primary) !important;
    border: none !important;
    transition: all 0.2s !important;
}

button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 8px rgba(79, 70, 229, 0.3) !important;
}

.input-section {
    background: var(--secondary) !important;
    padding: 20px !important;
    border-radius: 12px !important;
    margin-bottom: 20px !important;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05) !important;
}

.dark .input-section {
    background: #1e1e1e !important;
}

footer {
    text-align: center !important;
    margin-top: 30px !important;
    color: #666 !important;
    font-size: 0.9em !important;
}
"""

# Create the Gradio interface with modern styling
with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as dashboard:
    # Header section
    gr.Markdown("""
    <div style='text-align: center; margin-bottom: 30px;'>
        <h1 style='margin-bottom: 10px;'>📚 Semantic Book Recommender</h1>
        <p style='font-size: 1.1em; color: #555; max-width: 700px; margin: 0 auto;'>
            Discover your next favorite read based on themes, genres, and emotional tones.
            Simply describe what you're looking for and we'll find the perfect matches!
        </p>
    </div>
    """)

    # Input section with cards
    with gr.Row(variant="panel", elem_classes="input-section"):
        with gr.Column():
            user_query = gr.Textbox(
                label="Describe your ideal book:",
                placeholder="e.g. A sci-fi adventure about space exploration and alien friendships",
                lines=3,
                elem_id="query-box"
            )

            with gr.Row():
                category_dropdown = gr.Dropdown(
                    choices=categories,
                    label="Filter by Category",
                    value="All",
                    interactive=True
                )
                tone_dropdown = gr.Dropdown(
                    choices=tone,
                    label="Filter by Emotional Tone",
                    value="All",
                    interactive=True
                )

            submit_button = gr.Button(
                "🔍 Find Recommendations",
                variant="primary",
                size="lg",
                elem_id="submit-btn"
            )

    # Results section
    gr.Markdown("## 📖 Recommended Books")
    with gr.Row():
        output = gr.Gallery(
            label="",
            columns=4,
            rows=2,
            height="auto",
            object_fit="contain",
            preview=False
        )

    # Footer
    gr.Markdown("""
    <footer>
        <p>Built with ❤️ using Gradio, LangChain, and Chroma | Book data from OpenLibrary</p>
    </footer>
    """)

    # Examples for quick start
    examples = gr.Examples(
        examples=[
            ["A mystery novel with unexpected twists", "Fiction", "Surprising"],
            ["A story about love and loss", "Fiction", "Sad"],
            ["Children's book about friendship", "Children's Fiction", "Happy"],
            ["Technology book about artificial intelligence", "Nonfiction", "All"]
        ],
        inputs=[user_query, category_dropdown, tone_dropdown],
        label="Try these examples:"
    )

    submit_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output,
        api_name="recommend"
    )

if __name__ == "__main__":
    dashboard.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        favicon_path="https://cdn-icons-png.flaticon.com/512/2232/2232688.png"
    )