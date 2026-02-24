import streamlit as st
import pandas as pd
import numpy as np
import faiss
import torch
import re
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------

st.set_page_config(page_title="Fashion Search AI", layout="wide")

st.title("👗 Fashion Search AI with AI Recommendation")
st.write("Search fashion products using natural language.")

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("fashion_small.csv")

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["avg_rating"] = pd.to_numeric(df["avg_rating"], errors="coerce")
    df["colour"] = df["colour"].astype(str).str.lower()

    df["combined_text"] = (
        df["name"].astype(str) + " " +
        df["brand"].astype(str) + " " +
        df["colour"].astype(str) + " " +
        df["description"].astype(str)
    )

    return df

df = load_data()

# -------------------------------------------------
# LOAD EMBEDDING MODEL
# -------------------------------------------------

@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    return model

model = load_model()

# -------------------------------------------------
# CREATE EMBEDDINGS
# -------------------------------------------------

@st.cache_resource
def create_embeddings(df):
    embeddings = model.encode(
        df["combined_text"].tolist(),
        batch_size=32,
        convert_to_numpy=True
    )
    return embeddings

embeddings = create_embeddings(df)

# -------------------------------------------------
# CREATE FAISS INDEX
# -------------------------------------------------

@st.cache_resource
def load_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

index = load_index(embeddings)

# -------------------------------------------------
# LOAD GENERATION MODEL (LLM)
# -------------------------------------------------

@st.cache_resource
def load_generator():
    generator = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        device=-1  # CPU for Streamlit Cloud
    )
    return generator

generator = load_generator()

# -------------------------------------------------
# QUERY UNDERSTANDING
# -------------------------------------------------

def extract_price(query):
    match = re.search(r"(under|below)\s*(\d+)", query.lower())
    if match:
        return int(match.group(2))
    return None

def extract_colour(query):
    available_colours = df["colour"].dropna().unique().tolist()
    for colour in available_colours:
        if colour in query.lower():
            return colour
    return None

# -------------------------------------------------
# SEARCH FUNCTION
# -------------------------------------------------

def smart_search(query, top_k=20):

    max_price = extract_price(query)
    colour = extract_colour(query)

    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)

    results = df.iloc[indices[0]].copy()

    if max_price is not None:
        results = results[results["price"] <= max_price]

    if colour is not None:
        results = results[results["colour"] == colour]

    if len(results) == 0:
        return results

    results = results.reset_index(drop=True)

    sim_scores = 1 / (1 + distances[0][:len(results)])
    results["score"] = sim_scores + 0.1 * results["avg_rating"].fillna(0)

    results = results.sort_values(by="score", ascending=False)

    return results.head(5)

# -------------------------------------------------
# GENERATION FUNCTION
# -------------------------------------------------

def generate_response(query, results_df):

    if len(results_df) == 0:
        return "No relevant products found for your query."

    context = ""

    for _, row in results_df.iterrows():
        context += f"{row['name']} by {row['brand']} priced at ₹{row['price']} with rating {row['avg_rating']}.\n"

    prompt = f"""
    User Query: {query}

    Based on these products:
    {context}

    Provide a helpful shopping recommendation in 3-4 sentences.
    """

    output = generator(prompt, max_length=120, do_sample=True)[0]["generated_text"]

    return output

# -------------------------------------------------
# UI
# -------------------------------------------------

query = st.text_input("Enter your fashion query")

if st.button("Search"):

    if query.strip() == "":
        st.warning("Please enter a query.")
    else:
        results = smart_search(query)

        if len(results) == 0:
            st.warning("No matching products found.")
        else:
            # Show products
            for _, row in results.iterrows():

                col1, col2 = st.columns([1, 3])

                with col1:
                    if pd.notna(row["img"]):
                        st.image(row["img"], width=150)

                with col2:
                    st.subheader(row["name"])
                    st.write(f"Brand: {row['brand']}")
                    st.write(f"Colour: {row['colour']}")
                    st.write(f"Price: ₹{row['price']}")
                    st.write(f"Rating: {row['avg_rating']}")
                    st.markdown("---")

            # Show AI Recommendation
            st.subheader("🧠 AI Recommendation")
            ai_response = generate_response(query, results)
            st.write(ai_response)
