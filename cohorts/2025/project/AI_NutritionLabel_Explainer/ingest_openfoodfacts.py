"""
AI Nutrition Label Explainer - Data Ingestion Script
----------------------------------------------------
Reads Open Food Facts data (.csv.gz) from the local data_ingestion folder,
filters 5 key categories, embeds with Ollama (nomic-embed-text),
and stores results in a local Qdrant collection for RAG retrieval.

Usage:
    python ingest_openfoodfacts_optimized.py
"""

import os
import re
import pandas as pd
from tqdm import tqdm
from langchain_ollama import ChatOllama, OllamaEmbeddings
from qdrant_client import QdrantClient, models
import uuid
# ----------------------------
# Configuration
# ----------------------------

# File and directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data_ingestion", "en.openfoodfacts.org.products.csv.gz")

# Qdrant configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "openfoodfacts_filtered"
VECTOR_SIZE = 768
DISTANCE_METRIC = models.Distance.COSINE

# Embedding model
EMBED_MODEL = "nomic-embed-text"

embedding_model = OllamaEmbeddings(model=EMBED_MODEL)

# Performance tuning
CHUNKSIZE = 1000       # rows per read iteration
BATCH_SIZE = 25        # vectors per upsert
MAX_RECORDS = 5000     # limit for local demo/testing

# Category filters (broad, representative mix)
TARGET_CATEGORIES = [
    "Breakfast cereals",
    "Beverages",
    "Snacks",
    "Dairies",
    "Prepared meals"
]


# ----------------------------
# Utility Functions
# ----------------------------

def clean_text(text: str) -> str:
    """Remove line breaks and compress whitespace."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text.strip())
    return text


def normalize_nutrient_value(value):
    """Safely convert nutrient values to floats."""
    try:
        return round(float(value), 2)
    except (ValueError, TypeError):
        return None


def category_match(category_field: str) -> bool:
    """Check if a product matches one of the target categories."""
    if not isinstance(category_field, str):
        return False
    return any(cat.lower() in category_field.lower() for cat in TARGET_CATEGORIES)


def format_product_text(row: pd.Series) -> str:
    """Compose a descriptive text string for embeddings."""
    nutriments = [
        ("Energy (kcal)", row.get("energy-kcal_100g")),
        ("Fat (g)", row.get("fat_100g")),
        ("Saturated fat (g)", row.get("saturated-fat_100g")),
        ("Sugars (g)", row.get("sugars_100g")),
        ("Fiber (g)", row.get("fiber_100g")),
        ("Protein (g)", row.get("proteins_100g")),
        ("Salt (g)", row.get("salt_100g")),
    ]

    facts = ". ".join(
        [f"{name}: {normalize_nutrient_value(val)}" for name, val in nutriments if val not in (None, "")]
    )

    text = (
        f"Product: {row.get('product_name', 'Unknown')}. "
        f"Brand: {row.get('brands', 'N/A')}. "
        f"Category: {row.get('categories', 'N/A')}. "
        f"Ingredients: {clean_text(row.get('ingredients_text', ''))}. "
        f"Nutritional Facts per 100g: {facts}. "
        f"Labels: {clean_text(row.get('labels_tags', ''))}. "
        f"Allergens: {clean_text(row.get('allergens_tags', ''))}."
    )
    return text


# ----------------------------
# Qdrant Setup
# ----------------------------

def init_qdrant():
    """Initialize or recreate Qdrant collection."""
    client = QdrantClient(url=QDRANT_URL)
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=DISTANCE_METRIC)
    )
    print(f"✅ Qdrant collection '{COLLECTION_NAME}' ready at {QDRANT_URL}")
    return client


# ----------------------------
# Main Ingestion Logic
# ----------------------------

def ingest_openfoodfacts():
    """Stream data from gzipped OFF CSV, filter, embed, and store in Qdrant."""
    print(f"📥 Reading data from {DATA_PATH} ...")

    qdrant = init_qdrant()
    total_records = 0
    batch_points = []

    # Stream the .gz file
    chunks = pd.read_csv(
        DATA_PATH,
        compression="gzip",
        sep="\t",
        low_memory=False,
        chunksize=CHUNKSIZE
    )

    for chunk_idx, chunk in enumerate(chunks):
        # Filter only meaningful rows
        chunk = chunk[chunk["product_name"].notna()]
        chunk = chunk[chunk["categories"].apply(category_match)]

        if chunk.empty:
            continue

        print(f"🔍 Processing chunk {chunk_idx} ({len(chunk)} records after filtering)")

        for _, row in tqdm(chunk.iterrows(), total=len(chunk), desc=f"Chunk {chunk_idx}"):
            try:
                text = format_product_text(row)
                if len(text) < 50:
                    continue

                # emb = embed(model=EMBED_MODEL, input=text)["embedding"]

                
                emb = embedding_model.embed_query(text)
                if not emb:
                    continue

                payload = {
                    "product_name": row.get("product_name"),
                    "brands": row.get("brands"),
                    "categories": row.get("categories"),
                    "url": row.get("url"),
                    "nutrition_grade": row.get("nutrition_grade_fr"),
                }

                point = models.PointStruct(
                    # id=int(row.get("code", total_records + 1)),
                    id=str(row.get("code") or uuid.uuid4()),
                    vector=emb,
                    payload=payload
                )
                batch_points.append(point)

                if len(batch_points) >= BATCH_SIZE:
                    qdrant.upsert(collection_name=COLLECTION_NAME, points=batch_points)
                    batch_points = []

                total_records += 1
                if total_records >= MAX_RECORDS:
                    print("⚙️ Reached max record limit.")
                    break

            except Exception as e:
                print(f"⚠️ Skipped product due to error: {e}")

        if total_records >= MAX_RECORDS:
            break

    # Flush any remaining points
    if batch_points:
        qdrant.upsert(collection_name=COLLECTION_NAME, points=batch_points)

    print(f"✅ Ingestion complete. Indexed {total_records} products in Qdrant.")


if __name__ == "__main__":
    ingest_openfoodfacts()
