# preprocess.py - FOR LYRICS/TEXT DATASET (MEMORY OPTIMIZED)
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')

logging.info("🚀 Starting preprocessing for TEXT-BASED recommendations...")

try:
    df = pd.read_csv("spotify_millsongdata.csv")
    logging.info(f"✅ Dataset loaded: {len(df)} rows")

    # Clean data
    df = df.drop_duplicates(subset=['song', 'artist'], keep='first')
    df = df.dropna(subset=['song', 'artist', 'text'])
    logging.info(f"✅ Dataset after cleaning: {len(df)} songs")

    # SAMPLE ONLY 5000 SONGS TO SAVE MEMORY
    if len(df) > 5000:
        df = df.sample(n=5000, random_state=42)
        logging.info(f"📊 Sampled dataset to 5000 songs to optimize memory")

    # Fill empty text with song name
    df['text'] = df['text'].fillna(df['song'])

    # Reset index
    df = df.reset_index(drop=True)

    # Create TF-IDF matrix from lyrics
    logging.info("📝 Creating TF-IDF vectors from lyrics...")
    tfidf = TfidfVectorizer(
        max_features=3000,  # Reduced from 5000
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2
    )

    tfidf_matrix = tfidf.fit_transform(df['text'])
    logging.info(f"✅ TF-IDF matrix created: {tfidf_matrix.shape}")

    # Calculate cosine similarity
    logging.info("🧮 Calculating cosine similarity from lyrics...")
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    logging.info(f"✅ Similarity matrix: {cosine_sim.shape}")

    # Save files
    logging.info("💾 Saving files...")
    joblib.dump(df, 'df_cleaned.pkl')
    joblib.dump(cosine_sim, 'cosine_sim.pkl')

    logging.info("✅ Successfully saved files!")
    logging.info(f"📊 Total songs in final dataset: {len(df)}")

    print(f"\n🎵 First 10 songs:")
    print(df[['song', 'artist']].head(10))

    print(f"\n🎵 Last 10 songs:")
    print(df[['song', 'artist']].tail(10))

except Exception as e:
    logging.error(f"❌ Error: {e}")
    raise e