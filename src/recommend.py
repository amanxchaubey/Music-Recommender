import os
import logging
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "spotify_small.csv")


logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logging.info("🔁 Loading dataset...")


df = pd.read_csv(CSV_PATH)


required_cols = ["song", "artist", "text"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' is missing in CSV!")

df["text"] = df["text"].fillna("")
logging.info("✅ Dataset loaded: %d songs", len(df))


logging.info("🔁 Building TF-IDF matrix...")
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
tfidf_matrix = vectorizer.fit_transform(df["text"])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
logging.info("✅ Similarity matrix ready.")


def recommend_songs(song_name, top_n=5):
    logging.info("🎵 Recommending songs for: %s", song_name)

    matches = df[df["song"].str.lower() == song_name.lower()]

    if matches.empty:
        logging.warning("⚠️ Song not found.")
        return pd.DataFrame(columns=["artist", "song"])

    idx = matches.index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]
    song_indices = [i[0] for i in sim_scores]

    result_df = df.loc[song_indices, ["artist", "song"]].reset_index(drop=True)
    result_df.index += 1
    result_df.index.name = "S.No."

    logging.info("✅ %d recommendations generated.", top_n)
    return result_df


