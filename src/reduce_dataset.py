import pandas as pd

df = pd.read_csv("spotify_millsongdata.csv")

df = df.dropna(subset=["text"])
df = df.drop_duplicates(subset=["song", "artist"])
df = df.sample(10000, random_state=42)
df["text"] = df["text"].str[:500]
df = df[["artist", "song", "text"]]

df.to_csv("spotify_small.csv", index=False)
print(df.shape)


