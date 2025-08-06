import pandas as pd

# Load the full dataset
df = pd.read_csv("tmdb_5000_movies.csv")

# Take only the first 1000 rows
small_df = df.head(1000)

# Save to new smaller CSV file
small_df.to_csv("tmdb_small.csv", index=False)

print("Smaller dataset created: tmdb_small.csv")