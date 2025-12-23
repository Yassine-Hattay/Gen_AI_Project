import pickle
with open("data/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)
print(len(chunks))
print(chunks[0])
