import json


def search(query: str, filepath: str):
    with open(filepath, "r") as f:
        data = json.load(f)

    for i, movie in enumerate(data["movies"], start=1):
        if query in movie["title"]:
            print(f"{i}. {movie['title']}")
