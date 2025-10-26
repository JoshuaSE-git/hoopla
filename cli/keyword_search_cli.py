#!/usr/bin/env python3

import argparse
import json
import string

MOVIE_DATA_PATH = "data/movies.json"
MAX_SEARCH_RESULTS = 5


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            search(args.query, MOVIE_DATA_PATH, MAX_SEARCH_RESULTS)
        case _:
            parser.print_help()


def search(query: str, filepath: str, num_results: int):
    with open(filepath, "r") as f:
        data = json.load(f)

    translation_map = str.maketrans({c: None for c in string.punctuation})

    print("Searching for: " + query)
    results = []
    processed_query = process(query, translation_map)
    for movie in data["movies"]:
        processed_title = process(movie["title"], translation_map)

        for s1 in processed_query:
            for s2 in processed_title:
                if s1 in s2:
                    results.append(movie)

        if len(results) >= num_results:
            break

    for i, movie in enumerate(sorted(results, key=lambda x: x["id"]), start=1):
        print(f"{i}. {movie['title']}")


def process(s: str, trans: dict):
    return set(filter(lambda x: x != "", s.lower().translate(trans).split()))


if __name__ == "__main__":
    main()
