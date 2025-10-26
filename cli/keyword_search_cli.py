#!/usr/bin/env python3
import argparse
import json
import string
import application

from nltk.stem import PorterStemmer


MOVIE_DATA_PATH = "data/movies.json"
STOP_WORDS_DATA_PATH = "data/stopwords.txt"
MAX_SEARCH_RESULTS = 5


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    stemmer = PorterStemmer()
    translation = str.maketrans("", "", string.punctuation)

    with open(MOVIE_DATA_PATH, "r") as f:
        data = json.load(f)

    with open(STOP_WORDS_DATA_PATH, "r") as f:
        stopwords = f.read().splitlines()

    app = application.Application(
        data=data, stop_words=stopwords, stemmer=stemmer, translation=translation
    )

    match args.command:
        case "search":
            app.search(args.query, MAX_SEARCH_RESULTS)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
