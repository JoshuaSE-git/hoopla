#!/usr/bin/env python3
import argparse

from keyword_search.helpers import BM25_K1, BM25_B, DEFAULT_SEARCH_LIMIT
from keyword_search.commands import (
    build_command,
    search_command,
    tf_command,
    idf_command,
    tfidf_command,
    bmf25idf_command,
    bm25tf_command,
    bm25_command,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build docmap and index")

    tf_parser = subparsers.add_parser("tf", help="Get term frequency from document")
    tf_parser.add_argument("id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Target term")

    idf_parser = subparsers.add_parser("idf", help="Get IDF score for term")
    idf_parser.add_argument("term", type=str, help="Target term")

    tfidf_parser = subparsers.add_parser("tfidf", help="Get TF-IDF score from document")
    tfidf_parser.add_argument("id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Target term")

    bm25idf_parser = subparsers.add_parser("bm25idf", help="Get BM25-IDF score of term")
    bm25idf_parser.add_argument("term", type=str, help="Target term")

    bm25tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document id and term"
    )
    bm25tf_parser.add_argument("id", type=int, help="Document ID")
    bm25tf_parser.add_argument("term", type=str, help="Target term")
    bm25tf_parser.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1, help="K1 tuning parameter"
    )
    bm25tf_parser.add_argument(
        "b", type=float, nargs="?", default=BM25_B, help="B tuning parameter"
    )

    bm25search_parser = subparsers.add_parser(
        "bm25search", help="Search movies using full BM25 scoring"
    )
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument(
        "limit", nargs="?", default=DEFAULT_SEARCH_LIMIT, type=int, help="Results limit"
    )
    bm25search_parser.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1, help="K1 tuning parameter"
    )
    bm25search_parser.add_argument(
        "b", type=float, nargs="?", default=BM25_B, help="B tuning parameter"
    )

    args = parser.parse_args()

    match args.command:
        case "search":
            search_command(args.query)
        case "build":
            build_command()
        case "tf":
            tf_command(args.id, args.term)
        case "idf":
            idf_command(args.term)
        case "tfidf":
            tfidf_command(args.id, args.term)
        case "bm25idf":
            bmf25idf_command(args.term)
        case "bm25tf":
            bm25tf_command(args.id, args.term, args.k1, args.b)
        case "bm25search":
            bm25_command(args.query, args.limit, args.k1, args.b)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
