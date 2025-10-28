#!/usr/bin/env python3
import argparse

from commands import (
    build_command,
    search_command,
    tf_command,
    idf_command,
    tfidf_command,
    bmf25idf_command,
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
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
