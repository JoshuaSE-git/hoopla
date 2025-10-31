#!/usr/bin/env python3

import argparse

from semantic_search.helpers import DEFAULT_SEARCH_LIMIT

from semantic_search.commands import (
    handler_verify_model,
    handler_embed_text,
    handler_verify_embeddings,
    handler_embed_query,
    handler_search,
    handler_chunk,
    handler_semantic_chunk,
    handler_embed_chunks,
    handler_search_chunks,
)


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify", help="Display model information")

    embed_text_parser = subparsers.add_parser(
        "embed_text", help="Create embedding from text"
    )
    embed_text_parser.add_argument("text", help="The target text")

    verify_embeddings_parser = subparsers.add_parser(
        "verify_embeddings", help="Load/generate and verify embeddings"
    )

    embed_query_parser = subparsers.add_parser(
        "embedquery", help="Create embedding from query"
    )
    embed_query_parser.add_argument("query", help="The target query")

    search_parser = subparsers.add_parser(
        "search", help="Search for movies using semantic search"
    )
    search_parser.add_argument("query", help="The target query")
    search_parser.add_argument(
        "-l",
        "--limit",
        type=int,
        default=DEFAULT_SEARCH_LIMIT,
        help="Number of results",
    )

    chunk_parser = subparsers.add_parser("chunk", help="Turn text into n sized chunks")
    chunk_parser.add_argument("text", help="The target text")
    chunk_parser.add_argument(
        "--chunk-size", type=int, default=200, help="The number of words per chunk"
    )
    chunk_parser.add_argument(
        "--overlap", type=int, help="The overlap size between chunks"
    )

    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk", help="Turn text into n sized sentence chunks"
    )
    semantic_chunk_parser.add_argument("text", help="The target text")
    semantic_chunk_parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=4,
        help="The number of sentences per chunk",
    )
    semantic_chunk_parser.add_argument(
        "--overlap", type=int, default=0, help="The overlap size between chunks"
    )

    embed_chunks_parser = subparsers.add_parser(
        "embed_chunks", help="Create chunked embeddings from document"
    )

    search_chunked_parser = subparsers.add_parser(
        "search_chunked", help="Search for movies using chunked semantic search"
    )
    search_chunked_parser.add_argument("query", help="The target query")
    search_chunked_parser.add_argument(
        "--limit", type=int, default=5, help="The number of results"
    )

    args = parser.parse_args()

    match args.command:
        case "verify":
            handler_verify_model()
        case "embed_text":
            handler_embed_text(args.text)
        case "verify_embeddings":
            handler_verify_embeddings()
        case "embedquery":
            handler_embed_query(args.query)
        case "search":
            handler_search(args.query, args.limit)
        case "chunk":
            handler_chunk(args.text, args.chunk_size, args.overlap)
        case "semantic_chunk":
            handler_semantic_chunk(args.text, args.max_chunk_size, args.overlap)
        case "embed_chunks":
            handler_embed_chunks()
        case "search_chunked":
            handler_search_chunks(args.query, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
