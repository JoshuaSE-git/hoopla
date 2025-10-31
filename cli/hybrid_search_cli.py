import argparse

from hybrid_search.commands import handler_normalize


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser(
        "normalize", help="Normalize scores to the range 0 - 1"
    )
    normalize_parser.add_argument(
        "scores", type=float, nargs="+", help="The target score(s)"
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            handler_normalize(args.scores)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
