class Application:
    def __init__(self, data, stop_words, stemmer, translation):
        self.translation = translation
        self.stemmer = stemmer
        self.stopwords = stop_words
        self.data = data

    def search(self, query, num_results):
        results = []
        query_tokens = self._process(query)
        for movie in self.data["movies"]:
            title_tokens = self._process(movie["title"])

            if self._match(query_tokens, title_tokens):
                results.append(movie)

            if len(results) >= num_results:
                break

        for i, movie in enumerate(sorted(results, key=lambda x: x["id"]), start=1):
            print(f"{i}. {movie['title']}")

    def _match(self, query_tokens: set, title_tokens: set) -> bool:
        for s1 in query_tokens:
            for s2 in title_tokens:
                if s1 in s2:
                    return True
        return False

    def _process(self, s: str) -> set:
        tokens = set()
        for tok in s.lower().translate(self.translation).split():
            if tok and tok not in self.stopwords:
                tokens.add(self.stemmer.stem(tok))

        return tokens
