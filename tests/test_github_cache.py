"""GitHub search disk cache."""

from casefile.cache.github_search import GitHubSearchCache


def test_search_cache_roundtrip(tmp_path):
    cache = GitHubSearchCache(tmp_path, ttl_seconds=3600)
    page = {"items": [{"number": 1, "title": "test"}], "total_count": 42}
    assert cache.get("repo:o/r is:issue foo") is None
    cache.set("repo:o/r is:issue foo", page)
    assert cache.get("repo:o/r is:issue foo") == page
