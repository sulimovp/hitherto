from casefile.clients.discourse_html import extract_discourse_topic_urls


def test_extract_topic_urls_from_search_html():
    html = """
    <a href="/t/state-of-pytorch-core-september-2021-edition/332">State of PyTorch</a>
    <a href="https://dev-discuss.pytorch.org/t/what-and-why-is-torch-dispatch/557">dispatch</a>
    """
    urls = extract_discourse_topic_urls(
        html, "https://dev-discuss.pytorch.org", limit=5
    )
    assert "https://dev-discuss.pytorch.org/t/state-of-pytorch-core-september-2021-edition/332" in urls
    assert "https://dev-discuss.pytorch.org/t/what-and-why-is-torch-dispatch/557" in urls


def test_extract_skips_search_paths():
    html = '<a href="/search?q=MaskedTensor">search</a>'
    assert extract_discourse_topic_urls(html, "https://dev-discuss.pytorch.org") == []
