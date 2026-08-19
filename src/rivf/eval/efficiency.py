import tiktoken

# One consistent tokenizer across every method, including HippoRAG (which
# returns passages, not sentences) -- context tokens are the one fair common
# currency for comparing methods that select at different granularities.
_ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))
