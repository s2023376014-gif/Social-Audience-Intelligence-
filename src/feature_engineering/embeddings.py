"""Comment embeddings with on-disk caching.

Primary model: sentence-transformers 'all-MiniLM-L6-v2' (needs one-time download, runs on CPU).
Fallback (offline / not installed): character n-gram TF-IDF + SVD. Character n-grams actually cope with
Roman Urdu spelling variation reasonably, but they are NOT semantic: 'price' and 'how much' stay unrelated.
The backend used is always recorded so results are never silently mixed.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

import config

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MULTILINGUAL_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def _fingerprint(texts, backend: str) -> str:
    h = hashlib.sha1(backend.encode())
    for t in texts:
        h.update(t.encode("utf-8", "ignore"))
    return h.hexdigest()[:16]


def _embed_transformer(texts, model_name, batch_size=64):
    from sentence_transformers import SentenceTransformer  # imported lazily: heavy
    model = SentenceTransformer(model_name, device="cpu")
    return model.encode(texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)


def _embed_tfidf(texts, dim=64, seed=config.RANDOM_SEED):
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    X = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True).fit_transform(texts)
    k = max(2, min(dim, X.shape[1] - 1, X.shape[0] - 1))
    return normalize(TruncatedSVD(k, random_state=seed).fit_transform(X))


def embed_texts(texts, backend: str = "auto", model_name: str = DEFAULT_MODEL, cache_dir: Path = None):
    """Return (embeddings float32 [n, d], backend_used). Unique texts are embedded once (big speed-up,
    since many comments repeat). backend: 'auto' | 'transformer' | 'tfidf'."""
    cache_dir = Path(cache_dir or config.MODELS_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    texts = [t if isinstance(t, str) and t else " " for t in texts]
    uniq = sorted(set(texts))
    want = model_name if backend != "tfidf" else "tfidf"
    key = _fingerprint(uniq, want)
    cache_f, meta_f = cache_dir / f"emb_{key}.npy", cache_dir / f"emb_{key}.json"
    if cache_f.exists() and meta_f.exists():
        E, used = np.load(cache_f), json.loads(meta_f.read_text())["backend"]
    else:
        used = "tfidf-char-svd"
        E = None
        if backend in ("auto", "transformer"):
            try:
                E, used = _embed_transformer(uniq, model_name), model_name
            except Exception as exc:
                if backend == "transformer":
                    raise RuntimeError(f"Transformer embedding failed: {exc}") from exc
        if E is None:
            E = _embed_tfidf(uniq)
        E = E.astype("float32")
        np.save(cache_f, E)
        meta_f.write_text(json.dumps({"backend": used, "n_unique": len(uniq)}))
    index = {t: i for i, t in enumerate(uniq)}
    return E[[index[t] for t in texts]], used


def similarity_check(pairs, backend="auto", model_name=DEFAULT_MODEL):
    """Cosine similarity for hand-picked pairs, to judge Roman Urdu handling honestly."""
    flat = [t for p in pairs for t in p]
    E, used = embed_texts(flat, backend, model_name, cache_dir=config.MODELS_DIR / "similarity_checks")
    sims = [float(E[2 * i] @ E[2 * i + 1]) for i in range(len(pairs))]
    return sims, used
