"""
Frozen sentence embeddings for explanation targets, with a disk cache.

The auxiliary head regresses onto `E(text)` for a frozen encoder `E`.
Two properties matter and are easy to get wrong:

- **Frozen** means the same text always yields the same vector, for the
  life of a dataset. The embedding model and its version therefore
  belong in the manifest; vectors from a different model are a different
  dataset, not a drop-in substitute.
- **Cached by text**, because explanations repeat. A repeat costs a
  dictionary lookup rather than a request, which is what makes
  per-consultation embedding affordable at all.

The cache is keyed by `(model, text)` and persisted as JSON so a resumed
run reuses what an interrupted one already paid for.

Nothing here calls a chat model. It embeds text that a teacher already
produced.
"""

import hashlib
import json
import os
import threading


class EmbeddingProvider:
    """
    Embed explanation text, reusing anything already embedded.

    `dimension` is discovered from the first response rather than
    assumed, so a model change surfaces as an explicit mismatch instead
    of a silent shape error deep in the training loop.
    """

    def __init__(self, model='text-embedding-3-small', cache_path=None,
                 client=None):
        """
        Build the provider, loading any existing cache for this model.

        `client` is injectable so tests can exercise the caching and
        dimension logic without a network call.
        """

        self.model = model
        self.cache_path = cache_path
        self.dimension = None
        self._cache = {}
        self._lock = threading.Lock()
        self._client = client
        self.num_requests = 0
        self.num_texts_embedded = 0
        self.num_cache_hits = 0
        # Tokens as the API reports them, not as we estimate them.
        # Absent from a response (or from a test double), the counter
        # stays None so the summary says "unknown" instead of "0".
        self.num_tokens = None
        self._load()

    def _load(self):
        """
        Read a persisted cache, ignoring entries from another model.
        """

        if not self.cache_path or not os.path.exists(self.cache_path):
            return
        with open(self.cache_path, encoding='utf-8') as handle:
            payload = json.load(handle)
        if payload.get('model') != self.model:
            # A cache from another encoder is a different dataset. Do
            # not silently mix them.
            return
        self.dimension = payload.get('dimension')
        self._cache = {k: v for k, v in payload.get('vectors', {}).items()}

    def save(self):
        """
        Persist the cache so an interrupted run does not re-pay.
        """

        if not self.cache_path:
            return
        os.makedirs(os.path.dirname(self.cache_path) or '.', exist_ok=True)
        # Write to a temporary file and replace, so an interruption
        # leaves the previous cache intact rather than a truncated one.
        # NOTE: this is not multi-process safe. One writer per cache
        # path; a shared path across concurrent runs needs a lock.
        temporary = f'{self.cache_path}.tmp{os.getpid()}'
        with open(temporary, 'w', encoding='utf-8') as handle:
            json.dump({
                'model': self.model,
                'dimension': self.dimension,
                'vectors': self._cache,
            }, handle)
        os.replace(temporary, self.cache_path)

    @staticmethod
    def key_for(text):
        """
        Stable content hash of one explanation string.

        This is the EMBEDDING CACHE key, and nothing else. It is
        explicitly **not** a transition id: the same explanation recurs
        on distinct transitions, so using it as a sample id collapses
        them and lets the R3 shuffler hand a row its own target back.
        Transition ids are assigned by the caller from run, iteration,
        step and environment.
        """

        return hashlib.sha256(text.encode('utf-8')).hexdigest()[:32]

    def _client_or_build(self):
        """
        Return the injected client, or build the project's standard one.
        """

        if self._client is None:
            from teachers.base import build_openai_client
            self._client = build_openai_client()
        return self._client

    def embed(self, texts):
        """
        Return one L2-normalized vector per text, in order.

        Only texts absent from the cache are sent. The request count is
        tracked separately from the text count so the manifest can
        record what was actually paid for.
        """

        import numpy as np

        keys = [self.key_for(t) for t in texts]
        with self._lock:
            # DEDUPLICATE within the request. Building `missing` per
            # occurrence would send a previously unseen string once for
            # every transition that carries it, so a rollout where the
            # teacher repeats itself pays several times for one vector.
            # A dict keyed by the text hash keeps first occurrence only.
            missing_map = {}
            for key, text in zip(keys, texts):
                if key not in self._cache and key not in missing_map:
                    missing_map[key] = text
            missing = list(missing_map.items())
            self.num_cache_hits += sum(
                1 for k in keys if k in self._cache
            )

        if missing:
            # Guard actual text before a paid dispatch. The embedding
            # tokenizer is byte-BPE too; UTF-8 length is a conservative
            # bound. Unknown or oversized input must not spend a hold
            # computed for a smaller request.
            ceiling = int(os.getenv('LLM_MAX_EMBED_TOKENS', '') or 0)
            if ceiling and any(len(text.encode('utf-8')) > ceiling
                               for _, text in missing):
                raise ValueError('Explanation exceeds the reserved '
                                 'embedding input bound; not sending')
            response = self._client_or_build().embeddings.create(
                model=self.model, input=[t for _, t in missing]
            )
            self.num_requests += 1
            self.num_texts_embedded += len(missing)
            usage = getattr(response, 'usage', None)
            tokens = getattr(usage, 'total_tokens', None)
            if tokens is not None:
                self.num_tokens = (self.num_tokens or 0) + int(tokens)
            if len(response.data) != len(missing):
                raise ValueError(
                    f'embedding response returned {len(response.data)} '
                    f'vectors for {len(missing)} texts'
                )
            for (key, _), item in zip(missing, response.data):
                vector = np.asarray(item.embedding, dtype=np.float64)
                norm = float(np.linalg.norm(vector))
                vector = vector / max(norm, 1e-12)
                if not bool(np.isfinite(vector).all()):
                    raise ValueError(
                        'embedding response contained a non-finite '
                        'value; refusing to train on it'
                    )
                if self.dimension is None:
                    self.dimension = int(vector.shape[0])
                elif int(vector.shape[0]) != self.dimension:
                    raise ValueError(
                        f'embedding dimension changed from '
                        f'{self.dimension} to {vector.shape[0]}; a '
                        f'different encoder is a different dataset'
                    )
                with self._lock:
                    self._cache[key] = vector.tolist()

        return [np.asarray(self._cache[k], dtype=np.float32) for k in keys]

    def stats(self, price_per_million=0.0):
        """
        Accounting for the manifest: what was requested versus reused.

        `requests_issued` counts calls this class made. It is NOT a
        count of HTTP attempts: the SDK retries internally and those
        retries are invisible here, so the field is named for what it
        can actually claim and the gap is stated rather than left for a
        reader to assume away.

        Dollars are reported only when a price is supplied. A rate
        baked into the source would keep reporting a number long after
        it stopped being the price.
        """

        dollars = None
        if price_per_million > 0 and self.num_tokens is not None:
            dollars = self.num_tokens / 1e6 * price_per_million
        return {
            'embedding_model': self.model,
            'dimension': self.dimension,
            'requests_issued': self.num_requests,
            'sdk_retries_not_counted': True,
            'texts_embedded': self.num_texts_embedded,
            'cache_hits': self.num_cache_hits,
            'cached_unique_texts': len(self._cache),
            'tokens': self.num_tokens,
            'price_per_million': price_per_million or None,
            'dollars': dollars,
        }
