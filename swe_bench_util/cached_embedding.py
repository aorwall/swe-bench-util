import asyncio
import hashlib
import json
import os
from typing import List

from llama_index.core.base.embeddings.base import BaseEmbedding, Embedding
from llama_index.core.callbacks import CallbackManager
from pydantic import Field, PrivateAttr


class CachedEmbedding(BaseEmbedding):

    embed_model: BaseEmbedding = PrivateAttr()
    cache: dict = PrivateAttr()
    cache_file: str = PrivateAttr()

    def __init__(self, persist_dir: str, embed_model: BaseEmbedding, **kwargs) -> None:
        if persist_dir:
            cache_file = f"{persist_dir}/{embed_model.model_name}.json"
            if os.path.exists(cache_file):
                cache = json.load(open(cache_file, "r"))
            else:
                cache = {}
        else:
            cache_file = None
            cache = {}

        callback_manager = CallbackManager([])

        super().__init__(persist_dir=persist_dir,
                         embed_model=embed_model,
                         embed_batch_size=embed_model.embed_batch_size,
                         model_name=embed_model.model_name,
                         callback_manager=callback_manager,
                         cache=cache,
                         cache_file=cache_file,
                         **kwargs)

    def _get_query_embedding(self, query: str) -> Embedding:
        return self.embed_model._get_text_embeddings([query])[0]

    async def _aget_query_embedding(self, query: str) -> Embedding:
        return await self.embed_model._aget_query_embedding(query)

    def _get_text_embedding(self, text: str) -> Embedding:
        return self._get_text_embeddings([text])[0]

    async def _aget_text_embedding(self, text: str) -> Embedding:
        return await self._aget_text_embeddings([text])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[Embedding]:
        embeddings = []
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            cache_key = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if cache_key in self.cache:
                embeddings.append(self.cache[cache_key])
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        if uncached_texts:
            uncached_embeddings = self.embed_model._get_text_embeddings(uncached_texts)
            for text, embedding in zip(uncached_texts, uncached_embeddings):
                cache_key = hashlib.sha256(text.encode("utf-8")).hexdigest()
                self.cache[cache_key] = embedding

            for i, embedding in zip(uncached_indices, uncached_embeddings):
                embeddings.insert(i, embedding)

        return embeddings

    async def _aget_text_embeddings(self, texts: List[str]) -> List[Embedding]:
        embeddings = []
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            cache_key = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if cache_key in self.cache:
                embeddings.append(self.cache[cache_key])
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        if uncached_texts:
            uncached_embeddings = await self.embed_model._aget_text_embeddings(uncached_texts)
            for text, embedding in zip(uncached_texts, uncached_embeddings):
                cache_key = hashlib.sha256(text.encode("utf-8")).hexdigest()
                self.cache[cache_key] = embedding

            for i, embedding in zip(uncached_indices, uncached_embeddings):
                embeddings.insert(i, embedding)

        return embeddings

    def persist(self):
        if self.cache_file:
            json.dump(self.cache, open(self.cache_file, "w"))
