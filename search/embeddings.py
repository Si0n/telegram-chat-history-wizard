"""
OpenAI embedding service for semantic search.
"""
import asyncio
from typing import Optional
from openai import OpenAI, AsyncOpenAI

import config


class EmbeddingService:
    """Handle OpenAI embedding generation."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or config.OPENAI_API_KEY
        self.model = config.EMBEDDING_MODEL
        self.client = OpenAI(api_key=self.api_key)
        self.async_client = AsyncOpenAI(api_key=self.api_key)

    def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []

        # OpenAI supports batching natively
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )

        # Sort by index to maintain order
        embeddings = sorted(response.data, key=lambda x: x.index)
        return [e.embedding for e in embeddings]

    async def embed_text_async(self, text: str) -> list[float]:
        """Async embedding for a single text."""
        response = await self.async_client.embeddings.create(
            model=self.model,
            input=text
        )
        return response.data[0].embedding

    async def embed_batch_async(self, texts: list[str]) -> list[list[float]]:
        """Async batch embedding."""
        if not texts:
            return []

        response = await self.async_client.embeddings.create(
            model=self.model,
            input=texts
        )

        embeddings = sorted(response.data, key=lambda x: x.index)
        return [e.embedding for e in embeddings]

    async def embed_parallel_batches(
        self,
        all_texts: list[str],
        batch_size: int = 250,
        max_workers: int = 3
    ) -> list[list[float]]:
        """
        Process texts in parallel batches for maximum throughput.

        Args:
            all_texts: All texts to embed
            batch_size: Size of each batch (OpenAI limit is ~2048 per request)
            max_workers: Number of concurrent API requests

        Returns:
            All embeddings in order
        """
        if not all_texts:
            return []

        # Split into batches
        batches = []
        for i in range(0, len(all_texts), batch_size):
            batches.append(all_texts[i:i + batch_size])

        # Process batches with semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_workers)

        async def process_batch(batch_idx: int, batch: list[str]):
            async with semaphore:
                embeddings = await self.embed_batch_async(batch)
                return batch_idx, embeddings

        # Run all batches concurrently (limited by semaphore)
        tasks = [process_batch(i, batch) for i, batch in enumerate(batches)]
        results = await asyncio.gather(*tasks)

        # Sort by batch index and flatten
        results.sort(key=lambda x: x[0])
        all_embeddings = []
        for _, embeddings in results:
            all_embeddings.extend(embeddings)

        return all_embeddings

    def embed_parallel(
        self,
        texts: list[str],
        batch_size: int = None,
        max_workers: int = None
    ) -> list[list[float]]:
        """
        Sync wrapper for parallel batch embedding.
        Optimized for large-scale embedding jobs.
        """
        batch_size = batch_size or getattr(config, 'EMBEDDING_BATCH_SIZE', 250)
        max_workers = max_workers or getattr(config, 'EMBEDDING_WORKERS', 3)

        return asyncio.run(
            self.embed_parallel_batches(texts, batch_size, max_workers)
        )


class ChatService:
    """OpenAI chat completion for analysis."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or config.OPENAI_API_KEY
        self.model = config.CHAT_MODEL
        self.client = OpenAI(api_key=self.api_key)
        self.async_client = AsyncOpenAI(api_key=self.api_key)

    def complete(
        self,
        prompt: str,
        system: str = None,
        max_tokens: int = 1000
    ) -> str:
        """Generate chat completion."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.3
        )
        return response.choices[0].message.content

    async def complete_async(
        self,
        prompt: str,
        system: str = None,
        max_tokens: int = 1000
    ) -> str:
        """Async chat completion."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = await self.async_client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.3
        )
        return response.choices[0].message.content

    async def complete_stream_async(
        self,
        prompt: str,
        system: str = None,
        max_tokens: int = 1000
    ):
        """Async streaming chat completion. Yields content chunks."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = await self.async_client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.3,
            stream=True
        )
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
