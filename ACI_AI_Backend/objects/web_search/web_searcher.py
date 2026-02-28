from datetime import datetime, timezone
import asyncio
import http.client
import json
import re
import time

from ACI_AI_Backend.objects.chromadb_client import chromadb_client
from ACI_AI_Backend.objects.web_search.keyword_explainer.keyword_explainer import KeywordExplainer
from ACI_AI_Backend.objects.web_search.keyword_extractor.keyword_extractor import KeywordExtractor
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.models import CrawlResult
from django.conf import settings


class WebSearcher:
    """Search the web, crawl result pages, and build query explanations with caching.

    Workflow:
    1. Extract one or more focused queries from input text.
    2. For each query, try to reuse a non-expired cached explanation from ChromaDB.
    3. On cache miss, fetch result URLs from Serper, crawl pages, and summarize.
    4. Store fresh explanations back into ChromaDB with creation metadata.
    """

    description = "Given one or more search queries, perform web searches to find relevant pages, " "extract the most important facts, and return a concise, well-structured summary."

    def __init__(self, max_search_results: int = 10):
        """Initialize search settings.

        Args:
            max_search_results: Maximum number of search results to process per query.
                (Stored for configuration/use by callers and future tuning.)
        """
        self.max_search_results = max_search_results
        self.ttl_seconds = int(settings.SEARCH_CACHE_EXPIRY_TIME)

    @classmethod
    def context_to_str(web_search_context: dict):
        parts = []
        for keyword, data in web_search_context.items():
            explanation = data.get("explanation", "")
            if explanation:
                parts.append(f"{keyword}:\n```\n{explanation}\n```\n\n")
            
        return "".join(parts)

    def get_urls(self, search_term: str) -> list[str]:
        """Fetch organic search result URLs for a query using the Serper API.

        Args:
            search_term: Query text to send to Serper.

        Returns:
            A list of cleaned HTTP(S) URLs extracted from organic results.
        """
        connection = http.client.HTTPSConnection("google.serper.dev")
        payload = json.dumps({"q": search_term})
        headers = {
            "X-API-KEY": settings.SERPER_API_KEY,
            "Content-Type": "application/json",
        }
        connection.request("POST", "/search", payload, headers)
        res = connection.getresponse()
        results = json.loads(res.read().decode()).get("organic", [])
        urls = []

        url_pattern = re.compile(r"https?://[^\s/$.?#].[^\s]*", re.IGNORECASE)
        for result in results:
            raw_url = result.get("link", "")
            url_search = url_pattern.search(raw_url)
            if not url_search:
                continue
            cleaned_url = raw_url[url_search.start() : url_search.end()]
            urls.append(cleaned_url)
        return urls

    async def crawl_webpages(self, urls: list[str]) -> list[CrawlResult]:
        """Crawl a list of URLs and return text-focused markdown results.

        The crawler is configured to reduce noise by excluding common layout/content
        elements and returning text-only markdown.

        Args:
            urls: Target pages to crawl.

        Returns:
            A list of `CrawlResult` items from crawl4ai.
        """
        md_generator = DefaultMarkdownGenerator(
            options={
                "ignore_links": True,
                "ignore_images": True,
            }
        )
        crawler_config = CrawlerRunConfig(
            verbose=True,
            markdown_generator=md_generator,
            excluded_tags=["nav", "footer", "header", "form", "img", "a"],
            only_text=True,
            exclude_social_media_links=True,
            keep_data_attributes=False,
            cache_mode=CacheMode.BYPASS,
            remove_overlay_elements=True,
            page_timeout=20000,
        )
        browser_config = BrowserConfig(headless=True, text_mode=True, light_mode=True)

        async with AsyncWebCrawler(config=browser_config) as crawler:
            results = await crawler.arun_many(urls, config=crawler_config)
            return results

    def normalize_url(self, url: str) -> str:
        """Normalize a URL into a filesystem/cache-safe identifier string."""
        return url.replace("https://", "").replace("www.", "").replace("/", "_").replace("-", "_").replace(".", "_")

    async def research(self, text: str) -> dict[str, dict[str, str]]:
        """Generate per-query explanations for an input prompt.

        This method extracts keyword queries from the input text, processes each query
        concurrently, and returns a dictionary keyed by query containing:
        - explanation: synthesized explanation text
        - sources: URLs used for that explanation (cached or newly crawled)

        Args:
            text: Input prompt/topic to research.

        Returns:
            A mapping from query to explanation/source metadata.
        """
        extractor = KeywordExtractor()
        explainer = KeywordExplainer()
        query_knowledge: dict[str, dict[str, str]] = {}
        lock = asyncio.Lock()
        collection = chromadb_client.get_or_create_collection("web_search_results")

        async def search_with_query(query: str):
            """Resolve one query using cache-first lookup, then crawl+explain fallback."""
            # Try to get cached explanation from Chroma
            now_ts = time.time()
            db_query_result = collection.query(
                query_texts=[query],
                n_results=10,
            )

            # Filter out expired documents
            docs = []
            dists = []
            valid_indices = []
            print(db_query_result)
            for idx, meta in enumerate(db_query_result["metadatas"][0]):
                created_at = meta.get("created_at", 0)
                if now_ts - created_at <= self.ttl_seconds:
                    valid_indices.append(idx)
                else:
                    collection.delete(ids=db_query_result["ids"][0][idx])

            if valid_indices:
                docs = [db_query_result["documents"][0][i] for i in valid_indices]
                dists = [db_query_result["distances"][0][i] for i in valid_indices]

                min_dist = min(dists)
                min_idx = dists.index(min_dist)

                print(f"Query: {query}, Document: {docs[min_idx]}", min_dist)
                explanation = docs[min_idx]
                async with lock:
                    query_knowledge[query] = {
                        "explanation": explanation,
                        "sources": db_query_result["metadatas"][0][valid_indices[min_idx]].get("sources", []),
                    }
                return

            # Cache miss: find URLs
            urls: list[str] = self.get_urls(query)
            if not urls:
                return ""

            # Crawl all URLs concurrently
            crawl_results: list[CrawlResult] = await self.crawl_webpages(urls)
            crawl_results_text: str = "\n".join(result.markdown for result in crawl_results if result.markdown)

            # Generate explanation
            loop = asyncio.get_running_loop()
            explanation: str = await loop.run_in_executor(None, explainer.invoke, query, crawl_results_text)

            # Save explanation in Chroma
            async with lock:
                query_knowledge[query] = {
                    "explanation": explanation,
                    "sources": urls,
                }
                collection.add(documents=[explanation], ids=[f"{query}:{hash(explanation)}"], metadatas=[{"keyword": query, "created_at": now_ts, "sources": urls}])

        # Extract keywords and run queries concurrently
        queries: list[str] = extractor.invoke(text).split(",")
        print("Queries: ", queries)
        await asyncio.gather(*(search_with_query(k) for k in queries))

        return query_knowledge

    def run(self, text: str) -> dict[str, dict[str, str]]:
        """Synchronous entrypoint that runs `research` in its own event loop."""
        print("Running web search")
        knowledge = asyncio.run(self.research(text))
        print(knowledge)
        return knowledge
