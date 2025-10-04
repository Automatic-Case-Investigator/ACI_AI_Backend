from ACI_AI_Backend.objects.web_search.ollama_agents.keyword_extractor.keyword_extractor import (
    KeywordExtractor,
)
from ACI_AI_Backend.objects.web_search.ollama_agents.keyword_explainer.keyword_explainer import (
    KeywordExplainer,
)
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from ACI_AI_Backend.objects.redis_client import redis_client
from crawl4ai.models import CrawlResult
from urllib.parse import urlparse
from django.conf import settings
import asyncio
import http.client
import json
import re


class WebSearcher:
    """
    A utility class for extracting keywords from text, performing web searches, 
    and summarizing contextual explanations. 

    This class combines keyword extraction, web crawling, and information 
    summarization to provide meaningful context for given input text.
    """

    def __init__(
        self,
        max_search_results: int = 10,
    ):
        """Initialize the WebSearcher with configuration options.

        Args:
            max_search_results (int, optional): Maximum number of URLs retrieved per search term.
                Defaults to 10.
        """
        self.max_search_results = max_search_results

    def get_urls(self, search_term: str) -> list[str]:
        """Retrieve URLs for a given search term.

        Args:
            search_term (str): The term to search for.

        Returns:
            list[str]: A list of URLs matching the search term.
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
        """Crawl a list of web pages asynchronously and filter their content.

        Args:
            urls (list[str]): List of URLs to crawl.

        Returns:
            list[CrawlResult]: List of CrawlResult objects containing extracted content.
        """
        md_generator = DefaultMarkdownGenerator()

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
        """Normalize a URL string into a filesystem-safe identifier.

        Args:
            url (str): The original URL.

        Returns:
            str: A normalized string suitable for use as an ID.
        """
        normalized_url = (
            url.replace("https://", "")
            .replace("www.", "")
            .replace("/", "_")
            .replace("-", "_")
            .replace(".", "_")
        )
        return normalized_url

    def extract_keyword(self, text: str) -> set[str]:
        """
        Extract relevant keywords (CVEs, MITRE IDs, compliance, attack names, IPs, etc.) from text.

        Args:
            text (str): The input text to analyze.

        Returns:
            set[str]: A set of extracted keywords.
        """
        extractor = KeywordExtractor(settings.OLLAMA_URL)
        lines = extractor.invoke(text).split("\n")
        keywords = set()

        for line in lines:
            if len(line) > 0 or len(line) <= 50:
                keywords.add(line)

        return keywords

    async def research(self, text: str) -> dict[str, str]:
        """Conduct research on a given text by extracting keywords,
        crawling related web content, and storing results in the vector database.

        Args:
            text (str): Input text to research.

        Returns:
            dict[str, str]: A dictionary mapping keywords to relevant contextual documents.
        """
        keywords = self.extract_keyword(text)
        explainer = KeywordExplainer(settings.OLLAMA_URL)

        keyword_knowledge = {}

        async def process_keyword(keyword: str):
            urls = self.get_urls(keyword)

            if len(urls) == 0:
                return ""

            crawl_results = await self.crawl_webpages(urls)
            crawl_results_text = ""
            crawl_result_url = []
            for crawl_result in crawl_results:
                if not crawl_result.markdown:
                    continue

                crawl_results_text += crawl_result.markdown + "\n"
                crawl_result_url.append(crawl_result.url)

            explanation = explainer.invoke(keyword, crawl_results_text)
            return explanation

        for keyword in keywords:
            explanation = redis_client.get(keyword)
            if explanation is not None and len(explanation) > 0:
                keyword_knowledge[keyword] = explanation
                continue

            explanation = await process_keyword(keyword)
            keyword_knowledge[keyword] = explanation
            redis_client.set(keyword, explanation, ex=settings.SEARCH_CACHE_EXPIRY_TIME)

        print(keyword_knowledge)
        return keyword_knowledge

    def run(self, text: str) -> dict[str, str]:
        """Run the full research pipeline synchronously.

        Args:
            text (str): Input text to research.

        Returns:
            dict[str, str]: A dictionary mapping keywords to relevant contextual documents.
        """
        return asyncio.run(self.research(text))
