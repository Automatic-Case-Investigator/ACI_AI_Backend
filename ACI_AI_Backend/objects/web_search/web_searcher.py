from ACI_AI_Backend.objects.web_search.ollama_agents.keyword_extractor.keyword_extractor import KeywordExtractor
from ACI_AI_Backend.objects.web_search.ollama_agents.keyword_explainer.keyword_explainer import KeywordExplainer
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.models import CrawlResult
from urllib.parse import urlparse
from django.conf import settings
from duckpy import Client
import asyncio
import re


class WebSearcher:
    """A class for searching, crawling, and processing web content into a vector database.

    The WebSearcher integrates DuckDuckGo search, web crawling, content filtering,
    and text embedding into a ChromaDB vector database. It can be used to retrieve,
    filter, and contextualize information about keywords, CVEs, or MITRE techniques.
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
        self.web_client = Client()
        self.max_search_results = max_search_results

    def get_urls(self, search_term: str) -> list[str]:
        """Retrieve URLs for a given search term using DuckDuckGo.

        Args:
            search_term (str): The term to search for.

        Returns:
            list[str]: A list of URLs matching the search term.
        """         
        results = self.web_client.search(search_term)[:self.max_search_results]
        urls = []
        
        url_pattern = re.compile(
            r"https?://[^\s/$.?#].[^\s]*", re.IGNORECASE
        )

        # regex to remove "&rut=..." or "?rut=..." at the end
        rut_pattern = re.compile(r"([&?])rut=[^&]+(&)?")

        for result in results:
            raw_url = result.get("url", "")
            url_search = url_pattern.search(raw_url)
            if not url_search:
                continue

            cleaned_url = raw_url[url_search.start():url_search.end()]

            cleaned_url = rut_pattern.sub(
                lambda m: m.group(1) if m.group(2) else "", cleaned_url
            )

            parsed = urlparse(cleaned_url)
            if parsed.scheme in {"http", "https"} and parsed.netloc:
                urls.append(cleaned_url)

        return urls


    async def crawl_webpages(
        self, urls: list[str]
    ) -> list[CrawlResult]:
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
            context = await process_keyword(keyword)
            keyword_knowledge[keyword] = context

        return keyword_knowledge

    def run(self, text: str) -> dict[str, str]:
        """Run the full research pipeline synchronously.

        Args:
            text (str): Input text to research.

        Returns:
            dict[str, str]: A dictionary mapping keywords to relevant contextual documents.
        """
        return asyncio.run(self.research(text))