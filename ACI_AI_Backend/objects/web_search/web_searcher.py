from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from langchain_text_splitters import RecursiveCharacterTextSplitter
from crawl4ai.content_filter_strategy import BM25ContentFilter
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from urllib.parse import urlparse, urlencode, parse_qs
from crawl4ai.models import CrawlResult
from chromadb.config import Settings
from django.conf import settings
from duckpy import Client
import chromadb
import tempfile
import asyncio
import spacy
import os
import re


class WebSearcher:
    """A class for searching, crawling, and processing web content into a vector database.

    The WebSearcher integrates DuckDuckGo search, web crawling, content filtering,
    and text embedding into a ChromaDB vector database. It can be used to retrieve,
    filter, and contextualize information about keywords, CVEs, or MITRE techniques.
    """

    def __init__(
        self,
        region: str = "us-en",
        max_search_results: int = 10,
    ):
        """Initialize the WebSearcher with configuration options.

        Args:
            region (str, optional): Region for duckduckgo search
                Defaults to "us-en"
            max_search_results (int, optional): Maximum number of URLs retrieved per search term.
                Defaults to 10.
        """
        self.region = region
        self.max_search_results = max_search_results

        self.chroma_client = chromadb.HttpClient(
            host="chroma", port=8000, settings=Settings(anonymized_telemetry=False)
        )
        self.chroma_collection = self.chroma_client.get_or_create_collection(
            "web_search_results"
        )
        spacy.prefer_gpu()
        self.nlp = spacy.load("en_core_web_sm")

    def get_urls(self, search_term: str) -> list[str]:
        """Retrieve URLs for a given search term using DuckDuckGo.

        Args:
            search_term (str): The term to search for.

        Returns:
            list[str]: A list of URLs matching the search term.
        """         
        client = Client()
        results = client.search(search_term)
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
        self, urls: list[str], search_term: str
    ) -> list[CrawlResult]:
        """Crawl a list of web pages asynchronously and filter their content.

        Args:
            urls (list[str]): List of URLs to crawl.
            search_term (str): Search term used to guide content filtering.

        Returns:
            list[CrawlResult]: List of CrawlResult objects containing extracted content.
        """
        bm25_filter = BM25ContentFilter(user_query=search_term, bm25_threshold=0.5)
        md_generator = DefaultMarkdownGenerator(content_filter=bm25_filter)
        
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

    def add_to_vector_database(self, search_results: list[CrawlResult]) -> None:
        """Add crawled documents into the ChromaDB vector database.

        Args:
            search_results (list[CrawlResult]): List of crawled results to insert.
        """
        for result in search_results:
            documents, metadatas, ids = [], [], []

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=400,
                chunk_overlap=100,
                separators=["\n\n", "\n", ".", "?", "!", " ", ""],
            )
            print(result.markdown)
            if not result.markdown:
                continue

            markdown_result = result.markdown

            temp_file = tempfile.NamedTemporaryFile("w", suffix=".md", delete=False)
            temp_file.write(markdown_result)
            temp_file.flush()

            loader = UnstructuredMarkdownLoader(temp_file.name, mode="single")
            docs = loader.load()
            all_splits = text_splitter.split_documents(docs)

            os.unlink(temp_file.name)

            normalized_url = self.normalize_url(result.url)

            if all_splits:
                for idx, split in enumerate(all_splits):
                    documents.append(split.page_content)
                    metadatas.append({"source": result.url})
                    ids.append(f"{normalized_url}_{idx}")

                print("Upsert collection: ", id(self.chroma_collection))
                self.chroma_collection.upsert(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids,
                )
                
    def extract_keyword(self, text: str) -> set[str]:
        """
        Extract relevant keywords (CVEs, MITRE IDs, compliance, attack names, IPs, etc.) from text.

        Args:
            text (str): The input text to analyze.

        Returns:
            set[str]: A set of extracted keywords.
        """
        keywords = set()

        # --- CVEs ---
        cve_pattern = re.compile(r"\bCVE[-_]?(\d{4})[-_]?(\d+)\b", re.IGNORECASE)
        for year, num in cve_pattern.findall(text):
            keywords.add(f"CVE-{year}-{num}")

        # --- MITRE Technique IDs ---
        mitre_pattern = re.compile(r"\bT[-_]?(\d{4})(?:\.?(\d+))?\b", re.IGNORECASE)
        for major, minor in mitre_pattern.findall(text):
            tid = f"T{major}{'.' + minor if minor else ''}"
            keywords.add(tid)

        # --- Compliance Standards ---
        compliance_pattern = re.compile(
            r"\b(?:PCI[-\s]?DSS|GDPR|NIST|TSC)(?:[.\s:_-]?\w+)*\b", re.IGNORECASE
        )
        for comp in compliance_pattern.findall(text):
            keywords.add(comp.upper())

        # --- IP addresses ---
        ip_pattern = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
        for ip in ip_pattern.findall(text):
            keywords.add(ip)

        # --- Filenames / Paths ---
        file_pattern = re.compile(r"(?:/[\w._-]+)+")
        for path in file_pattern.findall(text):
            keywords.add(path)

        # --- NLP-based entity extraction ---
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ in {"ORG", "PRODUCT", "EVENT"}:
                keywords.add(ent.text)

        # --- Noun phrase candidates ---
        for chunk in doc.noun_chunks:
            token = chunk.text.strip()
            if len(token) > 2 and not token.islower():
                keywords.add(token)

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
        keyword_knowledge = {}

        async def process_keyword(keyword: str):
            urls = self.get_urls(keyword)
            crawl_results = await self.crawl_webpages(urls, keyword)
            print(crawl_results)
            
            await asyncio.to_thread(self.add_to_vector_database, crawl_results)
            
            query_results = await asyncio.to_thread(
                self.chroma_collection.query, query_texts=[keyword], n_results=10
            )
            
            context = query_results.get("documents", [[]])[0]
            return keyword, context

        results = await asyncio.gather(*(process_keyword(k) for k in keywords))

        for keyword, context in results:
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


if __name__ == "__main__":
    searcher = WebSearcher()
    text = (
        "### Timestamp\n| key | val |\n| ------ | ------ |\n| **timestamp** | 2025-04-19T02:42:15.087+0000 |\n### Rule\n| key | val |\n| ------ | ------ |\n| **rule.level** | 15 |\n| **rule.description** | Shellshock attack detected |\n| **rule.id** | 31168 |\n| **rule.mitre.id** | ['T1068', 'T1190'] |\n| **rule.mitre.tactic** | ['Privilege Escalation', 'Initial Access'] |\n| **rule.mitre.technique** | ['Exploitation for Privilege Escalation', 'Exploit Public-Facing Application'] |\n| **rule.info** | CVE-2014-6271https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2014-6271 |\n| **rule.firedtimes** | 1 |\n| **rule.mail** | True |\n| **rule.groups** | ['web', 'accesslog', 'attack'] |\n| **rule.pci_dss** | ['11.4'] |\n| **rule.gdpr** | ['IV_35.7.d'] |\n| **rule.nist_800_53** | ['SI.4'] |\n| **rule.tsc** | ['CC6.1', 'CC6.8', 'CC7.2', 'CC7.3'] |\n### Agent\n| key | val |\n| ------ | ------ |\n| **agent.id** | 001 |\n| **agent.name** | kali |\n| **agent.ip** | 10.0.2.15 |\n### Manager\n| key | val |\n| ------ | ------ |\n| **manager.name** | wazuh.manager |\n### Id\n| key | val |\n| ------ | ------ |\n| **id** | 1745030535.3859659 |\n### Full_log\n| key | val |\n| ------ | ------ |\n| **full_log** | 10.0.2.5 - - [19/Apr/2025:10:42:13 +0800] \"GET / HTTP/1.1\" 200 10926 \"-\" \"() { :; }; /bin/cat /etc/passwd\" |\n### Decoder\n| key | val |\n| ------ | ------ |\n| **decoder.name** | web-accesslog |\n### Data\n| key | val |\n| ------ | ------ |\n| **data.protocol** | GET |\n| **data.srcip** | 10.0.2.5 |\n| **data.id** | 200 |\n| **data.url** | / |\n### Location\n| key | val |\n| ------ | ------ |\n| **location** | /var/log/apache2/access.log |\n)"
    )
    print(searcher.run(text))
