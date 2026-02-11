"""
CAM Web Tool

Web searching and page fetching with soft dependencies — works with
or without `requests` and `beautifulsoup4`. Falls back to urllib and
regex when those libraries are missing.

Provides two main operations:
  - search(query) — DuckDuckGo HTML search for links + snippets
  - fetch_page(url) — HTTP GET with text extraction

Usage:
    from tools.web import WebTool

    web = WebTool()
    results = web.search("motorcycle diagnostic scanner")
    page = web.fetch_page(results[0].url)
    print(page.text[:500])
"""

import html as html_module
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode, unquote

# ---------------------------------------------------------------------------
# Soft dependencies — degrade gracefully if missing
# ---------------------------------------------------------------------------

_HAS_REQUESTS = False
_HAS_BS4 = False

try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    _requests = None  # type: ignore[assignment]

try:
    from bs4 import BeautifulSoup as _BeautifulSoup
    _HAS_BS4 = True
except ImportError:
    _BeautifulSoup = None  # type: ignore[assignment]


logger = logging.getLogger("cam.web_tool")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """A single search result from DuckDuckGo."""
    title: str
    url: str
    snippet: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"title": self.title, "url": self.url, "snippet": self.snippet}


@dataclass
class FetchedPage:
    """A fetched and text-extracted web page."""
    url: str
    title: str = ""
    text: str = ""
    status_code: int = 0
    fetch_time_ms: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "text": self.text[:500] if self.text else "",
            "status_code": self.status_code,
            "fetch_time_ms": round(self.fetch_time_ms, 1),
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# WebTool
# ---------------------------------------------------------------------------

DDG_URL = "https://html.duckduckgo.com/html/"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


class WebTool:
    """Web searching and page fetching tool.

    Searches DuckDuckGo via HTML scraping (no API key needed).
    Fetches pages and extracts readable text. Falls back to stdlib
    when requests/beautifulsoup4 are not installed.

    Args:
        timeout:     HTTP request timeout in seconds.
        max_retries: Number of retries on timeout or 5xx errors.
        user_agent:  Custom User-Agent header string.
    """

    def __init__(
        self,
        timeout: int = 10,
        max_retries: int = 2,
        user_agent: str | None = None,
    ):
        self.timeout = timeout
        self.max_retries = max_retries
        self.user_agent = user_agent or DEFAULT_USER_AGENT
        logger.info(
            "WebTool initialized (requests=%s, bs4=%s)",
            _HAS_REQUESTS, _HAS_BS4,
        )

    # -------------------------------------------------------------------
    # Search
    # -------------------------------------------------------------------

    def search(self, query: str, max_results: int = 8) -> list[SearchResult]:
        """Search DuckDuckGo and return results.

        Uses POST to html.duckduckgo.com/html/ and parses the
        result page for links and snippets.

        Args:
            query:       The search query string.
            max_results: Maximum number of results to return.

        Returns:
            List of SearchResult objects.
        """
        try:
            body = urlencode({"q": query}).encode("utf-8")
            html_text, status = self._fetch_raw(
                DDG_URL,
                method="POST",
                body=body,
                content_type="application/x-www-form-urlencoded",
            )
            if status != 200 or not html_text:
                logger.warning("DDG search returned status %d", status)
                return []

            results = self._parse_ddg_results(html_text, max_results)
            logger.info(
                "DDG search '%s': %d results", query[:60], len(results),
            )
            return results

        except Exception as e:
            logger.warning("DDG search failed for '%s': %s", query[:60], e)
            return []

    def _parse_ddg_results(self, html_text: str, max_results: int) -> list[SearchResult]:
        """Parse DuckDuckGo HTML results page."""
        results: list[SearchResult] = []

        if _HAS_BS4:
            soup = _BeautifulSoup(html_text, "html.parser")
            # DuckDuckGo HTML result links
            for link in soup.select("a.result__a")[:max_results]:
                title = link.get_text(strip=True)
                href = link.get("href", "")
                # DDG wraps URLs in a redirect — extract the actual URL
                url = self._extract_ddg_url(href)
                if not url or not title:
                    continue

                # Find the snippet (sibling element)
                snippet = ""
                snippet_el = link.find_parent("div", class_="result")
                if snippet_el:
                    snip = snippet_el.select_one("a.result__snippet")
                    if snip:
                        snippet = snip.get_text(strip=True)

                results.append(SearchResult(title=title, url=url, snippet=snippet))
        else:
            # Regex fallback
            link_pattern = re.compile(
                r'<a\s+[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
                re.DOTALL,
            )
            snippet_pattern = re.compile(
                r'<a\s+[^>]*class="result__snippet"[^>]*>(.*?)</a>',
                re.DOTALL,
            )

            links = link_pattern.findall(html_text)
            snippets = snippet_pattern.findall(html_text)

            for i, (href, raw_title) in enumerate(links[:max_results]):
                title = self._strip_html_tags(raw_title).strip()
                url = self._extract_ddg_url(href)
                if not url or not title:
                    continue

                snippet = ""
                if i < len(snippets):
                    snippet = self._strip_html_tags(snippets[i]).strip()

                results.append(SearchResult(title=title, url=url, snippet=snippet))

        return results

    @staticmethod
    def _extract_ddg_url(href: str) -> str:
        """Extract the real URL from DDG's redirect wrapper."""
        # DDG wraps URLs like: //duckduckgo.com/l/?uddg=https%3A%2F%2F...&rut=...
        if "uddg=" in href:
            match = re.search(r'uddg=([^&]+)', href)
            if match:
                return unquote(match.group(1))
        # Direct URL
        if href.startswith("http"):
            return href
        if href.startswith("//"):
            return "https:" + href
        return ""

    # -------------------------------------------------------------------
    # Fetch page
    # -------------------------------------------------------------------

    def fetch_page(self, url: str, max_text: int = 3000) -> FetchedPage:
        """Fetch a web page and extract readable text.

        Strips navigation, scripts, styles, and other non-content
        elements. Truncates extracted text to max_text characters.

        Args:
            url:      The URL to fetch.
            max_text: Maximum characters of extracted text.

        Returns:
            FetchedPage with extracted text and metadata.
        """
        start = time.monotonic()
        try:
            html_text, status = self._fetch_raw(url)
            elapsed_ms = (time.monotonic() - start) * 1000

            if status >= 400:
                return FetchedPage(
                    url=url, status_code=status, fetch_time_ms=elapsed_ms,
                    error=f"HTTP {status}",
                )

            if _HAS_BS4:
                title, text = self._extract_text_bs4(html_text)
            else:
                title, text = self._extract_text_regex(html_text)

            text = text[:max_text]

            return FetchedPage(
                url=url, title=title, text=text,
                status_code=status, fetch_time_ms=elapsed_ms,
            )

        except Exception as e:
            elapsed_ms = (time.monotonic() - start) * 1000
            logger.warning("Fetch failed for %s: %s", url, e)
            return FetchedPage(
                url=url, fetch_time_ms=elapsed_ms, error=str(e),
            )

    # -------------------------------------------------------------------
    # Raw HTTP
    # -------------------------------------------------------------------

    def _fetch_raw(
        self,
        url: str,
        method: str = "GET",
        body: bytes | None = None,
        content_type: str | None = None,
    ) -> tuple[str, int]:
        """Fetch raw HTML from a URL with retry logic.

        Uses requests library if available, falls back to urllib.

        Args:
            url:          The URL to fetch.
            method:       HTTP method (GET or POST).
            body:         Request body bytes (for POST).
            content_type: Content-Type header value.

        Returns:
            Tuple of (html_text, status_code).
        """
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                if _HAS_REQUESTS:
                    return self._fetch_with_requests(url, method, body, content_type)
                else:
                    return self._fetch_with_urllib(url, method, body, content_type)

            except Exception as e:
                last_error = e
                # Don't retry on 4xx — those won't change
                if "4" == str(e)[:1]:
                    raise
                if attempt < self.max_retries:
                    time.sleep(0.5 * attempt)
                    logger.debug("Retry %d/%d for %s", attempt, self.max_retries, url)

        raise last_error or RuntimeError(f"Failed to fetch {url}")

    def _fetch_with_requests(
        self, url: str, method: str, body: bytes | None, content_type: str | None,
    ) -> tuple[str, int]:
        """Fetch using the requests library."""
        headers = {"User-Agent": self.user_agent}
        if content_type:
            headers["Content-Type"] = content_type

        if method == "POST":
            resp = _requests.post(url, data=body, headers=headers, timeout=self.timeout)
        else:
            resp = _requests.get(url, headers=headers, timeout=self.timeout)

        return resp.text, resp.status_code

    def _fetch_with_urllib(
        self, url: str, method: str, body: bytes | None, content_type: str | None,
    ) -> tuple[str, int]:
        """Fetch using urllib (stdlib fallback)."""
        import urllib.request
        import urllib.error

        headers = {"User-Agent": self.user_agent}
        if content_type:
            headers["Content-Type"] = content_type

        req = urllib.request.Request(url, data=body, headers=headers, method=method)

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                charset = resp.headers.get_content_charset() or "utf-8"
                return resp.read().decode(charset, errors="replace"), resp.status
        except urllib.error.HTTPError as e:
            return "", e.code

    # -------------------------------------------------------------------
    # Text extraction
    # -------------------------------------------------------------------

    def _extract_text_bs4(self, html_text: str) -> tuple[str, str]:
        """Extract readable text using BeautifulSoup.

        Strips script, style, nav, footer, header elements.

        Returns:
            Tuple of (title, cleaned_text).
        """
        soup = _BeautifulSoup(html_text, "html.parser")

        # Extract title
        title = ""
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text(strip=True)

        # Remove non-content elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)

        # Collapse multiple blank lines
        text = re.sub(r'\n{3,}', '\n\n', text)

        return title, text.strip()

    def _extract_text_regex(self, html_text: str) -> tuple[str, str]:
        """Extract readable text using regex (fallback when bs4 missing).

        Returns:
            Tuple of (title, cleaned_text).
        """
        # Extract title
        title = ""
        title_match = re.search(r'<title[^>]*>(.*?)</title>', html_text, re.DOTALL | re.IGNORECASE)
        if title_match:
            title = self._strip_html_tags(title_match.group(1)).strip()

        # Remove script and style blocks
        text = re.sub(r'<script[^>]*>.*?</script>', '', html_text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<nav[^>]*>.*?</nav>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<footer[^>]*>.*?</footer>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<header[^>]*>.*?</header>', '', text, flags=re.DOTALL | re.IGNORECASE)

        # Strip remaining HTML tags
        text = self._strip_html_tags(text)

        # Decode HTML entities
        text = html_module.unescape(text)

        # Collapse whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]{2,}', ' ', text)

        return title, text.strip()

    @staticmethod
    def _strip_html_tags(text: str) -> str:
        """Remove HTML tags from a string."""
        return re.sub(r'<[^>]+>', '', text)

    # -------------------------------------------------------------------
    # Status
    # -------------------------------------------------------------------

    def get_status(self) -> dict:
        """Report dependency availability and config."""
        return {
            "has_requests": _HAS_REQUESTS,
            "has_bs4": _HAS_BS4,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }


# ---------------------------------------------------------------------------
# Direct execution — quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    web = WebTool()
    print(f"Status: {web.get_status()}\n")

    # Test search
    print("--- DuckDuckGo Search ---")
    results = web.search("motorcycle diagnostic scanner", max_results=5)
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r.title}")
        print(f"     {r.url}")
        if r.snippet:
            print(f"     {r.snippet[:100]}")
        print()

    # Test page fetch
    if results:
        print("--- Page Fetch ---")
        page = web.fetch_page(results[0].url, max_text=500)
        print(f"  Title: {page.title}")
        print(f"  Status: {page.status_code}")
        print(f"  Time: {page.fetch_time_ms:.0f}ms")
        print(f"  Text: {page.text[:200]}...")
    else:
        print("No search results to test page fetch.")
