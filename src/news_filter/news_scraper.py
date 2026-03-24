"""
Gold News Headline Scraper
Fetches latest gold-related headlines for sentiment analysis.
Uses multiple free sources for reliability.
"""

import warnings
import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from datetime import datetime, timezone
from typing import List
from src.utils.logger import setup_logger

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

logger = setup_logger("news_scraper")


class GoldNewsScraper:
    """Fetches gold-related news headlines from free sources."""

    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    def fetch_all_headlines(self) -> List[str]:
        """Fetch gold headlines from all available sources."""
        headlines = []

        headlines.extend(self._fetch_google_news_gold())

        if len(headlines) < 5:
            headlines.extend(self._fetch_reuters_gold_rss())

        if len(headlines) < 5:
            headlines.extend(self._fetch_kitco_headlines())

        # Remove duplicates, bad encoding, and short strings
        seen = set()
        unique = []
        for h in headlines:
            # Drop headlines with garbled/replacement characters
            if "\ufffd" in h or "?" in h[:3]:
                continue
            h_clean = h.strip()
            h_lower = h_clean.lower()
            if h_lower not in seen and len(h_clean) > 20:
                seen.add(h_lower)
                unique.append(h_clean)

        logger.info(f"  Fetched {len(unique)} unique gold headlines")
        return unique[:20]

    def _fetch_google_news_gold(self) -> List[str]:
        """Fetch gold news from Google News RSS (XML feed)."""
        headlines = []
        try:
            url = "https://news.google.com/rss/search?q=gold+price+XAUUSD&hl=en-US&gl=US&ceid=US:en"
            resp = requests.get(url, headers=self.HEADERS, timeout=10)

            if resp.status_code == 200:
                soup = BeautifulSoup(resp.content, "lxml-xml")
                for item in soup.find_all("item")[:15]:
                    title = item.find("title")
                    if title:
                        text = title.get_text(strip=True)
                        if text and "google news" not in text.lower():
                            headlines.append(text)

                logger.info(f"  Google News: {len(headlines)} headlines")

        except Exception as e:
            logger.warning(f"  Google News failed: {e}")

        return headlines

    def _fetch_reuters_gold_rss(self) -> List[str]:
        """Fetch from Reuters commodities RSS feed."""
        headlines = []
        try:
            url = "https://feeds.reuters.com/reuters/businessNews"
            resp = requests.get(url, headers=self.HEADERS, timeout=10)

            if resp.status_code == 200:
                soup = BeautifulSoup(resp.content, "lxml-xml")
                for item in soup.find_all("item")[:20]:
                    title = item.find("title")
                    if title:
                        text = title.get_text(strip=True)
                        # Only keep gold-relevant headlines
                        if any(kw in text.lower() for kw in ["gold", "xau", "bullion", "precious"]):
                            headlines.append(text)

                logger.info(f"  Reuters: {len(headlines)} gold headlines")

        except Exception as e:
            logger.warning(f"  Reuters failed: {e}")

        return headlines

    def _fetch_kitco_headlines(self) -> List[str]:
        """Fetch from Kitco (major gold news source)."""
        headlines = []
        try:
            url = "https://www.kitco.com/rss/index.html"
            resp = requests.get(url, headers=self.HEADERS, timeout=10)

            if resp.status_code == 200:
                soup = BeautifulSoup(resp.content, "lxml-xml")
                for item in soup.find_all("item")[:15]:
                    title = item.find("title")
                    if title:
                        text = title.get_text(strip=True)
                        if text and len(text) > 15:
                            headlines.append(text)

                logger.info(f"  Kitco: {len(headlines)} headlines")

        except Exception as e:
            logger.warning(f"  Kitco failed: {e}")

        return headlines
