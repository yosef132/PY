"""
Gold News Headline Scraper
Fetches latest gold-related headlines for sentiment analysis.
Uses multiple free sources for reliability.
"""

import requests
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from typing import List
from src.utils.logger import setup_logger

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

        # Try multiple sources for reliability
        headlines.extend(self._fetch_google_news_gold())

        if len(headlines) < 3:
            headlines.extend(self._fetch_kitco_headlines())

        # Remove duplicates while preserving order
        seen = set()
        unique = []
        for h in headlines:
            h_lower = h.lower().strip()
            if h_lower not in seen and len(h) > 15:
                seen.add(h_lower)
                unique.append(h)

        logger.info(f"  Fetched {len(unique)} unique gold headlines")
        return unique[:20]  # Max 20 headlines

    def _fetch_google_news_gold(self) -> List[str]:
        """Fetch gold news from Google News RSS."""
        headlines = []
        try:
            url = "https://news.google.com/rss/search?q=gold+price+XAUUSD&hl=en-US&gl=US&ceid=US:en"
            resp = requests.get(url, headers=self.HEADERS, timeout=10)

            if resp.status_code == 200:
                soup = BeautifulSoup(resp.content, "html.parser")
                items = soup.find_all("item") or soup.find_all("title")

                for item in items[:15]:
                    title = item.find("title")
                    if title:
                        text = title.get_text(strip=True)
                        if text and "google news" not in text.lower():
                            headlines.append(text)

                logger.info(f"  Google News: {len(headlines)} headlines")

        except Exception as e:
            logger.warning(f"  Google News failed: {e}")

        return headlines

    def _fetch_kitco_headlines(self) -> List[str]:
        """Fetch from Kitco (major gold news source)."""
        headlines = []
        try:
            url = "https://www.kitco.com/news/gold/"
            resp = requests.get(url, headers=self.HEADERS, timeout=10)

            if resp.status_code == 200:
                soup = BeautifulSoup(resp.content, "html.parser")

                # Look for headline elements
                for tag in soup.find_all(["h2", "h3", "h4"], limit=15):
                    text = tag.get_text(strip=True)
                    if text and len(text) > 15:
                        headlines.append(text)

                logger.info(f"  Kitco: {len(headlines)} headlines")

        except Exception as e:
            logger.warning(f"  Kitco failed: {e}")

        return headlines
