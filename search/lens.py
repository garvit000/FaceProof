"""
Google Lens reverse-image search integration via SerpAPI for FaceProof.
Queries the SerpAPI google_lens engine at runtime and extracts structured visual matches.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import httpx
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class LensCandidate:
    """Structured representation of a single visual match discovered by Google Lens."""
    title: str
    post_url: str
    source: str
    image_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    position: int = 0
    snippet: Optional[str] = None
    raw_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        data = {
            "source": self.source,
            "post_url": self.post_url,
            "title": self.title,
            "position": self.position,
        }
        if self.image_url:
            data["image_url"] = self.image_url
        if self.thumbnail_url:
            data["thumbnail_url"] = self.thumbnail_url
        if self.snippet:
            data["snippet"] = self.snippet
        return data


@dataclass
class LensSearchResult:
    """Aggregated result from a Google Lens search query."""
    engine: str = "Google Lens via SerpAPI"
    query_type: str = "reverse_image"
    query_image_url: str = ""
    total_results: int = 0
    candidates: List[LensCandidate] = field(default_factory=list)
    raw_response: Dict[str, Any] = field(default_factory=dict)


class SerpApiKeyMissingError(Exception):
    """Raised when SERPAPI_KEY is not configured in the environment."""
    pass


class LensSearchError(Exception):
    """Raised when Google Lens search fails due to API error or network issue."""
    pass


def upload_image_to_temp_host(image_path: Union[str, Path], timeout: int = 15) -> str:
    """
    Upload a local image to a temporary host so it can be queried by Google Lens API.
    Uses public ephemeral file hosting services (tmpfiles.org or 0x0.st).
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Local image file not found: {path}")

    # Try tmpfiles.org first
    try:
        with open(path, "rb") as f:
            resp = requests.post(
                "https://tmpfiles.org/api/v1/upload",
                files={"file": (path.name, f, "image/jpeg")},
                timeout=timeout,
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") == "success":
                    raw_url = data["data"]["url"]
                    # Convert to direct download URL (tmpfiles.org/dl/...)
                    direct_url = raw_url.replace("tmpfiles.org/", "tmpfiles.org/dl/")
                    return direct_url
    except Exception:
        pass

    # Fallback to 0x0.st
    try:
        with open(path, "rb") as f:
            resp = requests.post(
                "https://0x0.st",
                files={"file": (path.name, f, "image/jpeg")},
                timeout=timeout,
            )
            if resp.status_code == 200:
                return resp.text.strip()
    except Exception:
        pass

    # Fallback to file.io
    try:
        with open(path, "rb") as f:
            resp = requests.post(
                "https://file.io",
                files={"file": (path.name, f, "image/jpeg")},
                data={"expires": "1d"},
                timeout=timeout,
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("success"):
                    return data["link"]
    except Exception:
        pass

    raise LensSearchError(
        "Could not automatically stage local image for SerpAPI. "
        "Please provide a direct public image URL or verify your internet connection."
    )


class GoogleLensSearcher:
    """
    Executes real-time reverse image searches using SerpAPI's Google Lens engine.
    """

    SERPAPI_ENDPOINT = "https://serpapi.com/search.json"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("SERPAPI_KEY")

    def search(
        self,
        image_input: Union[str, Path],
        hl: str = "en",
        country: str = "us",
        timeout: int = 30,
    ) -> LensSearchResult:
        """
        Execute a runtime reverse-image search on Google Lens via SerpAPI.

        Args:
            image_input: Public image URL or local file path to the image/crop.
            hl: Language code (default 'en').
            country: Country code (default 'us').
            timeout: HTTP request timeout in seconds.

        Returns:
            LensSearchResult containing parsed candidates.
        """
        if not self.api_key or self.api_key.strip() in ("", "your_serpapi_key_here"):
            raise SerpApiKeyMissingError(
                "SERPAPI_KEY is not configured.\n"
                "Please add your SerpAPI key to the .env file:\n"
                "  SERPAPI_KEY=your_actual_key_here\n"
                "Get a free API key at https://serpapi.com/ (100 free searches/month)."
            )

        # Determine if input is a URL or a local file
        input_str = str(image_input)
        if input_str.startswith("http://") or input_str.startswith("https://"):
            query_url = input_str
        else:
            # Stage local image to a temporary public URL for Google Lens API
            query_url = upload_image_to_temp_host(input_str)

        params = {
            "engine": "google_lens",
            "url": query_url,
            "api_key": self.api_key,
            "hl": hl,
            "country": country,
        }

        try:
            response = requests.get(self.SERPAPI_ENDPOINT, params=params, timeout=timeout)
        except requests.exceptions.Timeout:
            raise LensSearchError("Reverse-image search timed out while connecting to SerpAPI.")
        except requests.exceptions.RequestException as e:
            raise LensSearchError(f"Network error during reverse-image search: {e}")

        if response.status_code == 401 or response.status_code == 403:
            raise SerpApiKeyMissingError("Invalid SerpAPI key or unauthorized request. Please check your SERPAPI_KEY in .env.")
        elif response.status_code == 429:
            raise LensSearchError("SerpAPI rate limit or monthly search quota exceeded.")
        elif response.status_code != 200:
            raise LensSearchError(f"SerpAPI returned HTTP status {response.status_code}: {response.text[:200]}")

        try:
            data = response.json()
        except Exception as e:
            raise LensSearchError(f"Failed to parse SerpAPI JSON response: {e}")

        if "error" in data:
            raise LensSearchError(f"SerpAPI error: {data['error']}")

        return self._parse_lens_response(data, query_url)

    def _parse_lens_response(self, data: Dict[str, Any], query_url: str) -> LensSearchResult:
        """Parse raw SerpAPI Google Lens JSON into structured LensSearchResult."""
        visual_matches = data.get("visual_matches", [])
        candidates: List[LensCandidate] = []

        for idx, match in enumerate(visual_matches):
            title = match.get("title", "").strip() or "Untitled Post / Page"
            post_url = match.get("link", "").strip()
            source = match.get("source", "").strip()
            thumbnail_url = match.get("thumbnail", None)
            original_image = match.get("original_image", None)
            snippet = match.get("snippet", None)
            position = match.get("position", idx + 1)

            # Infer source domain if missing
            if not source and post_url:
                try:
                    from urllib.parse import urlparse
                    domain = urlparse(post_url).netloc
                    source = domain.replace("www.", "")
                except Exception:
                    source = "Web"

            if post_url:
                candidates.append(
                    LensCandidate(
                        title=title,
                        post_url=post_url,
                        source=source,
                        image_url=original_image or thumbnail_url,
                        thumbnail_url=thumbnail_url,
                        position=position,
                        snippet=snippet,
                        raw_data=match,
                    )
                )

        return LensSearchResult(
            engine="Google Lens via SerpAPI",
            query_type="reverse_image",
            query_image_url=query_url,
            total_results=len(candidates),
            candidates=candidates,
            raw_response=data,
        )
