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
    Ensures the returned URL directly serves the binary image (image/* Content-Type),
    not an HTML landing page.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Local image file not found: {path}")

    # Method 1: FreeImage.host (very reliable, permanent CDN image link)
    try:
        with open(path, "rb") as f:
            resp = requests.post(
                "https://freeimage.host/api/1/upload",
                data={"key": "6d207e02198a847aa98d0a2a901485a5", "action": "upload", "format": "json"},
                files={"source": (path.name, f, "image/jpeg")},
                timeout=timeout,
            )
            if resp.status_code == 200:
                data = resp.json()
                img_url = data.get("image", {}).get("url")
                if img_url:
                    return img_url
    except Exception:
        pass

    # Method 2: Uguu.se (fast, ephemeral 48h direct image hosting)
    try:
        with open(path, "rb") as f:
            resp = requests.post(
                "https://uguu.se/upload",
                files={"files[]": (path.name, f, "image/jpeg")},
                timeout=timeout,
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("success") and data.get("files"):
                    direct_url = data["files"][0].get("url")
                    if direct_url:
                        return direct_url
    except Exception:
        pass

    # Method 3: Litterbox (Catbox temporary hosting)
    try:
        with open(path, "rb") as f:
            resp = requests.post(
                "https://litterbox.catbox.moe/resources/internals/api.php",
                data={"reqtype": "fileupload", "time": "1h"},
                files={"fileToUpload": (path.name, f, "image/jpeg")},
                timeout=timeout,
            )
            if resp.status_code == 200 and resp.text.startswith("http"):
                return resp.text.strip()
    except Exception:
        pass

    raise LensSearchError(
        "Could not automatically stage local image for SerpAPI. "
        "Please provide a direct public image URL via --image or check your internet connection."
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

        # Save sanitized debug response
        try:
            import copy
            import json
            debug_data = copy.deepcopy(data)
            if "search_parameters" in debug_data and "api_key" in debug_data["search_parameters"]:
                debug_data["search_parameters"]["api_key"] = "[REDACTED]"
            debug_path = Path("output/debug_lens_response.json")
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            with open(debug_path, "w", encoding="utf-8") as f:
                json.dump(debug_data, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

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

        # Also extract from knowledge_graph if available
        if "knowledge_graph" in data and isinstance(data["knowledge_graph"], dict):
            kg = data["knowledge_graph"]
            kg_link = kg.get("link") or kg.get("website")
            if kg_link and not any(c.post_url == kg_link for c in candidates):
                candidates.insert(
                    0,
                    LensCandidate(
                        title=kg.get("title", "Knowledge Graph Entry"),
                        post_url=kg_link,
                        source=kg.get("source", "Knowledge Graph"),
                        image_url=kg.get("thumbnail") or kg.get("header_images", [{}])[0].get("image") if isinstance(kg.get("header_images"), list) and kg.get("header_images") else None,
                        thumbnail_url=kg.get("thumbnail"),
                        position=0,
                        snippet=kg.get("description"),
                        raw_data=kg,
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
