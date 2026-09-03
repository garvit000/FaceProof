"""
FaceProof: Face Identification & Reverse Search Pipeline.
CLI Entrypoint for face detection, encoding, runtime Google Lens reverse search,
candidate ranking, deterministic serialization, and SHA-256 fingerprint generation.
"""

import argparse
import os
from pathlib import Path
import sys
from typing import Optional

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Load environment variables
load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Initialize Rich console
console = Console()

# Internal modules
from face.detector import detect_faces, load_image
from face.cropper import crop_face, save_face_crop
from face.encoder import encode_face
from search.lens import GoogleLensSearcher, SerpApiKeyMissingError, LensSearchError, LensSearchResult, LensCandidate
from search.matcher import CandidateMatcher, MatchResult
from utils.hashing import generate_sha256_fingerprint
from utils.metadata import build_handoff_payload, save_result_payload


def print_banner():
    """Print the FaceProof application header."""
    banner_text = Text()
    banner_text.append("FACEPROOF\n", style="bold cyan")
    banner_text.append("Face Identification & Web Verification Pipeline\n", style="bold white")
    banner_text.append("Deterministic SHA-256 Fingerprint Generator for Blockchain Handoff", style="dim")
    console.print(Panel(banner_text, border_style="cyan", expand=False))


def format_step(step_num: int, total_steps: int, title: str):
    """Format and print step headers."""
    console.print(f"\n[bold yellow][{step_num}/{total_steps}][/bold yellow] [bold white]{title}[/bold white]")


def print_success(msg: str):
    """Print a success line."""
    console.print(f"  [bold green]✓[/bold green] {msg}")


def print_warning(msg: str):
    """Print a warning line."""
    console.print(f"  [bold yellow]![/bold yellow] {msg}")


def print_error(msg: str):
    """Print an error line."""
    console.print(f"\n[bold red]Error:[/bold red] {msg}")


def run_pipeline(
    image_path: str,
    crop_padding: float = 0.25,
    top_k_validate: int = 5,
    output_path: str = "output/result.json",
    dry_run: bool = False,
) -> int:
    """Execute the end-to-end FaceProof verification pipeline."""
    print_banner()

    # -------------------------------------------------------------
    # Step 1: Load Image
    # -------------------------------------------------------------
    format_step(1, 6, "Loading image...")
    img_file = Path(image_path)
    if not img_file.exists():
        print_error(f"Image file not found: '{img_file}'")
        console.print("[dim]Please place a valid face image in 'samples/' or specify --image <path>[/dim]")
        return 1

    try:
        image_bgr = load_image(img_file)
        h, w = image_bgr.shape[:2]
        print_success(f"{img_file.name} ({w}x{h} px)")
    except Exception as e:
        print_error(f"Failed to load image '{img_file}': {e}")
        return 1

    # -------------------------------------------------------------
    # Step 2: Detect Face(s)
    # -------------------------------------------------------------
    format_step(2, 6, "Detecting face...")
    try:
        faces = detect_faces(image_bgr)
    except Exception as e:
        print_error(f"Face detector failure: {e}")
        return 1

    if not faces:
        print_error("Zero faces detected in the input image.")
        console.print("[yellow]FaceProof requires an input image with at least one clearly visible face.[/yellow]")
        return 1

    face_count = len(faces)
    if face_count == 1:
        print_success("1 face detected (confidence: {:.2f})".format(faces[0].confidence))
    else:
        print_warning(f"Multiple faces detected ({face_count} total).")
        print_warning(
            f"Processing primary face (largest bounding box: {faces[0].width}x{faces[0].height} px, "
            f"confidence: {faces[0].confidence:.2f})"
        )

    primary_face = faces[0]

    # -------------------------------------------------------------
    # Step 3: Face Crop & Embedding Generation
    # -------------------------------------------------------------
    format_step(3, 6, "Encoding face & preparing crop...")
    try:
        # Padded crop
        crop_bgr, crop_box = crop_face(image_bgr, primary_face, padding_ratio=crop_padding)
        crop_file_path = save_face_crop(crop_bgr, "output/processed/face_crop.jpg")
        print_success(f"Padded face crop saved ({crop_file_path})")

        # Embedding
        embedding = encode_face(image_bgr, primary_face)
        print_success(f"Face encoding generated (128-d L2-normalized vector)")
    except Exception as e:
        print_error(f"Failed during face processing / encoding: {e}")
        return 1

    # -------------------------------------------------------------
    # Step 4: Reverse Image Search
    # -------------------------------------------------------------
    format_step(4, 6, "Searching web via Google Lens...")
    search_info = {
        "engine": "Google Lens via SerpAPI",
        "query_type": "reverse_image",
        "dry_run": dry_run,
    }

    if dry_run:
        print_warning("DRY-RUN MODE ACTIVE: Skipping external SerpAPI query.")
        print_warning("Note: Dry-run results are for local syntax & hashing checks only, not valid for final demo.")
        search_result = LensSearchResult(
            engine="Google Lens (Dry-Run Simulation)",
            query_type="reverse_image",
            query_image_url="local://output/processed/face_crop.jpg",
            total_results=1,
            candidates=[
                LensCandidate(
                    title="Dry Run Demo Candidate",
                    post_url="https://example.com/dry-run-sample",
                    source="ExampleWeb",
                    image_url=None,
                    thumbnail_url=None,
                    position=1,
                    snippet="Dry-run simulation candidate for local pipeline validation.",
                )
            ],
        )
        print_success("1 simulation candidate created for dry-run")
    else:
        try:
            searcher = GoogleLensSearcher()
            # Search using the cropped face image
            search_result = searcher.search(crop_file_path)
            print_success("Reverse image search completed via Google Lens")
            print_success(f"{search_result.total_results} candidates discovered")
        except SerpApiKeyMissingError as e:
            print_error(str(e))
            console.print("\n[bold yellow]Tip:[/bold yellow] You can also run with [bold cyan]--dry-run[/bold cyan] to verify face detection and hashing offline without an API key.")
            return 1
        except LensSearchError as e:
            print_error(str(e))
            return 1
        except Exception as e:
            print_error(f"Unexpected search error: {e}")
            return 1

    search_info["query_image_url"] = search_result.query_image_url
    search_info["total_results_found"] = search_result.total_results

    # -------------------------------------------------------------
    # Step 5: Candidate Matching & Validation
    # -------------------------------------------------------------
    format_step(5, 6, "Validating & ranking candidates...")
    matcher = CandidateMatcher(top_k_validate=top_k_validate)
    match_result = matcher.evaluate_candidates(search_result, input_embedding=embedding)

    if not match_result.best_candidate:
        print_warning("No suitable matching web/social candidates found by Google Lens.")
        best_match_dict = None
    else:
        best = match_result.best_candidate
        best_match_dict = best.to_dict()
        print_success("Candidate ranking complete")

        # Print candidate table
        table = Table(title="Google Lens Candidate Matches", header_style="bold magenta")
        table.add_column("Rank", justify="center", style="cyan")
        table.add_column("Source", style="green")
        table.add_column("Title / Snippet", style="white", max_width=40)
        table.add_column("Face Similarity", justify="center")
        table.add_column("Status", style="dim")

        for eval_c in match_result.all_evaluated[:8]:  # Show top 8 in table
            sim_str = (
                f"{eval_c.face_similarity:.2f}"
                if eval_c.face_similarity is not None
                else "N/A"
            )
            status_style = "green" if eval_c.validation_status == "face_verified" else "yellow"
            table.add_row(
                str(eval_c.visual_rank),
                eval_c.candidate.source,
                eval_c.candidate.title[:38] + ("..." if len(eval_c.candidate.title) > 38 else ""),
                sim_str,
                f"[{status_style}]{eval_c.validation_status}[/{status_style}]",
            )
        console.print(table)

        console.print(f"\n  [bold green]★ Best Match:[/bold green] [bold white]{best.candidate.title}[/bold white]")
        console.print(f"    [dim]Source:[/dim]   {best.candidate.source}")
        console.print(f"    [dim]URL:[/dim]      [link={best.candidate.post_url}]{best.candidate.post_url}[/link]")
        if best.face_similarity is not None:
            console.print(f"    [dim]Face Sim:[/dim] {best.face_similarity:.4f}")

    # -------------------------------------------------------------
    # Step 6: Deterministic Serialization & SHA-256 Fingerprint
    # -------------------------------------------------------------
    format_step(6, 6, "Creating deterministic fingerprint...")

    input_info = {
        "filename": img_file.name,
        "image_path": str(img_file),
        "face_detected": True,
        "face_count": face_count,
        "primary_face_box": {
            "x": primary_face.x,
            "y": primary_face.y,
            "width": primary_face.width,
            "height": primary_face.height,
        },
        "confidence": round(float(primary_face.confidence), 4),
    }

    all_candidates_dict = [c.to_dict() for c in match_result.all_evaluated]

    final_payload = build_handoff_payload(
        input_info=input_info,
        search_info=search_info,
        best_match=best_match_dict,
        all_candidates=all_candidates_dict,
        crop_path=str(crop_file_path),
        dry_run=dry_run,
    )

    fingerprint = final_payload["fingerprint"]
    out_file = save_result_payload(final_payload, output_path=output_path)

    if fingerprint:
        console.print("\n[bold green][SEARCH SUCCESS + MATCH FOUND][/bold green]")
        print_success(f"Deterministic SHA-256 fingerprint generated")
        console.print(
            Panel(
                f"[bold green]{fingerprint}[/bold green]",
                title="Cryptographic Fingerprint (SHA-256)",
                border_style="green",
                expand=False,
            )
        )
        print_success(f"Pipeline complete → [bold cyan]{out_file}[/bold cyan] saved")
        print_success("Ready for blockchain integration handoff")
    else:
        console.print("\n[bold yellow][SEARCH SUCCESS + NO MATCH][/bold yellow]")
        print_warning("No matching web or social media post was discovered for this face.")
        print_warning("Blockchain fingerprint generation skipped (a verifiable match is required before generating a blockchain record).")
        print_success(f"Search record saved → [bold cyan]{out_file}[/bold cyan]")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="FaceProof: Face Identification & Content Verification Pipeline"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="samples/test.jpg",
        help="Path to input face image (default: samples/test.jpg)",
    )
    parser.add_argument(
        "--crop-padding",
        type=float,
        default=0.25,
        help="Padding ratio around detected face (default: 0.25)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Maximum candidate images to download and validate with face similarity (default: 5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/result.json",
        help="Target path for output JSON payload (default: output/result.json)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run pipeline offline without calling SerpAPI (useful for local syntax and hashing testing)",
    )

    args = parser.parse_args()

    exit_code = run_pipeline(
        image_path=args.image,
        crop_padding=args.crop_padding,
        top_k_validate=args.top_k,
        output_path=args.output,
        dry_run=args.dry_run,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
