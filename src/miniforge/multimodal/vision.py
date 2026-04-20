"""Vision and multimodal processing."""

from typing import Union, Optional
from pathlib import Path
import base64
from io import BytesIO
import logging

logger = logging.getLogger(__name__)


class VisionProcessor:
    """
    Process images for multimodal understanding.

    Handles:
    - Image loading and format conversion
    - Resizing to model-compatible dimensions
    - Base64 encoding for API transmission
    """

    SUPPORTED_FORMATS = ["JPEG", "PNG", "WEBP", "GIF", "BMP"]
    DEFAULT_MAX_SIZE = 1024  # Max dimension for resize

    def __init__(self, max_size: int = DEFAULT_MAX_SIZE):
        self.max_size = max_size
        self._pil_available = self._check_pil()

    def _check_pil(self) -> bool:
        """Check if PIL is available."""
        try:
            from PIL import Image

            return True
        except ImportError:
            logger.warning("PIL not available, vision features disabled")
            return False

    def load_image(self, source: Union[str, Path, bytes]) -> "Image.Image":
        """
        Load image from various sources.

        Args:
            source: File path, URL, or bytes

        Returns:
            PIL Image object
        """
        if not self._pil_available:
            raise ImportError("PIL required for image processing")

        from PIL import Image

        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.exists():
                return Image.open(path)
            else:
                # Try as URL
                return self._load_from_url(str(source))
        elif isinstance(source, bytes):
            return Image.open(BytesIO(source))
        else:
            raise ValueError(f"Unsupported image source type: {type(source)}")

    def _load_from_url(self, url: str) -> "Image.Image":
        """Load image from URL."""
        import requests
        from PIL import Image

        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))

    def preprocess(
        self,
        image: Union[str, Path, "Image.Image", bytes],
        target_size: Optional[tuple] = None,
    ) -> "Image.Image":
        """
        Preprocess image for model input.

        Args:
            image: Image source
            target_size: Optional (width, height) to resize to

        Returns:
            Preprocessed PIL Image
        """
        if not self._pil_available:
            raise ImportError("PIL required for image processing")

        from PIL import Image

        # Load if needed
        if not isinstance(image, Image.Image):
            img = self.load_image(image)
        else:
            img = image

        # Convert to RGB
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Resize if needed
        if target_size:
            img = img.resize(target_size, Image.Resampling.LANCZOS)
        elif max(img.size) > self.max_size:
            # Resize maintaining aspect ratio
            img.thumbnail((self.max_size, self.max_size), Image.Resampling.LANCZOS)

        return img

    def encode_base64(
        self,
        image: Union[str, Path, "Image.Image", bytes],
        format: str = "PNG",
    ) -> str:
        """
        Encode image to base64 string.

        Args:
            image: Image source
            format: Output format (PNG, JPEG, WEBP)

        Returns:
            Base64-encoded image string
        """
        img = self.preprocess(image)

        buffer = BytesIO()
        img.save(buffer, format=format)
        img_bytes = buffer.getvalue()

        return base64.b64encode(img_bytes).decode("utf-8")

    def create_vision_message(
        self,
        text: str,
        image: Union[str, Path, "Image.Image", bytes],
        image_format: str = "PNG",
    ) -> str:
        """
        Create a multimodal message with image.

        Args:
            text: Text prompt
            image: Image source
            image_format: Image encoding format

        Returns:
            Formatted multimodal prompt
        """
        image_b64 = self.encode_base64(image, image_format)

        # MiniMax format (similar to other multimodal models)
        return f"<image>data:image/{image_format.lower()};base64,{image_b64}</image>\n{text}"

    def batch_preprocess(
        self,
        images: list,
        target_size: Optional[tuple] = None,
    ) -> list:
        """Preprocess multiple images."""
        return [self.preprocess(img, target_size) for img in images]


class MultimodalPromptBuilder:
    """
    Build complex multimodal prompts.

    Supports:
    - Multiple images
    - Interleaved text and images
    - Video frames (as sequence of images)
    """

    def __init__(self, vision_processor: Optional[VisionProcessor] = None):
        self.vision = vision_processor or VisionProcessor()
        self.parts = []

    def add_text(self, text: str) -> "MultimodalPromptBuilder":
        """Add text segment."""
        self.parts.append(("text", text))
        return self

    def add_image(
        self,
        image: Union[str, Path, "Image.Image", bytes],
        description: Optional[str] = None,
    ) -> "MultimodalPromptBuilder":
        """Add image segment."""
        image_b64 = self.vision.encode_base64(image)
        image_tag = f"<image>data:image/png;base64,{image_b64}</image>"
        if description:
            image_tag += f"\n[{description}]"
        self.parts.append(("image", image_tag))
        return self

    def add_images(
        self,
        images: list,
        descriptions: Optional[list] = None,
    ) -> "MultimodalPromptBuilder":
        """Add multiple images."""
        for i, img in enumerate(images):
            desc = descriptions[i] if descriptions and i < len(descriptions) else None
            self.add_image(img, desc)
        return self

    def build(self) -> str:
        """Build the final prompt string."""
        return "\n\n".join(content for _, content in self.parts)

    def clear(self) -> None:
        """Clear all parts."""
        self.parts = []
