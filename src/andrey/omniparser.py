"""OmniParser wrapper for UI element detection from screenshots."""

import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class UIElement:
    """A detected UI element with bounding box and description."""

    id: int
    type: str  # "text" or "icon"
    content: str  # text content or icon description
    bbox: list[float]  # [x1, y1, x2, y2] in pixels
    interactive: bool
    center_x: int = 0
    center_y: int = 0

    def __post_init__(self):
        if self.bbox and len(self.bbox) == 4:
            self.center_x = int((self.bbox[0] + self.bbox[2]) / 2)
            self.center_y = int((self.bbox[1] + self.bbox[3]) / 2)


@dataclass
class ParseResult:
    """Result of OmniParser processing a screenshot."""

    annotated_image: Image.Image
    elements: list[UIElement] = field(default_factory=list)
    latency_ms: float = 0.0


class OmniParserClient:
    """Wraps OmniParser V2 for local UI element detection.

    Falls back gracefully if OmniParser is not installed — returns the
    raw screenshot with an empty element list.
    """

    def __init__(
        self,
        omniparser_path: str = "",
        weights_path: str = "",
        device: str = "mps",
        box_threshold: float = 0.05,
        iou_threshold: float = 0.7,
        use_paddleocr: bool = False,
    ):
        self._omniparser_path = omniparser_path
        self._weights_path = weights_path
        self._device = device
        self._box_threshold = box_threshold
        self._iou_threshold = iou_threshold
        self._use_paddleocr = use_paddleocr
        self._som_model = None
        self._caption_model_processor = None
        self._loaded = False
        self._available = None  # None = not checked yet

    @property
    def available(self) -> bool:
        """Check if OmniParser dependencies are available."""
        if self._available is None:
            self._available = self._check_available()
        return self._available

    def _check_available(self) -> bool:
        """Check if OmniParser can be imported."""
        try:
            import torch  # noqa: F401
            import ultralytics  # noqa: F401
            return True
        except ImportError:
            logger.warning(
                "OmniParser dependencies not installed. "
                "Install with: pip install -e '.[omniparser]' "
                "Element detection will be disabled."
            )
            return False

    def _find_omniparser_path(self) -> Optional[Path]:
        """Find the OmniParser installation directory."""
        candidates = [
            Path(self._omniparser_path) if self._omniparser_path else None,
            Path.home() / "OmniParser",
            Path.home() / "projects" / "OmniParser",
            Path("/opt/OmniParser"),
        ]
        for p in candidates:
            if p and p.exists() and (p / "util" / "utils.py").exists():
                return p
        return None

    def _find_weights_path(self, omniparser_dir: Path) -> Optional[Path]:
        """Find the OmniParser weights directory."""
        candidates = [
            Path(self._weights_path) if self._weights_path else None,
            omniparser_dir / "weights",
        ]
        for p in candidates:
            if p and p.exists() and (p / "icon_detect").exists():
                return p
        return None

    def _load_models(self) -> bool:
        """Load OmniParser models. Returns True if successful."""
        if self._loaded:
            return True

        if not self.available:
            return False

        omniparser_dir = self._find_omniparser_path()
        if not omniparser_dir:
            logger.warning(
                "OmniParser directory not found. "
                "Clone it: git clone https://github.com/microsoft/OmniParser.git ~/OmniParser"
            )
            self._available = False
            return False

        weights_dir = self._find_weights_path(omniparser_dir)
        if not weights_dir:
            logger.warning(
                f"OmniParser weights not found in {omniparser_dir}/weights. "
                "Download with: huggingface-cli download microsoft/OmniParser-v2.0 --local-dir weights"
            )
            self._available = False
            return False

        # Add OmniParser to sys.path so we can import its utilities
        omniparser_str = str(omniparser_dir)
        if omniparser_str not in sys.path:
            sys.path.insert(0, omniparser_str)

        # OmniParser's utils.py unconditionally imports paddleocr at module level.
        # If paddleocr isn't installed (we use EasyOCR), stub it out to avoid ImportError.
        if "paddleocr" not in sys.modules:
            try:
                import paddleocr  # noqa: F401
            except ImportError:
                import types

                stub = types.ModuleType("paddleocr")
                stub.PaddleOCR = lambda **kwargs: None
                sys.modules["paddleocr"] = stub
                logger.debug("Stubbed paddleocr module (not installed, using EasyOCR)")

        try:
            from util.utils import get_caption_model_processor, get_yolo_model

            logger.info(f"Loading OmniParser models on device={self._device}...")
            t0 = time.monotonic()

            # Load YOLO detection model
            detect_path = str(weights_dir / "icon_detect" / "model.pt")
            self._som_model = get_yolo_model(model_path=detect_path)
            self._som_model.to(self._device)

            # Load Florence-2 caption model
            caption_path = weights_dir / "icon_caption_florence"
            if not caption_path.exists():
                caption_path = weights_dir / "icon_caption"

            # OmniParser's get_caption_model_processor uses float16 for
            # non-CPU devices, which causes dtype mismatch on MPS.
            # For MPS: load on CPU first (float32), then move to MPS.
            load_device = self._device if self._device == "cuda" else "cpu"
            self._caption_model_processor = get_caption_model_processor(
                model_name="florence2",
                model_name_or_path=str(caption_path),
                device=load_device,
            )

            # WORKAROUND: OmniParser's get_parsed_content_icon checks
            # 'florence' in model.config.name_or_path to decide the
            # code path. When loaded from a local path like
            # /Users/.../icon_caption, that check fails and it takes
            # a wrong branch that passes attention_mask incorrectly.
            # Patch name_or_path to include 'florence'.
            model = self._caption_model_processor["model"]
            if "florence" not in model.config.name_or_path:
                model.config.name_or_path = "florence2-local"

            # Move to MPS if needed (already float32 from CPU load)
            if self._device == "mps":
                self._caption_model_processor["model"] = model.to(self._device)

            elapsed = (time.monotonic() - t0) * 1000
            logger.info(f"OmniParser models loaded in {elapsed:.0f}ms")
            self._loaded = True
            return True

        except Exception as e:
            logger.warning(f"Failed to load OmniParser models: {e}")
            self._available = False
            return False

    def parse(self, image: Image.Image) -> ParseResult:
        """Parse a screenshot to detect UI elements.

        Returns ParseResult with annotated image and element list.
        Falls back to raw image with empty elements if OmniParser unavailable.
        """
        if not self._load_models():
            return ParseResult(annotated_image=image, elements=[], latency_ms=0.0)

        t0 = time.monotonic()

        try:
            result = self._run_detection(image)
            result.latency_ms = (time.monotonic() - t0) * 1000
            logger.info(
                f"OmniParser: {len(result.elements)} elements detected "
                f"in {result.latency_ms:.0f}ms"
            )
            return result
        except Exception as e:
            logger.warning(f"OmniParser detection failed: {e}")
            return ParseResult(
                annotated_image=image,
                elements=[],
                latency_ms=(time.monotonic() - t0) * 1000,
            )

    def _run_detection(self, image: Image.Image) -> ParseResult:
        """Run the full OmniParser pipeline on an image."""
        from util.utils import check_ocr_box, get_som_labeled_img

        w, h = image.size

        # Step 1: Run OCR (check_ocr_box accepts PIL Image directly)
        ocr_bbox_rslt, _ = check_ocr_box(
            image,
            display_img=False,
            output_bb_format="xyxy",
            goal_filtering=None,
            easyocr_args={"paragraph": False, "text_threshold": 0.9},
            use_paddleocr=self._use_paddleocr,
        )
        text_list, ocr_bbox = ocr_bbox_rslt

        # Step 2: Run detection + captioning
        box_overlay_ratio = max(image.size) / 3200
        draw_bbox_config = {
            "text_scale": 0.8 * box_overlay_ratio,
            "text_thickness": max(int(2 * box_overlay_ratio), 1),
            "text_padding": max(int(3 * box_overlay_ratio), 1),
            "thickness": max(int(3 * box_overlay_ratio), 1),
        }

        # get_som_labeled_img accepts PIL Image directly
        # output_coord_in_ratio only affects label_coordinates (2nd return),
        # not parsed_content_list (3rd return) which always has ratio coords
        dino_labeled_img_b64, label_coordinates, parsed_content_list = (
            get_som_labeled_img(
                image,
                self._som_model,
                BOX_TRESHOLD=self._box_threshold,
                output_coord_in_ratio=True,
                ocr_bbox=ocr_bbox,
                draw_bbox_config=draw_bbox_config,
                caption_model_processor=self._caption_model_processor,
                ocr_text=text_list,
                use_local_semantics=True,
                iou_threshold=self._iou_threshold,
                scale_img=False,
                batch_size=128,
            )
        )

        # Decode annotated image from base64
        import base64
        from io import BytesIO

        annotated_image = Image.open(
            BytesIO(base64.b64decode(dino_labeled_img_b64))
        )

        # Convert parsed content to UIElement list
        # Bounding boxes in parsed_content_list are always in ratio format (0-1)
        elements = []
        for idx, item in enumerate(parsed_content_list):
            bbox_ratio = item.get("bbox", [0, 0, 0, 0])
            # Convert ratio coords to pixel coords
            bbox_px = [
                bbox_ratio[0] * w,
                bbox_ratio[1] * h,
                bbox_ratio[2] * w,
                bbox_ratio[3] * h,
            ]
            elements.append(
                UIElement(
                    id=idx,
                    type=item.get("type", "unknown"),
                    content=item.get("content", "") or "",
                    bbox=[float(c) for c in bbox_px],
                    interactive=item.get("interactivity", False),
                )
            )

        return ParseResult(
            annotated_image=annotated_image,
            elements=elements,
        )

    @staticmethod
    def format_elements_text(
        elements: list[UIElement], screen_height: int = 0
    ) -> str:
        """Format element list as text for Claude.

        Args:
            elements: Detected UI elements.
            screen_height: If >0, elements in the bottom 10% are filtered
                out (typically ad banners).
        """
        if not elements:
            return "No UI elements detected. Use tap(x, y) with estimated coordinates."

        ad_cutoff = int(screen_height * 0.9) if screen_height > 0 else 0

        lines = ["Detected UI elements (use tap_element with the element ID):"]
        for el in elements:
            # Skip elements in the ad banner zone
            if ad_cutoff and el.center_y > ad_cutoff:
                continue
            interactive_str = " [INTERACTIVE]" if el.interactive else ""
            w = int(el.bbox[2] - el.bbox[0]) if len(el.bbox) == 4 else 0
            h = int(el.bbox[3] - el.bbox[1]) if len(el.bbox) == 4 else 0
            lines.append(
                f"  [{el.id}] {el.type}: \"{el.content}\" "
                f"at ({el.center_x}, {el.center_y}) "
                f"size {w}x{h}{interactive_str}"
            )
        return "\n".join(lines)
