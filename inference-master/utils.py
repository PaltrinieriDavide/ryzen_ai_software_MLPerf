import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
from torchvision import transforms
import numpy as np
import sys
from PIL import Image

# --- Constants ---
IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]
IMAGE_RESIZE = 256
IMAGE_CROP = 224

def setup_logging(log_dir: Path) -> None:
    log = logging.getLogger("MLPerfMultiStream")

    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = (log_dir / "benchmark.log").resolve()

    logging.basicConfig(level=logging.INFO)
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)-5.5s] %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    log.addHandler(file_handler)
    log.addHandler(console_handler)
    log.propagate = False
    log.info(f"Logging configured. Log file at: {log_file}")
    return log


class ImagenetDataset:
    def __init__(self, image_paths: List[Path], preprocessor: transforms.Compose):
        self.image_paths = image_paths
        self.preprocessor = preprocessor
        self.cache: Dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.image_paths)

    def load_samples(self, indices: List[int]) -> None:
        """Loads and preprocesses images into an in-memory cache."""
        for idx in indices:
            if idx in self.cache:
                continue
            try:
                img = Image.open(self.image_paths[idx]).convert("RGB")
                # unsqueeze(0) adds the batch dimension, creating a (1, C, H, W) tensor
                tensor = self.preprocessor(img).unsqueeze(0).numpy()
                self.cache[idx] = tensor
            except Exception as e:
                self.cache[idx] = (np.zeros((1, 3, IMAGE_CROP, IMAGE_CROP), dtype=np.float32), -1)

    def unload_samples(self, indices: List[int]) -> None:
        """Removes samples from the in-memory cache."""
        for idx in indices:
            self.cache.pop(idx, None)

    def get_sample(self, idx: int) -> Tuple[np.ndarray, int]:
        """Retrieves a single sample from the cache."""
        return self.cache[idx]