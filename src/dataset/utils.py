import hashlib
import requests
from io import BytesIO

import imageio.v3 as iio

from src.settings import CACHE_PATH


class ImageCache:
    def __init__(self):
        pass

    def get_cache_path(self, url):
        # Create a unique filename based on URL hash, using PNG format
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return CACHE_PATH / "bicycle_data" / "eval" / f"{url_hash}.png"

    def get_image(self, url):
        cache_path = self.get_cache_path(url)

        # If image is cached, load it from disk
        if cache_path.exists():
            try:
                return iio.imread(cache_path)
            except Exception as e:
                print(f"Error loading cached image {cache_path}: {e}")
                cache_path.unlink(missing_ok=True)

        # Download and cache the image
        try:
            res = requests.get(url)
            res.raise_for_status()
            np_img = iio.imread(BytesIO(res.content))
            # Save to cache as PNG
            iio.imwrite(cache_path, np_img, plugin="pillow")
            return np_img
        except Exception as e:
            print(f"Error downloading/caching image from {url}: {e}")
            return None
