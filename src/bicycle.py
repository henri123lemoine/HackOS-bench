# This code was modified from ExoLabs' https://github.com/exobyte-labs/betamark
# Note from Laurence: I removed a couple of entires after checking a few for BICYCLE presence/absence.

import os
import hashlib
from pathlib import Path
import tqdm
import requests
from io import BytesIO
import imageio.v3 as iio


### Train dataset ###

LIST_OF_BICYCLES = [
    "http://images.cocodataset.org/train2017/000000483108.jpg",
    "http://images.cocodataset.org/train2017/000000293802.jpg",
    "http://images.cocodataset.org/train2017/000000079841.jpg",
    "http://images.cocodataset.org/train2017/000000515289.jpg",
    "http://images.cocodataset.org/train2017/000000562150.jpg",
    "http://images.cocodataset.org/train2017/000000412151.jpg",
    "http://images.cocodataset.org/train2017/000000462565.jpg",
    "http://images.cocodataset.org/train2017/000000509822.jpg",
    "http://images.cocodataset.org/train2017/000000321107.jpg",
    "http://images.cocodataset.org/train2017/000000061181.jpg",
    "http://images.cocodataset.org/train2017/000000018783.jpg",
    "http://images.cocodataset.org/train2017/000000012896.jpg",
    "http://images.cocodataset.org/train2017/000000465692.jpg",
    "http://images.cocodataset.org/train2017/000000391584.jpg",
    "http://images.cocodataset.org/train2017/000000241350.jpg",
    "http://images.cocodataset.org/train2017/000000438024.jpg",
    "http://images.cocodataset.org/train2017/000000442726.jpg",
    "http://images.cocodataset.org/train2017/000000435937.jpg",
    "http://images.cocodataset.org/train2017/000000292819.jpg",
    "http://images.cocodataset.org/train2017/000000157416.jpg",
    "http://images.cocodataset.org/train2017/000000010393.jpg",
    "http://images.cocodataset.org/train2017/000000084540.jpg",
    "http://images.cocodataset.org/train2017/000000007125.jpg",
    "http://images.cocodataset.org/train2017/000000507249.jpg",
    "http://images.cocodataset.org/train2017/000000075923.jpg",
    "http://images.cocodataset.org/train2017/000000240918.jpg",
    "http://images.cocodataset.org/train2017/000000122302.jpg",
    "http://images.cocodataset.org/train2017/000000140006.jpg",
    "http://images.cocodataset.org/train2017/000000536444.jpg",
    "http://images.cocodataset.org/train2017/000000344271.jpg",
    "http://images.cocodataset.org/train2017/000000420081.jpg",
    "http://images.cocodataset.org/train2017/000000148668.jpg",
    "http://images.cocodataset.org/train2017/000000390137.jpg",
    "http://images.cocodataset.org/train2017/000000114183.jpg",
    "http://images.cocodataset.org/train2017/000000020307.jpg",
    "http://images.cocodataset.org/train2017/000000280736.jpg",
    "http://images.cocodataset.org/train2017/000000536321.jpg",
    "http://images.cocodataset.org/train2017/000000188146.jpg",
    "http://images.cocodataset.org/train2017/000000559312.jpg",
    "http://images.cocodataset.org/train2017/000000535808.jpg",
    "http://images.cocodataset.org/train2017/000000451944.jpg",
    "http://images.cocodataset.org/train2017/000000212558.jpg",
    "http://images.cocodataset.org/train2017/000000377867.jpg",
    "http://images.cocodataset.org/train2017/000000139291.jpg",
    "http://images.cocodataset.org/train2017/000000456323.jpg",
    "http://images.cocodataset.org/train2017/000000549386.jpg",
    "http://images.cocodataset.org/train2017/000000254491.jpg",
    "http://images.cocodataset.org/train2017/000000314515.jpg",
    "http://images.cocodataset.org/train2017/000000415904.jpg",
    "http://images.cocodataset.org/train2017/000000101636.jpg",
    "http://images.cocodataset.org/train2017/000000315173.jpg",
    "http://images.cocodataset.org/train2017/000000260627.jpg",
    "http://images.cocodataset.org/train2017/000000001722.jpg",
    "http://images.cocodataset.org/train2017/000000031092.jpg",
    "http://images.cocodataset.org/train2017/000000556205.jpg",
    "http://images.cocodataset.org/train2017/000000049097.jpg",
    "http://images.cocodataset.org/train2017/000000070815.jpg",
    "http://images.cocodataset.org/train2017/000000467000.jpg",
    "http://images.cocodataset.org/train2017/000000416733.jpg",
    "http://images.cocodataset.org/train2017/000000203912.jpg",
    "http://images.cocodataset.org/train2017/000000408143.jpg",
    "http://images.cocodataset.org/train2017/000000120340.jpg",
    "http://images.cocodataset.org/train2017/000000124462.jpg",
    "http://images.cocodataset.org/train2017/000000142718.jpg",
    "http://images.cocodataset.org/train2017/000000108838.jpg",
    "http://images.cocodataset.org/train2017/000000445309.jpg",
    "http://images.cocodataset.org/train2017/000000140197.jpg",
    "http://images.cocodataset.org/train2017/000000012993.jpg",
    "http://images.cocodataset.org/train2017/000000111099.jpg",
    "http://images.cocodataset.org/train2017/000000215867.jpg",
    "http://images.cocodataset.org/train2017/000000565085.jpg",
    "http://images.cocodataset.org/train2017/000000314986.jpg",
    "http://images.cocodataset.org/train2017/000000158708.jpg",
    "http://images.cocodataset.org/train2017/000000263961.jpg",
    "http://images.cocodataset.org/train2017/000000192128.jpg",
    "http://images.cocodataset.org/train2017/000000377832.jpg",
    "http://images.cocodataset.org/train2017/000000187286.jpg",
    "http://images.cocodataset.org/train2017/000000195510.jpg",
    "http://images.cocodataset.org/train2017/000000406949.jpg",
    "http://images.cocodataset.org/train2017/000000330455.jpg",
]


LIST_OF_NON_BICYCLES = [
    "http://images.cocodataset.org/train2017/000000522418.jpg",
    "http://images.cocodataset.org/train2017/000000184613.jpg",
    "http://images.cocodataset.org/train2017/000000318219.jpg",
    "http://images.cocodataset.org/train2017/000000554625.jpg",
    "http://images.cocodataset.org/train2017/000000574769.jpg",
    "http://images.cocodataset.org/train2017/000000060623.jpg",
    "http://images.cocodataset.org/train2017/000000309022.jpg",
    "http://images.cocodataset.org/train2017/000000005802.jpg",
    "http://images.cocodataset.org/train2017/000000222564.jpg",
    "http://images.cocodataset.org/train2017/000000118113.jpg",
    "http://images.cocodataset.org/train2017/000000193271.jpg",
    "http://images.cocodataset.org/train2017/000000224736.jpg",
    "http://images.cocodataset.org/train2017/000000403013.jpg",
    "http://images.cocodataset.org/train2017/000000374628.jpg",
    "http://images.cocodataset.org/train2017/000000328757.jpg",
    "http://images.cocodataset.org/train2017/000000384213.jpg",
    "http://images.cocodataset.org/train2017/000000086408.jpg",
    "http://images.cocodataset.org/train2017/000000372938.jpg",
    "http://images.cocodataset.org/train2017/000000386164.jpg",
    "http://images.cocodataset.org/train2017/000000223648.jpg",
    "http://images.cocodataset.org/train2017/000000204805.jpg",
    "http://images.cocodataset.org/train2017/000000113588.jpg",
    "http://images.cocodataset.org/train2017/000000384553.jpg",
    "http://images.cocodataset.org/train2017/000000337264.jpg",
    "http://images.cocodataset.org/train2017/000000368402.jpg",
    "http://images.cocodataset.org/train2017/000000012448.jpg",
    "http://images.cocodataset.org/train2017/000000542145.jpg",
    "http://images.cocodataset.org/train2017/000000540186.jpg",
    "http://images.cocodataset.org/train2017/000000242611.jpg",
    "http://images.cocodataset.org/train2017/000000051191.jpg",
    "http://images.cocodataset.org/train2017/000000269105.jpg",
    "http://images.cocodataset.org/train2017/000000294832.jpg",
    "http://images.cocodataset.org/train2017/000000144941.jpg",
    "http://images.cocodataset.org/train2017/000000173350.jpg",
    "http://images.cocodataset.org/train2017/000000060760.jpg",
    "http://images.cocodataset.org/train2017/000000324266.jpg",
    "http://images.cocodataset.org/train2017/000000166532.jpg",
    "http://images.cocodataset.org/train2017/000000262284.jpg",
    "http://images.cocodataset.org/train2017/000000360772.jpg",
    "http://images.cocodataset.org/train2017/000000191381.jpg",
    "http://images.cocodataset.org/train2017/000000111076.jpg",
    "http://images.cocodataset.org/train2017/000000340559.jpg",
    "http://images.cocodataset.org/train2017/000000258985.jpg",
    "http://images.cocodataset.org/train2017/000000229643.jpg",
    "http://images.cocodataset.org/train2017/000000125059.jpg",
    "http://images.cocodataset.org/train2017/000000455483.jpg",
    "http://images.cocodataset.org/train2017/000000436141.jpg",
    "http://images.cocodataset.org/train2017/000000129001.jpg",
    "http://images.cocodataset.org/train2017/000000232262.jpg",
    "http://images.cocodataset.org/train2017/000000166323.jpg",
    "http://images.cocodataset.org/train2017/000000580041.jpg",
    "http://images.cocodataset.org/train2017/000000326781.jpg",
    "http://images.cocodataset.org/train2017/000000387362.jpg",
    "http://images.cocodataset.org/train2017/000000138079.jpg",
    "http://images.cocodataset.org/train2017/000000556616.jpg",
    "http://images.cocodataset.org/train2017/000000472621.jpg",
    "http://images.cocodataset.org/train2017/000000192440.jpg",
    "http://images.cocodataset.org/train2017/000000086320.jpg",
    "http://images.cocodataset.org/train2017/000000256668.jpg",
    "http://images.cocodataset.org/train2017/000000383445.jpg",
    "http://images.cocodataset.org/train2017/000000565797.jpg",
    "http://images.cocodataset.org/train2017/000000081922.jpg",
    "http://images.cocodataset.org/train2017/000000050125.jpg",
    "http://images.cocodataset.org/train2017/000000364521.jpg",
    "http://images.cocodataset.org/train2017/000000394892.jpg",
    "http://images.cocodataset.org/train2017/000000001146.jpg",
    "http://images.cocodataset.org/train2017/000000310391.jpg",
    "http://images.cocodataset.org/train2017/000000097434.jpg",
    "http://images.cocodataset.org/train2017/000000463836.jpg",
    "http://images.cocodataset.org/train2017/000000241876.jpg",
    "http://images.cocodataset.org/train2017/000000156832.jpg",
    "http://images.cocodataset.org/train2017/000000270721.jpg",
    "http://images.cocodataset.org/train2017/000000462341.jpg",
    "http://images.cocodataset.org/train2017/000000310103.jpg",
    "http://images.cocodataset.org/train2017/000000032992.jpg",
    "http://images.cocodataset.org/train2017/000000122851.jpg",
    "http://images.cocodataset.org/train2017/000000540763.jpg",
    "http://images.cocodataset.org/train2017/000000138246.jpg",
    "http://images.cocodataset.org/train2017/000000197254.jpg",
    "http://images.cocodataset.org/train2017/000000032907.jpg",
]


class ImageCache:
    def __init__(self, cache_dir=None):
        if cache_dir is None:
            cache_dir = os.path.join(str(Path.home()), ".betamark", "image_cache")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_cache_path(self, url):
        # Create a unique filename based on URL hash, using PNG format
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return self.cache_dir / f"{url_hash}.png"
    
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
            iio.imwrite(cache_path, np_img, plugin='pillow')
            return np_img
        except Exception as e:
            print(f"Error downloading/caching image from {url}: {e}")
            return None

def run_eval(user_func) -> dict:
    cache = ImageCache()
    correct_answers = 0
    total_answers = 0
    failed_downloads = 0

    print("Evaluating bicycle images...")
    for i in tqdm.trange(len(LIST_OF_BICYCLES)):
        np_img = cache.get_image(LIST_OF_BICYCLES[i])
        if np_img is None:
            failed_downloads += 1
            continue
            
        total_answers += 1
        if user_func(np_img) == 1:
            correct_answers += 1

    print("Evaluating non-bicycle images...")
    for i in tqdm.trange(len(LIST_OF_NON_BICYCLES)):
        np_img = cache.get_image(LIST_OF_NON_BICYCLES[i])
        if np_img is None:
            failed_downloads += 1
            continue
            
        total_answers += 1
        if user_func(np_img) == 0:
            correct_answers += 1

    if failed_downloads > 0:
        print(f"Warning: {failed_downloads} images failed to download/load")
    
    acc = correct_answers / total_answers if total_answers > 0 else 0
    return {
        "acc": acc,
        "total_evaluated": total_answers,
        "failed_downloads": failed_downloads
    }
