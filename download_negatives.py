import sys
import os

# Polyfill for imghdr which was removed in Python 3.13
class DummyImghdr:
    @staticmethod
    def what(file, h=None):
        if h is None:
            if isinstance(file, str) and os.path.exists(file):
                with open(file, 'rb') as f:
                    h = f.read(32)
            else:
                return None
        if not h: return None
        if h.startswith(b'\xff\xd8'): return 'jpeg'
        elif h.startswith(b'\x89PNG\r\n\x1a\n'): return 'png'
        elif h.startswith(b'GIF87a') or h.startswith(b'GIF89a'): return 'gif'
        elif h.startswith(b'RIFF') and h[8:12] == b'WEBP': return 'webp'
        if isinstance(file, str):
            ext = os.path.splitext(file)[1].lower()
            if ext in ['.jpg', '.jpeg']: return 'jpeg'
            if ext == '.png': return 'png'
            if ext == '.gif': return 'gif'
            if ext == '.webp': return 'webp'
        return 'jpeg'

sys.modules['imghdr'] = DummyImghdr()

from bing_image_downloader import downloader

# Search terms to train the AI on what NOT to detect as a weapon
queries = [
    "empty classroom cctv",
    "person holding cell phone",
    "office hallway security camera",
    "people walking with umbrellas",
    "person holding a wallet",
    "person drinking from water bottle",
    "person holding pen close up",
    "person holding remote control",
    "person holding kitchen utensil",
    "person holding scissors closed"
]

output_dir = "negative_samples_backgrounds"

print("Starting automatic download of Negative Background Images...")

# Download 80 images per query (Total: 800 images)
for query in queries:
    downloader.download(
        query, 
        limit=80,  
        output_dir=output_dir, 
        adult_filter_off=False, 
        force_replace=False, 
        timeout=60, 
        verbose=True
    )

print(f"\n✅ SUCCESS: All negative images downloaded to the '{output_dir}' folder!")
print("Next Step: Copy all the images inside those folders and paste them directly into your dataset's 'train/images' folder!")
