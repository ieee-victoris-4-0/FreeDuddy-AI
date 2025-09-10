# Import Libraries
import pandas as pd
import os
import hashlib
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from tqdm import tqdm

# Get Data from Hugging face
df = pd.read_parquet("hf://datasets/DBQ/Farfetch.Product.prices.Macao/data/train-00000-of-00001-e43844630b47de3b.parquet")
print(f"Shape of the data : {df.shape}")

# images folder
output_dir = 'images'
os.makedirs(output_dir, exist_ok=True)


# Reuse session for efficiency
session = requests.Session()

# headers to avid errors
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/139.0.0.0 Safari/537.36"
})

# images downloading function
def download_image(idx, url, retries=3, delay=2):
    """Download one image with retry logic and error logging."""
    try:
        ext = url.split('.')[-1].split('?')[0] if '.' in url else 'jpg'
        filename = hashlib.md5(url.encode()).hexdigest() + f".{ext}"
        filepath = os.path.join(output_dir, filename)

        # Skip if already exists
        if os.path.exists(filepath):
            return idx, filepath, None

        for attempt in range(retries):
            try:
                with session.get(url, stream=True, timeout=15) as r:
                    r.raise_for_status()
                    with open(filepath, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                return idx, filepath, None  # success
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(delay)  # wait before retry
                else:
                    return idx, None, f"Error ({type(e).__name__}): {e}"

    except Exception as e:
        return idx, None, f"Fatal error: {e}"

# Prepare URLs
urls_with_index = list(enumerate(df["imageurl"]))

# Run threaded download
max_workers = 50
results = []
errors = []

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(download_image, idx, url) for idx, url in urls_with_index]

    for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading images"):
        idx, filepath, error = future.result()
        if filepath:
            df.at[idx, "local_image_path"] = filepath
        if error:
            errors.append((idx, df.loc[idx, "imageurl"], error))

# Save updated DataFrame
df.to_csv("updated_dataset.csv", index=False)

# Errors 
print(f"✅ Downloaded {df['local_image_path'].notna().sum()}/{len(df)} images")
if errors:
    print(f"❌ {len(errors)} errors. First few:\n", errors[:5])
    pd.DataFrame(errors, columns=["index", "url", "error"]).to_csv("failed_downloads.csv", index=False)