from tqdm import tqdm
import requests

from tools.shasum_check import sha1_check


def fetch(url, target_file, sha1_hash=None):
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

    with open(target_file, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)

    progress_bar.close()

    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        Exception("Reeled in an unexpected amount of data.")

    if sha1_hash is not None and not sha1_check(target_file, sha1_hash):
        Exception(f"SHA1 hash for {target_file} does not match!")