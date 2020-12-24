import hashlib

BUFFER_SIZE = 2 ** 16


def sha1_check(file_name, sha1_match):
    sha1 = hashlib.sha1()

    with open(file_name, 'rb') as f:
        while True:
            data = f.read(BUFFER_SIZE)
            if not data:
                break
            sha1.update(data)

    return sha1_match == sha1
