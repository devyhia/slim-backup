import requests
import argparse
from tqdm import tqdm
from IPython.terminal.embed import embed

parser = argparse.ArgumentParser()
parser.add_argument('file', help='File name to download from Google Drive ...')
parser.add_argument('saveas', help='Name of file to be saved ...')

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def get_size(size):
    Kb = pow(2,10)
    Mb = pow(2,20)
    Gb = pow(2,30)
    Tb = pow(2,40)

    if size < Kb:
        return size, "Bytes"
    elif size < Mb:
        return size / Kb, "KB"
    elif size < Gb:
        return size / Mb, "MB"
    elif size < Tb:
        return size / Gb, "GB"
    else:
        return size / Tb, "TB"

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    Size = 0

    with open(destination, "wb") as f:
        pb = tqdm(desc=args.saveas, unit="Chunk", miniters=1)
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
                Size += len(chunk)
                val, unit = get_size(Size)
                pb.update(1)
                pb.set_description("{} ({:.1f} {})".format(args.saveas, val, unit))


if __name__ == "__main__":
    args = parser.parse_args()
    download_file_from_google_drive(args.file, args.saveas)
https://drive.google.com/open?id=0B4aAfZAusJeuYm0zT21VU0MzSGc
