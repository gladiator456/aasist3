"""
AASIST3
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
 
import os
import sys
 
if __name__ == "__main__":
    url = "https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip?sequence=3&isAllowed=y"
 
    # -L: follow redirects (this host redirects to the actual file storage)
    # -f: fail loudly on HTTP errors instead of saving an error page as LA.zip
    cmd = f'curl -L -f -o ./LA.zip -# "{url}"'
    ret = os.system(cmd)
    if ret != 0:
        sys.exit("Download failed — check your internet connection / Kaggle 'Internet: On' setting.")
 
    ret = os.system("unzip -q ./LA.zip")
    if ret != 0:
        sys.exit("Unzip failed — the downloaded file may be incomplete or corrupted.")
 
    print("Dataset ready at ./LA")
 
