"""Download and extract the ASVspoof 2019 LA dataset into ./LA."""
import os
import urllib.request
import zipfile

URL = "https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip?sequence=3&isAllowed=y"
DEST = os.path.abspath("./LA")
ARCHIVE = os.path.abspath("./LA.zip")
EXPECTED = (
    "ASVspoof2019_LA_train",
    "ASVspoof2019_LA_dev",
    "ASVspoof2019_LA_cm_protocols",
)


def main():
    os.makedirs(DEST, exist_ok=True)
    if not os.path.isfile(ARCHIVE):
        print("Downloading ASVspoof 2019 LA dataset...")
        try:
            urllib.request.urlretrieve(URL, ARCHIVE)
        except Exception as exc:
            raise SystemExit(
                f"Download failed: {exc}\nIf running on Kaggle, enable Internet or add the dataset as an input."
            ) from exc

    if not zipfile.is_zipfile(ARCHIVE):
        raise SystemExit(f"{ARCHIVE} is not a valid ZIP archive; remove it and download again.")

    print(f"Extracting dataset into {DEST}...")
    try:
        with zipfile.ZipFile(ARCHIVE) as archive:
            dest_real = os.path.realpath(DEST)
            for member in archive.infolist():
                target = os.path.realpath(os.path.join(DEST, member.filename))
                if os.path.commonpath([dest_real, target]) != dest_real:
                    raise RuntimeError(f"Unsafe ZIP path: {member.filename}")
            archive.extractall(DEST)
    except Exception as exc:
        raise SystemExit(f"Extraction failed: {exc}") from exc

    for current, dirs, _ in os.walk(DEST):
        if all(name in dirs for name in EXPECTED):
            print(f"Dataset extracted. Root: {current}")
            print("The training script can usually auto-detect this directory.")
            return
    if all(os.path.isdir(os.path.join(DEST, name)) for name in EXPECTED):
        print(f"Dataset extracted. Root: {DEST}")
        print("The training script can usually auto-detect this directory.")
        return
    raise SystemExit(
        "Extraction finished, but expected ASVspoof folders were not found. "
        "Inspect LA.zip and the extracted ./LA directory."
    )


if __name__ == "__main__":
    main()
