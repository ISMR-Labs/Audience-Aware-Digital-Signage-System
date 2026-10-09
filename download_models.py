#!/usr/bin/env python3
"""Download the two models used by people_counter.py into ./models (about 4.5 MB total)."""
import io
import os
import sys
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")

YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
             "face_detection_yunet_2023mar.onnx")
SSD_URL = ("https://storage.googleapis.com/download.tensorflow.org/models/tflite/"
           "coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip")


def fetch(url):
    print(f"downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "pi-people-counter/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def main():
    os.makedirs(MODELS, exist_ok=True)
    yunet = os.path.join(MODELS, "face_detection_yunet_2023mar.onnx")
    if not os.path.exists(yunet):
        data = fetch(YUNET_URL)
        if len(data) < 100_000:
            sys.exit("YuNet download looks wrong (too small); check the URL / network")
        open(yunet, "wb").write(data)
    print(f"ok  {yunet}")

    ssd = os.path.join(MODELS, "detect.tflite")
    if "--faces-only" in sys.argv:
        print("skipping person detector (--faces-only)")
        return
    if not os.path.exists(ssd):
        data = fetch(SSD_URL)
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for name in ("detect.tflite", "labelmap.txt"):
                open(os.path.join(MODELS, name), "wb").write(z.read(name))
    print(f"ok  {ssd}")


if __name__ == "__main__":
    main()
