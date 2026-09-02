#!/usr/bin/env python3
"""
Run this once after `uv sync` to download required model/lexicon data.

    uv run python setup_data.py
"""

import subprocess
import sys

print("Downloading spaCy English model...")
# uv doesn't use pip; install the spaCy model as a package directly
subprocess.run(
    [
        "uv",
        "pip",
        "install",
        "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl",
    ],
    check=True,
)

print("\nDownloading NLTK data...")
import nltk

nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")

print("\nDownloading Moral Foundations lexicon...")
import os
import urllib.request

mft_url = "https://raw.githubusercontent.com/medianeuroscience/mft/master/dictionaries/mfd2.0.dic"
os.makedirs("lexicons", exist_ok=True)
try:
    urllib.request.urlretrieve(mft_url, "lexicons/mfd2.dic")
    print("MFT lexicon saved to lexicons/mfd2.dic")
except Exception as e:
    print(f" Could not auto-download MFT lexicon: {e}")
    print("Falling back to bundled minimal MFT lexicon (see toi_bias_analyzer.py)")

print("\nSetup complete. Run: uv run python toi_bias_analyser.py")
