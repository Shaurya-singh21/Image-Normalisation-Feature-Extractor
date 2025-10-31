# Image Dataset Preprocessing & Feature Extractor

A Python script using OpenCV and Pandas to automate the preprocessing of an image dataset. This script reads a directory of labeled image folders, processes each image, and saves the results in new directories while also cataloging extracted features into a CSV file.

## Features

* **Iterates Subdirectories:** Automatically scans a main directory (e.g., `seg_train`) for labeled subfolders (e.g., `label1`, `label2`, ...).
* **Image Normalization:** Converts each image to grayscale and resizes it to a uniform **256x256** pixels.
* **Edge Detection:** Generates a 256x256 Canny edge map for each image.
* **Feature Extraction:** For each image, it calculates and records:
    * Overall brightness
    * Mean intensity for Red, Green, and Blue channels
    * Edge density score (a ratio of edge pixels)
* **CSV Catalog:** Saves all extracted metadata and file information into a single `image_data.csv` file for easy analysis.

---

