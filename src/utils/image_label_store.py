import csv
import os
from typing import Dict, Optional

LABELS_CSV_PATH = os.path.join("static", "images", "labels.csv")


# labels.csv에 label을 저장(추가 또는 업데이트)
def save_label(filename: str, label: str):
    labels = load_labels()
    labels[filename] = label
    os.makedirs(os.path.dirname(LABELS_CSV_PATH), exist_ok=True)
    with open(LABELS_CSV_PATH, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "label"])
        for fname, lbl in labels.items():
            writer.writerow([fname, lbl])


# labels.csv에서 label을 조회
def get_label(filename: str) -> Optional[str]:
    labels = load_labels()
    return labels.get(filename)


# labels.csv 전체를 dict로 반환
def load_labels() -> Dict[str, str]:
    labels = {}
    if os.path.exists(LABELS_CSV_PATH):
        with open(LABELS_CSV_PATH, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                labels[row["filename"]] = row["label"]
    return labels
