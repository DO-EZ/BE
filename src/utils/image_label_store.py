import csv
import os
from typing import Dict, Optional

LABELS_CSV_PATH = os.path.join("static", "images", "labels.csv")


# labels.csv에 label을 저장(추가 또는 업데이트)
def save_label(filename: str, label: str):
    os.makedirs(os.path.dirname(LABELS_CSV_PATH), exist_ok=True)
    file_exists = os.path.exists(LABELS_CSV_PATH)
    with open(LABELS_CSV_PATH, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["filename", "label"])
        writer.writerow([filename, label])


# labels.csv 전체를 dict로 반환
def load_labels() -> Dict[str, str]:
    labels = {}
    if os.path.exists(LABELS_CSV_PATH):
        with open(LABELS_CSV_PATH, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                labels[row["filename"]] = row["label"]
    return labels


# labels.csv에서 label을 조회
def get_label(filename: str) -> Optional[str]:
    labels = load_labels()
    return labels.get(filename)
