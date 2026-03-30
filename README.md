# 🛠️ BERT Bug Classifier — Preprocessing Studio

A lightweight offline GUI to prepare raw bug report datasets for BERT models. It focuses on preserving context while cleaning noisy real-world data.

---

## ✨ Key Features

This tool solves common issues in messy bug report datasets (e.g., from Kaggle or internal trackers) without destroying meaningful text.

### 🔹 Tiered Auto-Relabeling (v2)

A redesigned relabeling engine using strong and weak signal detection. It requires multiple weak keyword matches (e.g., *render*, *layout*, *font*) before changing labels—reducing false positives significantly.

### 🔹 Non-Destructive Cleaning

Removes only pure noise (like hex stack traces such as `x a3 ff b2`) while keeping meaningful words intact. No aggressive cleaning like removing numbers or URLs that could break context.

### 🔹 Zero Truncation

Descriptions are never cut mid-sentence. A dedicated view ensures complete text visibility and data integrity.

### 🔹 Built-in Data Splitting

Custom Train/Validation/Test splits with stratified sampling based on bug type.

### 🔹 Offline & Free

Runs fully locally using open-source Python libraries—no APIs, no subscriptions.

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/bert-bug-preprocessor.git
cd bert-bug-preprocessor
```

### 2. Install dependencies

```bash
pip install pandas scikit-learn
```

### 3. Run the application

```bash
python "BERT Bug Classifier — Preprocessing Studio.py"
```

---

## 📊 Expected Dataset Format

The tool expects a CSV file. Recommended columns:

| Column Name         | Description                 | Example                         |
| ------------------- | --------------------------- | ------------------------------- |
| Bug_Type            | Target classification label | Crash, UI/Visual, Network       |
| Severity            | Impact of the bug           | critical, major, trivial        |
| Fixing_time         | Resolution speed            | fast, medium, slow              |
| Cleaned_Description | Bug report text             | App hangs when clicking save... |

*Note: The tool handles missing columns gracefully.*

---

## 🖥️ Workflow Guide

### 📂 Load Data

Import your dataset using **Load CSV**.

### 🛠️ Clean & Fix

* Run **Clean Hex / Stack Noise**
* Use **Auto-Relabel Bug Types** to correct labels using keyword logic

### 🔍 Audit

Generate a **Label Audit Report** to:

* Check keyword match rates
* Detect class imbalance
* Identify short descriptions (< 5 words)

### ✏️ Edit

* Use table preview to manually edit cells
* Merge or refine labels interactively

### 📊 Encode & Split

* Generate label encoding (`_ID` columns)
* Export stratified Train / Validation / Test CSV files

---
---

## 🤝 Contributing
Varad D Kulkarni, Gargi Vyawahare  - Developers  and Researchers

---

## 📄 License

This project is licensed under the MIT License.
