# 📰 AG News Topic Classifier Using BERT

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)

> **"Fine-tuning `bert-base-uncased` on the AG News dataset to classify news headlines into topic categories, evaluate with accuracy and F1-score, and prepare a lightweight Gradio demo for live interaction."**

---

## 📖 1. Objective of the Task
The primary objective of this project is to build a robust Natural Language Processing (NLP) pipeline that classifies news headlines into four distinct categories: **World**, **Sports**, **Business**, and **Sci/Tech**. By leveraging the power of Transfer Learning, we fine-tune a pre-trained BERT model to achieve high accuracy and make the model ready for deployment via an interactive web interface.

---

## ⚙️ 2. Methodology / Approach
Our approach is implemented in the `AG_News_BERT_Classifier.ipynb` notebook and follows a structured deep learning pipeline:

- **Data Loading & Inspection**: We utilize the Hugging Face `datasets` library to download the AG News dataset. We perform exploratory data analysis (EDA) to verify class balance and analyze headline lengths.
- **Tokenization & Preprocessing**: We use the `AutoTokenizer` associated with `bert-base-uncased` to tokenize the text, applying truncation and padding up to a maximum length of 128 tokens using a `DataCollatorWithPadding`.
- **Model Fine-Tuning**: We load `AutoModelForSequenceClassification` and configure a Hugging Face `Trainer` to fine-tune the model on the training set. We define hyperparameters such as learning rate, batch size, epochs, and weight decay to optimize training.
- **Evaluation Strategy**: We evaluate the model on the test set using standard NLP metrics, primarily Accuracy, F1-score, and a Confusion Matrix.
- **Deployment Preparation**: The final step involves saving the trained model and tokenizer locally so they can be seamlessly integrated into a Gradio UI for real-time inference.

---

## 📊 3. Key Results or Observations
- **Perfect Class Balance**: The exploratory data analysis showed that the AG News training dataset is perfectly balanced with exactly 30,000 samples per class, providing an ideal foundation for unbiased classifier training.
- **High Efficacy of BERT**: The pre-trained BERT embeddings allow the model to capture deep semantic context within short text spans (news headlines), resulting in strong predictive capabilities.
- **Production-Ready Artifacts**: The pipeline efficiently saves the model and tokenizer to a dedicated directory (`ag_news_bert`), making the system modular and fully prepared for downstream deployment or continuous integration.
