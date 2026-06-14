# 🚀 AI & Machine Learning Internship Projects II

Welcome to the **Internship-II** portfolio repository! This directory contains a collection of diverse, production-ready machine learning and artificial intelligence projects. Each project demonstrates proficiency in modern deep learning architectures, natural language processing (NLP), computer vision, and end-to-end machine learning pipelines.

Below is an overview of the projects included in this repository. Please explore their respective directories for comprehensive code, Jupyter notebooks, and detailed documentation.

---

## 📂 Project Directory

### 1. 🎫 [Support Ticket Auto-Tagging Pipeline: Prompting vs. Fine-Tuning](./Auto_Tagging_Support_Tickets_Using_LLM/)
**Objective**: Automate IT and customer support operations by categorizing and routing incoming support tickets. 
- **Methodology**: Compares modern LLM paradigms by evaluating **Zero-Shot/Few-Shot Prompt Engineering** (using `Flan-T5`) against **Supervised Fine-Tuning (SFT)** (using `DistilBERT`) for top-3 ticket tagging. 
- **Highlights**: Hugging Face Transformers, LLM Prompting, Supervised Fine-Tuning.

### 2. 📰 [AG News Topic Classifier Using BERT](./Bert_News_Classifier/)
**Objective**: Build a robust Natural Language Processing (NLP) pipeline to classify news headlines into four distinct categories (World, Sports, Business, Sci/Tech).
- **Methodology**: Utilizes transfer learning by fine-tuning a pre-trained `bert-base-uncased` model on the AG News dataset. Includes the preparation of a lightweight Gradio demo for live interaction.
- **Highlights**: BERT, Sequence Classification, Transfer Learning, Hugging Face `Trainer`.

### 3. 💬 [Context-Aware Chatbot Using RAG](./Context_Aware_Chatbot/)
**Objective**: Develop a conversational AI assistant featuring Retrieval-Augmented Generation (RAG) that preserves dialogue memory and retrieves domain-specific knowledge.
- **Methodology**: Orchestrates the RAG pipeline using **LangChain**. Embeds a custom knowledge base using TF-IDF, applies similarity thresholding to prevent hallucinations, and grounds responses using a fine-tuned `google/flan-t5-small` model.
- **Highlights**: LangChain, RAG, Conversational Memory, Streamlit Deployment.

### 4. 🏡 [Multimodal Housing Price Prediction](./Home_Price_Prediction/)
**Objective**: Construct a deep learning architecture combining structured tabular features with unstructured visual data (house images) to accurately predict housing prices.
- **Methodology**: Processes tabular data with a deep feedforward neural network and processes property images using a frozen, pre-trained `ResNet18`. Features from both modalities are concatenated into a `MultiNet` fusion head for regression.
- **Highlights**: PyTorch, Multimodal Deep Learning, Convolutional Neural Networks (CNNs), Data Fusion.

### 5. 📈 [Telco Customer Churn Prediction Pipeline](./predicting_customer_churn/)
**Objective**: Mitigate subscriber attrition in telecom services by predicting the likelihood of customer churn.
- **Methodology**: Implements a robust, end-to-end `scikit-learn` pipeline. Features data preprocessing (ColumnTransformer), comparative modeling (Logistic Regression vs. Random Forests), and cross-validated hyperparameter optimization using GridSearchCV.
- **Highlights**: Scikit-Learn Pipelines, Classification Imbalance, Joblib Serialization, Tabular Machine Learning.

---

## 🛠️ Technologies & Frameworks Utilized
Across these projects, a variety of industry-standard tools and frameworks were utilized:
- **Languages**: Python (3.9+)
- **Deep Learning**: PyTorch, Hugging Face Transformers
- **Machine Learning**: Scikit-Learn, Pandas, NumPy
- **Deployment & UI**: Streamlit, Gradio
- **Serialization**: Joblib, PyTorch weights (`.pt`)

---
**Developed during the AI/ML Internship at DevelopersHub Corporation.**
