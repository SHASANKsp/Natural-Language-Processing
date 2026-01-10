# ASD Text Classification Project
## Overview
This project explores **Autism Spectrum Disorder (ASD) classification from caregiver-written behavioral text**. The goal is to evaluate how effectively different machine learning paradigms—**classical ML vs deep learning**—can detect ASD-related signals from short natural-language descriptions of toddler behavior.

---

## Methods
### Deep Learning Track
* Pretrained word embeddings (frozen)
* Bidirectional LSTM for contextual modeling
* Attention mechanism for identifying salient behavioral phrases
* Binary classification (ASD vs non-ASD)
This approach captures **sequential and contextual language patterns** in caregiver narratives.

### Classical ML Track
* Bag-of-words / binary text features
* Minimal normalization
* Models such as Bernoulli Naive Bayes
This approach tests whether **lexical presence alone** is sufficient for classification and prioritizes interpretability.

---

## Key Findings
* **Bernoulli Naive Bayes outperformed LSTM and BiLSTM models** on this dataset.
* This indicates that ASD-related signals in caregiver text are largely **presence-based and lexical**, rather than dependent on sentence structure or word order.
* The result highlights the importance of **choosing models aligned with the data**, not defaulting to complex architectures.

---

## Interpretation

The findings suggest that:

* Caregiver narratives often contain explicit behavioral indicators.
* Simple, interpretable models can be highly effective for early screening tasks.
* Deep learning models may require larger, more nuanced datasets to provide additional value.

