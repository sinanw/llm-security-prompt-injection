# Security of Large Language Models (LLM) - Prompt Injection Classification
In this project, we investigate the security of large language models in terms of [prompt injection attacks](https://www.techopedia.com/definition/prompt-injection-attack). Primarily, we perform binary classification of a dataset of input prompts in order to discover malicious prompts that represent injections.

*In short: prompt injections aim at manipulating the LLM using crafted input prompts to steer the model into ignoring previous instructions and, thus, performing unintended actions.*

To do so, we analyzed several AI-driven mechanisms to do the classification task, we particularly examined 1) classical ML algorithms, 2) a pre-trained LLM model, and 3) a fine-tuned LLM model.


## Data Set (Deepset Prompt Injection Dataset)
The dataset used in this demo is: [Prompt Injection Dataset](https://huggingface.co/datasets/deepset/prompt-injections) provided by [deepset](https://www.deepset.ai/), an AI company specialized in offering tools to build NLP-driven applications using LLMs. <br/>
- The dataset combines hundreds of samples of both normal and manipulated prompts labeled as injections.
- It contains prompts mainly in English, along with some other prompts translated into other languages, primarily German.
- The original data set is already split into training and holdout subsets. We maintained this split across the multiple experiments to compare results using a unified testing benchmark.


### METHOD 1 - Classification Using Traditional ML
> Corresponding notebook:  [ml-classification.ipynb](https://github.com/sinanw/llm-security-prompt-injection/blob/main/notebooks/1-ml-classification.ipynb)

Analysis steps:
1. Loading the dataset from HuggingFace library and exploring it.
2. Tokenizing prompt texts and generating embeddings using the [multilingual BERT (Bidirectional Encoder Representations from Transformers)](https://huggingface.co/bert-base-multilingual-uncased) LLM model.
3. Training the following ML algorithms on the downstream prompt classification task:
    - [Naive Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)
    - [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
    - [Support Vector Machine](https://scikit-learn.org/stable/modules/svm.html)
    - [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
4. Analyzing and comparing the performance of classification models.
5. Investigating incorrect predictions of the best-performing model.

#### Results:
|                      | Accuracy | Precision | Recall   | F1 Score |
|----------------------|----------|-----------|----------|----------|
| Naive Bayes          | 88.79%   | 87.30%    | 91.67%   | 89.43%   |
| Logistic Regression  | 96.55%   | 100.00%   | 93.33%   | 96.55%   |
| Support Vector Machine | 95.69% | 100.00%   | 91.67%   | 95.65%   |
| Random Forest        | 89.66%   | 100.00%   | 80.00%   | 88.89%   |


### METHOD 2 - Classification Using a Pre-trained LLM (XLM-RoBERTa)
> Corresponding notebook:  [llm-classification-pretrained.ipynb](https://github.com/sinanw/llm-security-prompt-injection/blob/main/notebooks/2-llm-classification-pretrained.ipynb)

Analysis steps:
1. Loading the dataset from HuggingFace library.
2. Loading the pre-trained [XLM-RoBERTa](https://huggingface.co/xlm-roberta-large) model, the multilingual version of RoBERTa, the enhanced version of BERT, from HuggingFace library.
3. Using HuggingFace [zero-shot classification](https://huggingface.co/tasks/zero-shot-classification) pipeline and XLM-RoBERTa to perform prompt classification on the testing dataset (without fine-tuning).
4. Analyzing classification results and model performance.

#### Results:
|              | Accuracy | Precision | Recall   | F1 Score |
|--------------|----------|-----------|----------|----------|
| Testing Data | 55.17%   | 55.13%    | 71.67%   | 62.32%   |


### METHOD 3 - Classification Using a Fine-tuned LLM (XLM-RoBERTa)
> Corresponding notebook:  [llm-classification-finetuned.ipynb](https://github.com/sinanw/llm-security-prompt-injection/blob/main/notebooks/3-llm-classification-finetuned.ipynb)

Analysis steps:
1. Loading the dataset from HuggingFace library.
2. Loading the pre-trained [XLM-RoBERTa](https://huggingface.co/xlm-roberta-large) model, the multilingual version of RoBERTa, the enhanced version of BERT, from HuggingFace library.
3. Fine-tuning XLM-RoBERTa to perform prompt classification on the training dataset.
4. Analyzing the fine-tuning accuracy across 5 epochs on the testing dataset.
5. Analyzing the final model accuracy and its performance, and comparing it with previous experiments.

#### Results:
| Epoch | Accuracy | Precision | Recall   | F1      |
|-------|----------|-----------|----------|---------|
| 1     | 62.93%   | 100.00%   | 28.33%   | 44.16%  |
| 2     | 91.38%   | 100.00%   | 83.33%   | 90.91%  |
| 3     | 93.10%   | 100.00%   | 86.67%   | 92.86%  |
| 4     | 96.55%   | 100.00%   | 93.33%   | 96.55%  |
| 5     | 97.41%   | 100.00%   | 95.00%   | 97.44%  |











