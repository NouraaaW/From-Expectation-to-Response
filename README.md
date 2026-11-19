# From-Expectation-to-Response

## Overview

When one speaker asks a question or makes a request, the utterance creates an expectation about what kind of reply should follow (in Conversation Analysis, the first part of an *adjacency pair* makes a particular second part relevant). Replies do not always meet that expectation. This repository contains an NLP project on dialogue pragmatics that makes this relation measurable: for adjacent utterance pairs from the DailyDialog corpus, three open-weight large language models (LLMs) annotate (1) what kind of response the previous utterance calls for, (2) what the response actually does, and (3) whether the expectation was fulfilled. The resulting labels are then used to train and evaluate classifiers that predict them from the text.

## Key Highlights & Methodology

- **Dataset:** 3,000 adjacent pairs from DailyDialog (Li et al., 2017), whose previous utterance is a Question (1,512) or a Directive (1,488) according to the original DailyDialog labels (used only for sampling, never shown to the models). Question and Directive are balanced; same-act vs. different-act responses are not (1,061 vs. 1,939), because the corpus contains few Directive-Directive pairs.
- **Annotation:** Two-stage annotation by Qwen3-14B, Gemma-3-12B-IT and Llama-3.1-8B-Instruct (4-bit), three sampled runs each. Stage 1 labels the Expected Response Function without seeing the response; Stage 2 labels Actual Function, Fulfillment and Mismatch Type given the model's own Stage-1 label.
- **Label aggregation:** Two-level majority voting (within and across models), at least 2 of 3 voters required. Only models that share the final expected function vote on Actual Function and Fulfillment. Agreement is measured with unanimity rates, Krippendorff's alpha and Cohen's kappa. The final dataset has **2,215 pairs from 1,958 dialogues** (73.8% of the 3,000).
- **Classification (TF-IDF baselines):** Logistic Regression, Linear SVM and Complement Naive Bayes (word and character n-gram TF-IDF) vs. majority and random baselines, evaluated with dialogue-grouped stratified 5-fold cross-validation; Macro-F1 is the main metric.

### Baseline results (Macro-F1, mean over 5 folds)

| Task | Best TF-IDF model | Macro-F1 | Majority baseline |
|---|---|---|---|
| Expected Response Function | Linear SVM | 0.690 | 0.156 |
| Actual Function | Logistic Regression | 0.608 | 0.115 |
| Fulfillment | Logistic Regression | 0.733 | 0.336 |

A classifier given only the response reaches 0.759 on Fulfillment, so TF-IDF features capture the previous-utterance/response relation only partly. Pretrained-model and LLM-based classifiers (in progress) will be evaluated with the same folds and metrics.

## Repository Contents

- `Data Collection/`: `Dataset_Collection.ipynb` and `dialogue_expectation_mismatch.csv` (the 3,000 collected pairs, label columns empty).
- `Dialogue Annotation/`: `LLM_Annotation.ipynb` and `llm_annotations_master.csv` (raw outputs of all models and runs).
- `Data Preparation/`: `Aggregation_and_Agreement.ipynb` and `final_annotated_dataset.csv` (final labels after voting).
- `Dialogue Classification/`: classification notebooks, which read `final_annotated_dataset.csv`.

Notebooks read files from their own folder, so copy the input CSV next to the notebook before running it.

## Labels

| Label | Meaning |
|---|---|
| Expected Response Function | Kind of response the previous utterance calls for (Answer, Accept/Reject, Perform/Provide, Clarification, Acknowledgment) |
| Actual Function | What the response actually does (the five above plus Information Request) |
| Fulfillment | Whether the response achieves the expected function (Fulfilled / Not Fulfilled) |
| Mismatch Type | How an unfulfilled response departs from the expectation (annotated; not analysed or classified because three of its four non-None categories have 28 or fewer examples) |

## Limitations

Labels are the consensus of three LLMs and have not been validated against human annotators; the final dataset keeps only pairs with a model majority.

## Tech Stack

- Language: Python
- LLM annotation: Hugging Face Transformers, bitsandbytes (4-bit), PyTorch; requires a GPU and access to the gated Gemma and Llama models
- Data manipulation & metrics: Pandas, NumPy, scikit-learn
- Visualization: Matplotlib

## Licence

DailyDialog is released under CC BY-NC-SA 4.0; the derived CSV files are for non-commercial research use under the same terms.

## Reference

Y. Li, H. Su, X. Shen, W. Li, Z. Cao and S. Niu, "DailyDialog: A manually labelled multi-turn dialogue dataset," IJCNLP 2017.
