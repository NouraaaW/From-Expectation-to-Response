# From-Expectation-to-Response

## Overview
This repository contains an NLP project on dialogue pragmatics. Using adjacent utterance pairs from the DailyDialog dataset, three open-weight LLMs annotate what kind of response a previous utterance calls for, what the response actually does, and whether the expectation was fulfilled. The resulting labels are then used to train and evaluate classifiers that predict them from the text.

## Key Highlights & Methodology
* **Dataset:** 3,000 Question/Directive pairs sampled from DailyDialog (Li et al., 2017), balanced across Question/Directive and same/different original dialogue act.
* **Annotation:** Two-stage annotation by Qwen3-14B, Gemma-3-12B-IT and Llama-3.1-8B-Instruct (4-bit), three runs each. Stage 1 labels the Expected Response Function without seeing the response; Stage 2 labels Actual Function, Fulfillment and Mismatch Type.
* **Label aggregation:** Two-level majority voting (within and across models) with at least 2 of 3 voters required; agreement measured with Krippendorff's α and Cohen's κ. The final dataset has 2,215 pairs from 1,958 dialogues.
* **Classification:** Predicting Expected Response Function, Actual Function and Fulfillment from the text, evaluated with dialogue-grouped stratified 5-fold cross-validation and Macro-F1 as the main metric.

## Repository Contents
* `Data Collection/`: `Dataset_Collection.ipynb` and `dialogue_expectation_mismatch.csv` (the 3,000 collected pairs, label columns empty).
* `Dialogue Annotation/`: `LLM_Annotation.ipynb` and `llm_annotations_master.csv` (raw outputs of all models and runs).
* `Data Preparation/`: `Aggregation_and_Agreement.ipynb` and `final_annotated_dataset.csv` (final labels after voting).
* `Dialogue Classification/`: classification notebooks, which read `final_annotated_dataset.csv`.

Notebooks read files from their own folder, so copy the input CSV next to the notebook before running it.

## Labels

| Label | Meaning |
|---|---|
| Expected Response Function | kind of response the previous utterance calls for (Answer, Accept/Reject, Perform/Provide, Clarification, Acknowledgment) |
| Actual Function | what the response actually does (the five above plus Information Request) |
| Fulfillment | whether the response achieves the expected function (Fulfilled / Not Fulfilled) |
| Mismatch Type | how an unfulfilled response departs from the expectation (annotated, not analysed or classified) |

## Tech Stack
* **Language:** Python
* **LLM annotation:** Hugging Face Transformers, bitsandbytes (4-bit), PyTorch; requires a GPU and access to the gated Gemma and Llama models
* **Data manipulation & metrics:** Pandas, NumPy, Scikit-Learn
* **Visualization:** Matplotlib

## Licence
DailyDialog is released under CC BY-NC-SA 4.0; the derived CSV files are for non-commercial research use under the same terms.

## Reference
Y. Li, H. Su, X. Shen, W. Li, Z. Cao and S. Niu, "DailyDialog: A manually labelled multi-turn dialogue dataset," IJCNLP 2017.
