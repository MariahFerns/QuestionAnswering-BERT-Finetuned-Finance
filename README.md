# Finetuning BERT for Question Answering in Finance

###### [Open model in Hugging Face](https://huggingface.co/Mariah64/distilbert-base-uncased-finetuned-squad-v2) 🤗


## Introduction
This project aims to develop a model capable of answering questions based on documents while enhancing content comprehension and generating new context to support the answers. This is particularly useful in educational tools or support systems where deeper understanding and additional context are required.

## Objective
- Develop a question answering model that can generate new content to support the answers.
- Use the SQuAD (Stanford Question Answering Dataset) for training and testing.
- Implement the model using HuggingFace datasets and transformers.

## Dataset Description
The dataset used in this project is the Stanford Question Answering Dataset (SQuAD). It provides a robust foundation for training question answering systems and can be augmented with generative tasks.
Additional data focused on financial information is used to further finetune the model for the financial domain.

## Steps and Tasks Performed
### 1. Download and Prepare the Dataset
- Download the SQuAD dataset from HuggingFace datasets.
- Split the dataset into training and testing sets.

### 2. Text Preprocessing
- Perform text preprocessing on the training data using an auto tokenizer.
- Explore the tokenized output to return the start and end positions of the answer from the context.

### 3. Hyperparameter Tuning
- Use the `hyperopt` library to find optimal parameters for number of epochs, batch size, learning rate, etc.

### 4. Model Fine-Tuning
- Fine-tune the `distilbert-base-uncased` model on the SQuAD dataset using TensorFlow.
- Save the model and results to the HuggingFace Hub using callbacks.
- Use MLflow to log parameters and metrics for experiment tracking.
- Utilize TensorBoard callback to log accuracy and loss graphs.

### 5. Model Evaluation
- Reprocess the validation data for evaluation.
- Make predictions on the validation data.
- Post-process predictions (which are in the form of probabilities) to extract the answer.

### 6. Evaluation Metrics
- Format the actual answers and predicted answers.
- Evaluate using the SQuAD evaluation metric.

### 7. Model Inference
- Use the saved model from the HuggingFace Hub to make inferences.
- Test with random data using the Pipeline API for quick inferencing.

### 8. Financial Data Integration
- Create financial data from the company's financial reports in the same format as the SQuAD data for training and testing.
- Fine-tune the model again on this financial data so that the model can answer questions frequently asked in the financial domain.

## Results and Findings
- The fine-tuned `distilbert-base-uncased` model showed promising results in question answering tasks.
- Evaluation metrics indicated a high level of accuracy in the model's predictions.
- The integration of financial data allowed the model to effectively answer domain-specific financial questions with accuracy.

## Conclusion
This project successfully developed a model capable of answering questions and generating new content to support the answers. The use of the SQuAD dataset and HuggingFace tools proved effective in training and evaluating the model. This system can be particularly useful in financial systems like banks where quick responses are required by the senior leadership. 

## Installation and Usage
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/MariahFerns/QuestionAnswering-BERT-Finetuned-Finance.git
   cd QuestionAnswering-BERT-Finetuned-Finance

