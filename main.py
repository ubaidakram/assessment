import os
import warnings
from typing import Dict

from openfabric_pysdk.utility import SchemaUtil

from ontology_dc8f06af066e4a7880a5938933236037.simple_text import SimpleText

from openfabric_pysdk.context import Ray, State
from openfabric_pysdk.loader import ConfigClass
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
#Load the squad dataset from hugging face hub
squad_df = load_dataset("squad")
context=("Data science is a multidisciplinary field that uses scientific methods, processes, algorithms, and systems "
         "to extract insights and knowledge from structured and unstructured data. It encompasses a variety of "
         "techniques and approaches, including machine learning, statistical analysis, data visualization, "
         "and more. In the realm of machine learning, algorithms like linear regression, decision trees, "
         "support vector machines, and neural networks are commonly employed for tasks such as classification, "
         "regression, and clustering. Supervised learning involves training models on labeled data, "
         "while unsupervised learning deals with unlabeled data, seeking patterns and structures. The data science "
         "process typically involves problem definition, data collection, data cleaning, exploratory data analysis, "
         "feature engineering, modeling, evaluation, and deployment. Cross-validation is a technique used to assess "
         "model performance, and overfitting is a common challenge that can be addressed through proper model tuning. "
         "Feature engineering is a critical step in enhancing model performance, involving the creation and "
         "modification of features. The bias-variance tradeoff is an essential concept in machine learning, "
         "balancing the model's ability to fit training data with its ability to generalize to new data. Natural "
         "Language Processing (NLP) is a branch of artificial intelligence focused on enabling machines to "
         "understand, interpret, and generate human-like text. Decision trees, deep learning, and reinforcement "
         "learning are advanced topics within the data science and machine learning landscape. Visualization "
         "techniques such as scatter plots, bar charts, histograms, heatmaps, and interactive dashboards are employed "
         "to effectively communicate insights. Data scientists play a vital role in collecting, analyzing, "
         "and interpreting large volumes of data to inform business decisions. Common algorithms in unsupervised "
         "learning include k-means clustering, hierarchical clustering, principal component analysis (PCA), "
         "and t-distributed stochastic neighbor embedding (t-SNE). Ensemble learning methods, such as bagging and "
         "boosting, are used to improve overall model performance. Feature scaling is crucial in machine learning to "
         "ensure that numerical features have a similar scale. Challenges in handling big data include storage, "
         "processing speed, privacy concerns, data quality, and the need for scalable and distributed computing "
         "frameworks. Time series analysis. Data Science Overview: Data science, a multidisciplinary field, "
         "leverages scientific methods to extract insights from structured and unstructured data. Key steps include "
         "data collection from diverse sources, cleaning, and preprocessing. Exploratory Data Analysis (EDA) involves "
         "descriptive statistics and visualization. Feature engineering enhances model performance through new "
         "feature creation and dimensionality reduction. The modeling phase employs machine learning algorithms for "
         "tasks like classification and regression. Subfields within data science include: 1. Machine Learning: - "
         "Supervised Learning: Models learn from labeled data (input-output pairs). - Unsupervised Learning: Models "
         "identify patterns in unlabeled data. - Semi-Supervised and Reinforcement Learning: Other paradigms under "
         "machine learning. 2. Deep Learning: - Neural Networks: Complex structures inspired by the human brain, "
         "effective for tasks like image and speech recognition. - Deep Neural Architectures: Multiple layers of "
         "interconnected nodes for more intricate learning. 3. Computer Vision: - Involves algorithms for "
         "interpreting visual information from the world, used in image and video analysis. 4. Natural Language "
         "Processing (NLP): - Focuses on interactions between computers and human languages, enabling tasks like text "
         "analysis, sentiment analysis, and language translation. 5. Anomaly Detection: - Identifying unusual "
         "patterns or outliers in data that may indicate potential issues or interesting events. 6. Time Series "
         "Forecasting and Analysis: - Analyzing and predicting trends in time-ordered data, crucial for fields like "
         "finance, weather forecasting, and sales. 7. Data Preprocessing: - Cleaning and transforming raw data into a "
         "usable format, including handling missing values and outliers. 8. Feature Engineering: - Creating new "
         "features from existing data to improve model performance. 9. Feature Selection: - Identifying and using "
         "only the most relevant features to reduce model complexity and improve efficiency. In summary, data science "
         "subfields encompass machine learning, deep learning, computer vision, NLP, anomaly detection, time series "
         "forecasting, along with essential techniques like data preprocessing, feature engineering, and feature "
         "selection.")


long_context=context
# Load the pretrained model and tokenizer
model_checkpoint = "distilbert-base-cased-distilled-squad"
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
# Tokenize the context and the question using the given tokenizer.
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
# Split the long context into chunks
chunk_size = 512
chunks = [long_context[i:i + chunk_size] for i in range(0, len(long_context), chunk_size)]

# Initialize variables to store results
answers = []
############################################################
# Callback function called on update config
############################################################
def config(configuration: Dict[str, ConfigClass], state: State):
    # TODO Add code here
    pass


############################################################
# Callback function called on each execution pass
############################################################
def execute(request: SimpleText, ray: Ray, state: State) -> SimpleText:
    output = []
    for text in request.text:
        # Loop through each chunk
        for chunk in chunks:
            # Tokenize the chunk and question
            inputs = tokenizer(text, chunk, add_special_tokens=True, return_tensors="pt", truncation=True, max_length=512)

            # Get the model output
            outputs = model(**inputs)

            # Process the outputs as needed
            answer_start_index = torch.argmax(outputs.start_logits)
            answer_end_index = torch.argmax(outputs.end_logits) + 1

            # Extract the answer tokens from the input using start and end index
            predict_answer_tokens = inputs.input_ids[0, answer_start_index: answer_end_index]

            # Decode the answer tokens
            ans = tokenizer.decode(predict_answer_tokens)

            # Append the answer to the list
            answers.append(ans)

        # Combine results if needed
        response = " ".join(answers)
        output.append(response)

    return SchemaUtil.create(SimpleText(), dict(text=output))
