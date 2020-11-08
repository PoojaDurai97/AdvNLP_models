# AdvNLP Assessment
## Notebooks

Below are the paths of the various notebooks:

1.Demo notebook that contains the similarity outputs displayed for four URLs of choice : https://colab.research.google.com/drive/1nAwi0r8fWy0zN8rJHpUobeU4OuOzkRli?usp=sharing

2.Notebook that contains T5 base model fine tuned on news summary data ; https://colab.research.google.com/drive/1hOZtUmf0f8B1wv9FdxLJ7gYqXswXPrKa?usp=sharing

3.Notebook that contains ALBERT base v2 model finetuned on sentence transformer framework using NLI dataset : https://colab.research.google.com/drive/1SWpfeYhhHcAb5y8hyxW1voL9QJYCRWgv?usp=sharing

## Demo notebook

The demo notebook displays the similarity metrics output for various sentence embedding methods for both with and without summarization of reference and input URLs content.

The demo notebook has the following functions:

### content_similarity(url,id):

I have taken 4 input URLs to check similarity with the reference URL.I have created parsers using beautiful soup to scrape the articles from these URLS.Since these websites have ads and texts from other links,I have removed them.Since I have created separate parsers I am identifying the urls with the help of id so that I can pass them into the respective function and scrape.The content_similarity function takes two arguments:url and id.url is the url of choice and id is its corresponding id

id 1:https://www.republicworld.com/india-news/politics/dissenting-letter-brainchild-of-2-congress-functionaries-who-are-not-s.html

id 2:https://www.timesnownews.com/sports/cricket/article/chennai-super-kings-vs-mumbai-indians-head-to-head-record-important-stats-ahead-of-csks-must-win-tie/671418

id 3:https://scroll.in/latest/971665/letter-to-sonia-gandhi-kerala-congress-chief-steps-in-after-state-legislator-attacks-shashi-tharoor

id 4:https://www.india.com/entertainment/filmmaker-sudarshan-rattan-passes-away-due-to-covid-19-bollywood-mourns-4200504/

This is the main function which contains several sub functions inside it.

### given_url_parser():

This function parses article from the reference URL https://www.ndtv.com/india-news/details-of-dissent-letter-to-sonia-gandhi-steady-decline-no-honest-inspection-2286399

### input_url_parser():

Parsing text from the input URL of choice passsed into the function.Based on the corresponding id,the input URLs are scraped 

### text_summarization(text):

Performs summarization of text.Loading the T5 base model and tokenizer fine tuned on news summary data from  drive and generating summary.I have custom trained the model in this notebook https://colab.research.google.com/drive/1hOZtUmf0f8B1wv9FdxLJ7gYqXswXPrKa?authuser=1

### sentence_embeddings(text):

Obtaining sentence embeddings through four methods

METHOD 1:ALBERT base v2 model finetuned on sentence transformer framework using NLI dataset (Notebook:https://colab.research.google.com/drive/1SWpfeYhhHcAb5y8hyxW1voL9QJYCRWgv?authuser=1#scrollTo=_YZvXF7ejH6C)

METHOD 2:Universal sentence encoder(pre trained)

METHOD 3:sent2vec (pre trained)

METHOD 4:DistilBERT on sentence transformer (pre trained)

### def similarity_metrics(embed1,embed2,txt1,txt2):

Calculation of various similarity metrics.Passing the reference url text and input url text along with their embeddings as arguments.

The various similarity metrics are,

	Word mover’s distance

	Jaccard similarity

	Cosine similarity

	Manhattan distance

	Euclidean distance

	Inner product

	Theta

	Magnitude difference

	Triangle area similarity

	Sector area similarity

	Triangle sector similarity(TS-SS)

## Finetuned ALBERT base v2 on Sentence transformer framework notebook

I have trained the ALBERT base v2 model from HuggingFace transformers on sentence transformer framework (Siamese BERT) using NLI dataset.This framework allows an easy fine-tuning of custom embeddings models, to achieve maximal performance on your specific task.The dataset contains sentence pairs and one of three labels: entailment, neutral, contradiction.The model trains ALBERT base v2 model on the SNLI and MultiNLI (AllNLI) dataset with softmax loss function. At every 1000 training steps, the model is evaluated on the STS benchmark dev dataset and finally the model is tested on STS benchmark test dataset.I have trained the model for 41,000 steps which took around 5 hours.The Pearson and Spearman coefficient of cosine similarity, Manhattan distance, Euclidean distance and dot product are calculated on the STSb dev dataset for each 1000 training steps which can be seen in the similarity evaluation STS dev results file.And similarity evaluation STS test results file shows a correlation coefficient of 0.74.
The results of the training and evaluation are present in two excel files

## Finetuned T5 base on news summary data notebook

I have fine-tuned a transformer model for summarization task. A summary of a given article/document is generated when passed through the network. There are 2 types of summary generation mechanisms-extractive and abstractive summary. This model gives out abstractive summary.I have used the T5 base model from HuggingFace transformers to fine tune on the news summary data available in Kaggle.I have used 80% of the data for training and 20% for evaluation




