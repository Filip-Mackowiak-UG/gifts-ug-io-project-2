from transformers import BertTokenizer, BertModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from model_training import utils
import nltk
from nltk.corpus import stopwords

# Choose the test data
USER_PREFERENCE_ID = 1

# Initialize the tokenizer and model_training

# MUST DOWNLOAD FIRST TIME
# nltk.download('stopwords')
# nltk.download('punkt')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


def filter_tokens(description):
    stop_words = set(stopwords.words('english'))
    tokens = nltk.tokenize.word_tokenize(description)
    addit_stopwords = [",", ".", " ", ";"]
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    filtered_tokens = [word for word in filtered_tokens if word.lower() not in addit_stopwords]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


def create_embeddings(items):
    embeddings = []
    for item in items:
        description = item
        inputs = tokenizer(description, return_tensors='pt', truncation=True, padding=True, max_length=512)
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        embeddings.append(embedding)
    return np.vstack(embeddings)


def get_similarity_scores(preferences_text, embedding_matrix):
    preferences_inputs = tokenizer(preferences_text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    preferences_outputs = model(**preferences_inputs)
    preferences_embedding = preferences_outputs.last_hidden_state.mean(dim=1).detach().numpy()
    return cosine_similarity(preferences_embedding, embedding_matrix)


def recommend_product(similarity_scores, products):
    recommended_index = np.argmax(similarity_scores)
    return products[recommended_index]['name']


def calculate_probabilities(similarity_scores):
    return similarity_scores[0] / np.sum(similarity_scores)


def plot_probabilities(product_names, probabilities_list):
    sorted_indices = np.argsort(probabilities_list)[::-1]
    sorted_product_names = [product_names[i] for i in sorted_indices]
    sorted_probabilities_list = [probabilities_list[i] for i in sorted_indices]

    plt.figure(figsize=(20, 10))
    plt.barh(sorted_product_names, sorted_probabilities_list, color='skyblue')
    plt.xlabel('Probability', fontsize=10)
    plt.title('Probability of Each Product Being the Best Gift', fontsize=12)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("probabilities.png")


def plot_embeddings(embedding_matrix, product_names):
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embedding_matrix)

    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1])
    for i, txt in enumerate(product_names):
        plt.annotate(txt, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]))
    plt.title('Embeddings Visualization')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.grid(True)
    plt.savefig("reduced_embeddings.png")


def plot_similarity_distribution(similarity_scores):
    plt.figure(figsize=(10, 6))
    sns.histplot(similarity_scores.flatten(), bins=20, kde=True)
    plt.title('Similarity Scores Distribution')
    plt.xlabel('Similarity Score')
    plt.ylabel('Frequency')
    plt.savefig("probability_distribution.png")


products = utils.read_json_file("../../data/products.json")
filtered_descriptions = [filter_tokens(product["description"]) for product in products]  # failed experiment
# descriptions = [product["description"] for product in products]
product_embeddings = create_embeddings(filtered_descriptions)
print(product_embeddings.shape)
# The output (16, 768) indicates the shape of your embedding_matrix numpy array.
# Here, 16 represents the number of embeddings (or products) you have
# 768 is the dimensionality of each embedding vector.
# This means there are 16 different embeddings, each being a vector with 768 elements.
# print(embeddings)

user_preferences = utils.read_json_file("../../data/test_data/test_preferences.json")
selected_preference = next((pref for pref in user_preferences if pref['id'] == USER_PREFERENCE_ID), None)
print(selected_preference)
filtered_preference = filter_tokens(selected_preference["preferences"])
similarity_scores = get_similarity_scores(filtered_preference, product_embeddings)
print(similarity_scores)
probabilities = calculate_probabilities(similarity_scores)

best_gift = recommend_product(similarity_scores, products)
print("The best gift based on your preferences is:", best_gift)

product_names = [product['name'] for product in products]
plot_probabilities(product_names, probabilities)
plot_embeddings(product_embeddings, product_names)
plot_similarity_distribution(similarity_scores)
