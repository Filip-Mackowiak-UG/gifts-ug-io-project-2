from transformers import BertTokenizer, BertModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from model_training import utils
import nltk
from nltk.corpus import stopwords
import os


# Choose the test data
USER_PREFERENCE_ID = 1

# Initialize the tokenizer and model_training

# MUST DOWNLOAD FIRST TIME
# nltk.download('stopwords')
# nltk.download('punkt')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


def create_dir_if_not_exist(filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)


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


def plot_probabilities_old(product_names, probabilities_list, filename="probabilities.png"):
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
    plt.savefig(filename)


def plot_probabilities(product_names, probabilities_list, title="Probability of Each Product Being the Best Gift", filename="probabilities.png"):
    create_dir_if_not_exist(filename)
    plt.figure(figsize=(20, 10))
    plt.barh(product_names, probabilities_list, color='skyblue')
    plt.xlabel('Probability', fontsize=10)
    plt.title(title, fontsize=12)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(filename)


def plot_embeddings(embedding_matrix, product_names, filename="reduced_embeddings.png"):
    create_dir_if_not_exist(filename)
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
    plt.savefig(filename)


def plot_similarity_distribution(similarity_scores, filename="probability_distribution.png"):
    plt.figure(figsize=(10, 6))
    sns.histplot(similarity_scores.flatten(), bins=20, kde=True)
    plt.title('Similarity Scores Distribution')
    plt.xlabel('Similarity Score')
    plt.ylabel('Frequency')
    plt.savefig(filename)


products = utils.read_json_file("../data/products.json")
preferences_list = utils.read_json_file("../data/preferences_minimal.json")["preferences"]

for preference in preferences_list:
    preference_options = preference["options"]
    preference_category_embeddings = create_embeddings(preference_options)
    plot_embeddings(preference_category_embeddings, preference_options,
                    f"./product_categories_data/embeddings/{preference['category']}/embeddings.png")
    for product in products:
        product_description = filter_tokens(product["description"])

        product_preference_similarity_scores = get_similarity_scores(product_description,
                                                                     preference_category_embeddings)
        probabilities = calculate_probabilities(product_preference_similarity_scores)
        plot_probabilities(preference_options, probabilities, f"Probability of product: {product['id']} matching given categories",
                           f"./product_categories_data/{product['id']}/{preference['category']}/prob.png")
