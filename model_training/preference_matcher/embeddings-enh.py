from transformers import BertTokenizer, BertModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
import nltk
from nltk.corpus import stopwords
import os
from model_training import utils
import csv
import json

# Choose the test data
USER_PREFERENCE_ID = 1

# Initialize the tokenizer and model
nltk.download('stopwords')
nltk.download('punkt')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


def create_dir_if_not_exist(filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)


def filter_tokens(description):
    stop_words = set(stopwords.words('english'))
    tokens = nltk.tokenize.word_tokenize(description)
    addit_stopwords = [",", ".", " ", ";"]
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words and word.lower() not in addit_stopwords]
    return ' '.join(filtered_tokens)


def create_embeddings(items):
    embeddings = []
    for item in items:
        description = item
        inputs = tokenizer(description, return_tensors='pt', truncation=True, padding=True, max_length=512)
        outputs = model(**inputs)
        # Use CLS token embedding for better representation
        embedding = outputs.last_hidden_state[:, 0, :].detach().numpy()
        embeddings.append(embedding)
    return np.vstack(embeddings)


def get_similarity_scores(preferences_text, embedding_matrix):
    preferences_inputs = tokenizer(preferences_text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    preferences_outputs = model(**preferences_inputs)
    preferences_embedding = preferences_outputs.last_hidden_state[:, 0, :].detach().numpy()
    return cosine_similarity(preferences_embedding, embedding_matrix)


def recommend_product(similarity_scores, products):
    recommended_index = np.argmax(similarity_scores)
    return products[recommended_index]['name']


def calculate_probabilities(similarity_scores):
    exp_scores = np.exp(similarity_scores[0])
    return exp_scores / np.sum(exp_scores)


def plot_probabilities_v1(product_names, probabilities_list, filename="probabilities.png"):
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


def plot_probabilities_v2(product_names, probabilities_list, title="Probability of Each Product Being the Best Gift", filename="probabilities.png"):
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


def plot_probabilities(product_names, probabilities_list, title="Probability of Each Product Being the Best Gift",
                       filename="probabilities.png"):
    create_dir_if_not_exist(filename)

    # Sort probabilities and product names in descending order
    sorted_indices = np.argsort(probabilities_list)[::-1]
    sorted_product_names = [product_names[i] for i in sorted_indices]
    sorted_probabilities_list = [probabilities_list[i] for i in sorted_indices]

    plt.figure(figsize=(20, 10))
    bars = plt.barh(sorted_product_names, sorted_probabilities_list, color='skyblue')
    plt.xlabel('Probability', fontsize=10)
    plt.title(title, fontsize=12)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.gca().invert_yaxis()

    # print(title)

    # Add text annotations on the bars
    for bar, prob in zip(bars, sorted_probabilities_list):
        # print(prob)
        plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f'{prob}',
                 va='center', ha='left', fontsize=8)

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


def save_probabilities_to_file(product_id, category, preference_options, probabilities,
                               output_dir="./product_probabilities"):
    # Ensure the directory exists
    create_dir_if_not_exist(output_dir)

    # Create a dictionary to save probabilities
    probabilities_data = {
        "product_id": product_id,
        "category": category,
        "preferences": preference_options,
        "probabilities": probabilities.tolist()
    }

    # Save the probabilities to a JSON file
    with open(f"{output_dir}/{product_id}_{category}_probabilities.json", "w") as f:
        json.dump(probabilities_data, f, indent=4)

    # Save the probabilities to a CSV file
    with open(f"{output_dir}/{product_id}_{category}_probabilities.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Preference", "Probability"])
        for pref, prob in zip(preference_options, probabilities):
            writer.writerow([pref, prob])


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

        save_probabilities_to_file(product['id'], preference['category'], preference_options, probabilities)

        plot_probabilities(preference_options, probabilities, f"Probability of product: {product['id']} matching given categories",
                           f"./product_categories_data/{product['id']}/{preference['category']}/prob.png")
