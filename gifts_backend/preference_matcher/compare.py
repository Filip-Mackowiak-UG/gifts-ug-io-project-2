import json
import numpy as np
import matplotlib.pyplot as plt
import os


def create_dir_if_not_exist(filename):
    dir_name = os.path.dirname(filename)
    if dir_name:  # Only create directory if dir_name is not empty
        os.makedirs(dir_name, exist_ok=True)


def load_probabilities(product_id, category):
    filename = f"./preference_matcher/product_probabilities/{product_id}_{category}_probabilities.json"
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"File not found: {filename}")
        return None


def calculate_combined_probabilities(products, input_preferences):
    combined_probabilities = {}

    for product in products:
        product_id = product['id']
        combined_probability = 1.0
        valid_product = True

        for category, preference_value in input_preferences.items():
            saved_probabilities = load_probabilities(product_id, category)
            if not saved_probabilities:
                combined_probability *= 0.0
                valid_product = False
                break

            preferences = saved_probabilities["preferences"]
            probabilities = saved_probabilities["probabilities"]
            if preference_value in preferences:
                index = preferences.index(preference_value)
                combined_probability *= probabilities[index]
            else:
                combined_probability *= 0.0
                valid_product = False
                break

        if valid_product:
            combined_probabilities[product['name']] = combined_probability

    total_prob = sum(combined_probabilities.values())

    if total_prob == 0:
        print("Total probability is zero. No valid products found.")
        return combined_probabilities

    for product_name in combined_probabilities:
        combined_probabilities[product_name] /= total_prob

    return combined_probabilities


def plot_probabilities(product_names, probabilities_list, title="Probability of Each Product Being the Best Gift",
                       filename="probabilities.png"):
    create_dir_if_not_exist(filename)

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

    for bar, prob in zip(bars, sorted_probabilities_list):
        plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f'{prob:.4f}',
                 va='center', ha='left', fontsize=8)

    plt.tight_layout()
    plt.savefig(filename)


def get_recommendations(input_preferences):
    # Load input preferences (if empty load test ones)
    if not input_preferences:
        with open("input_preferences.json", "r") as f:
            input_preferences = json.load(f)

    # Load product information
    products = json.load(open("./preference_matcher/data/products.json"))
    print(products)

    # Calculate combined probabilities
    combined_probabilities = calculate_combined_probabilities(products, input_preferences)

    # Combine product details with their probabilities
    product_recommendations = []
    for product in products:
        product_name = product['name']
        if product_name in combined_probabilities:
            product_with_prob = product.copy()
            product_with_prob['probability'] = combined_probabilities[product_name]
            product_recommendations.append(product_with_prob)

    return product_recommendations


if __name__ == '__main__':
    # Load input preferences
    with open("input_preferences.json", "r") as f:
        input_preferences = json.load(f)

    # Load product information
    products = json.load(open("./data/products.json"))

    # Calculate combined probabilities
    combined_probabilities = calculate_combined_probabilities(products, input_preferences)

    # Check if combined probabilities are valid
    if combined_probabilities:
        # Print combined probabilities for visual representation
        print("Combined Probabilities for Products:")
        for product_name, probability in combined_probabilities.items():
            print(f"Product: {product_name}, Probability: {probability:.4f}")

        # Save the combined probabilities to a JSON file
        output_filename = "combined_probabilities.json"
        create_dir_if_not_exist(output_filename)
        with open(output_filename, "w") as f:
            json.dump(combined_probabilities, f, indent=4)

        # Plot the combined probabilities
        product_names = list(combined_probabilities.keys())
        probabilities_list = list(combined_probabilities.values())
        plot_probabilities(product_names, probabilities_list, title="Combined Probabilities for All Products",
                           filename="combined_probabilities.png")

        print("Combined probabilities saved and plotted successfully.")
    else:
        print("No valid combined probabilities to save or plot.")
