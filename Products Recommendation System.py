import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def get_title_from_index(index):
    return df[df.index == index]["ProductTitle"].values[0]


def get_index_from_title(title):
    return df[df.ProductTitle == title]["index"].values[0]


# Step 1: Read CSV File
df = pd.read_csv("product_sample.csv")
# print (df.columns)

# Step 2: Select Features
features = ['ProductsCategory', 'ProductTitle',
            'ProductDescription', "Brand"]  # use
# to remove NaN error - null values
for feature in features:
    df[feature] = df[feature].fillna('')

# Step 3: Create a column in DF which combines all selected features
def combine_features(row):
    # try block to see in my rows are combined or not
    # and if not printing that row
    try:
        return row['ProductsCategory']+" "+row['ProductTitle']+" "+row['ProductDescription']+" "+row['Brand']
    except:
        print("Error:", row)


df["combined_features"] = df.apply(
    combine_features, axis=1)  # axis is to combine row only
# print("Combined Features:", df['combined_features'].head())

# Step 4: Create count matrix from this new combined column
cv = CountVectorizer()
count_matrix = cv.fit_transform(df["combined_features"])

# Step 5: Compute the Cosine Similarity based on the count_matrix
cosine_sim = cosine_similarity(count_matrix)

# Step 6: get index of the product from user bought product
# product_user_buy = "Flipkart Supermart Select Soya Bean(500 g)"
# product_user_buy = "Cinthol Dive Deodorant Spray - For Men(150 ml)"
product_user_buy = "Maybelline New York Color Sensational Powder Matte Lipstick(Cherry Chic, 3.9 g)"
product_index = get_index_from_title(product_user_buy)

# Step 7: Get a list of similar products in descending order of similarity score
similar_products = list(enumerate(cosine_sim[product_index]))
sorted_similar_products = sorted(
    similar_products, key=lambda x: x[1], reverse=True)

# Step 8: Print the names of first 5 products
print("Recommended Products for Product: ", product_user_buy)
i = 1
for product in sorted_similar_products[1:]:
    print(i, ": ", get_title_from_index(product[0]))
    i = i+1
    if i > 5:
        break
