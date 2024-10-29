# Student name: Terence Chu
# Student number: 301220117

from sklearn.datasets import fetch_olivetti_faces
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import cv2

import warnings
warnings.filterwarnings("ignore", message="KMeans is known to have a memory leak")

# Load Olivetti faces dataset
olivetti = fetch_olivetti_faces()

print('Olivetti faces data shape', olivetti.data.shape) # 64 × 64
print('Olivetti faces target shape', olivetti.target.shape)

X = olivetti.data
y = olivetti.target

print('\nPixel values:\n', X)
print('Pixel maximum:', X.max())
print('Pixel minimum:', X.min())
print('Data is already normalized')

# Display the first 12 images of the dataset
plt.figure(figsize=(7,7))

for i in range(12):
    plt.subplot(3, 4, i+1) # 3 rows, 4 columns
    plt.imshow(X[i].reshape(64,64), cmap='gray')
    plt.xticks([])
    plt.yticks([])

plt.suptitle("First 12 images of the dataset", fontsize=16, y=0.9)
plt.show()

# Split the dataset into train, test, and validation sets
# stratify=y ensures the class distribution in each split set is the same as the original dataset
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=17)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, stratify=y_train_full, test_size=0.25, random_state=17)

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print('X_valid shape:', X_valid.shape)

# Create an instance of PCA and preserve 99% of the variance
pca = PCA(n_components=0.99)

# Apply PCA to the training data
X_train_reduced = pca.fit_transform(X_train)

print(f'\nShape of the reduced X_train dataset: {X_train_reduced.shape}') # PCA reduced data to 177 dimensions

# Display the first 12 images of the dimension-reduced X_train dataset
plt.figure(figsize=(7,7))

# Use inverse_transform to map the reduced data back to its original space
for i in range(12):
    plt.subplot(3, 4, i+1) # 3 rows, 4 columns
    plt.imshow(pca.inverse_transform(X_train_reduced)[i].reshape(64,64), cmap='gray')
    plt.xticks([])
    plt.yticks([])

plt.suptitle("First 12 images of the reduced X_train dataset", fontsize=16, y=0.9)
plt.show()

# According to Scikit-Learn documentation, GMM can use the following covariance types:
# Source: https://scikit-learn.org/dev/modules/generated/sklearn.mixture.GaussianMixture.html
# 1) full
# 2) tied
# 3) diag
# 4) spherical

covariance_types = ['tied', 'full', 'diag', 'spherical']

bic_values_cov = []

print('\nBIC vs. Covariance Type Analysis:')

# Loop through the four covariance types and instantiate and fit GaussianMixture models with each covariance type
# Calculate BIC and add it and the covariance type as a tuple to the bic_values_cov list
for covariance_type in covariance_types:
    gmm = GaussianMixture(n_components=50, covariance_type=covariance_type, reg_covar=1e-4, random_state=42)
    gmm.fit(X_train_reduced)
    bic = gmm.bic(X_train_reduced)
    print(f'bic_value: {bic} - covariance_type: {covariance_type}')
    bic_values_cov.append((bic, covariance_type))

# The most suitable covariance type is the one that mimimizes AIC or BIC (using BIC in this assignment)
minimum_bic_cov = min(bic_values_cov)
print(f'\nMinimum BIC value and most suitable covariance type: {minimum_bic_cov}')

# Extract the most suitable covariance type (second element of minimum_bic_cov tuple)
most_suitable_covariance = minimum_bic_cov[1]

x = []
y = []

# Extract x (covariance type) from the second element (index 1) of bic_value
# Extract y (BIC values) from the first element (index 0) bic_value
for bic_value in bic_values_cov:
    x.append(bic_value[1])
    y.append(bic_value[0])

plt.figure(figsize=(15,6))

# Plot the BIC values for the four covariance types
plt.bar(x, y)
plt.title('BIC Values of Different Covariance Types')
plt.xlabel('Covariance')
plt.ylabel('BIC (millions)')
plt.show()

bic_values_comp = []

print('\nBIC vs. n_components Analysis:')

# Get the BIC value for number of clusters between 2 and 50
for gmm_n_components in range(2, 50):
    gmm = GaussianMixture(n_components=gmm_n_components, covariance_type=most_suitable_covariance, reg_covar=1e-4, random_state=42)
    gmm.fit(X_train_reduced)
    bic = gmm.bic(X_train_reduced)
    print(f'bic: {bic}, n_components: {gmm_n_components}')
    bic_values_comp.append((bic, gmm_n_components))

# The minimum number of clusters is the one that mimimizes AIC or BIC (using BIC in this assignment)
minimum_bic_comp = min(bic_values_comp) 
print(f'\nMinimum BIC value and numbers of cluster: {minimum_bic_comp}')

# Extract the minimum number of clusters (second element of minimum_bic_comp tuple)
minimum_number_of_clusters = minimum_bic_comp[1]

x = []
y = []

# Extract x (covariance type) from the second element (index 1) of bic_value
# Extract y (BIC values) from the first element (index 0) bic_value
for bic_value in bic_values_comp:
    x.append(bic_value[1])
    y.append(bic_value[0])

plt.figure(figsize=(15,6))

# Plot the BIC values for different number of clusters
plt.plot(x, y, '-o')
plt.title('BIC Values of Different Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('BIC')
plt.show()

# Instantiate an instance of GaussianMixture and pass to it the most suitable covariance and minimum number of clusters
# most_suitable_covariance is 'spherical'
# minimum_number_of_clusters is 19 
gmm = GaussianMixture(n_components=minimum_number_of_clusters, covariance_type=most_suitable_covariance, reg_covar=1e-4, random_state=42)
gmm.fit(X_train_reduced)

# Use predict to get each image's cluster assignment
hard_clustering_assignment = gmm.predict(X_train_reduced)

print('\nHard Clustering Assignment')

# Print hard clustering assignment for each image
for i in range(len(hard_clustering_assignment)):
    print(f'Image index: [{i}] - Cluster assigned: {hard_clustering_assignment[i]}')

# Use predict_proba to get each image's probability of beloning to a cluster
soft_clustering_prob = gmm.predict_proba(X_train_reduced)

print('\nSoft Clustering Probability')

# Print soft clustering probability for each image
for i in range(len(soft_clustering_prob)):
    print(f'Image index: [{i}] - Cluster probability: {soft_clustering_prob[i]}')

number_of_images = 24

# Generate new images with the trained GMM model
generated_images, generated_images_class = gmm.sample(number_of_images)

print('\nShape of the generated images:', generated_images.shape)

# Transform the images to their original (not dimensionality reduced) space (i.e., 4096 dimensions)
generated_images_inversed_transformed = pca.inverse_transform(generated_images)

plt.figure(figsize=(25,7))

# Visualize generated images
for i in range(number_of_images):
    plt.subplot(3, 8, i+1) # 3 rows, 8 columns
    plt.imshow(generated_images_inversed_transformed[i].reshape(64,64), cmap='gray')
    plt.xticks([])
    plt.yticks([])

plt.show()

score_samples_orig_imgs = gmm.score_samples(generated_images)
print(f'\nScore samples output - original (unmodified) images:\n {score_samples_orig_imgs}')

start = 0
stop = number_of_images
step = 5

# Modfiy images
# Studied OpenCV documentation for cv2 image transformation functions
# Source: https://docs.opencv.org/3.4/d2/de8/group__core__array.html
for i in range(start, stop, step):
    selected_image = generated_images_inversed_transformed[i].reshape(64,64)

    if (i == 0): 
        modified_image = cv2.rotate(selected_image, cv2.ROTATE_90_CLOCKWISE) 
    elif(i == 5):
        modified_image = cv2.flip(selected_image, flipCode = 0) # flipCode 0 flips image on X-axis
    elif(i == 10):
        modified_image = cv2.rotate(selected_image, cv2.ROTATE_90_COUNTERCLOCKWISE) 
    elif(i == 15):
        modified_image = cv2.convertScaleAbs(selected_image, alpha=5, beta=10) # alpha affects the contrast while beta affects the brightness
    elif(i == 20):
        modified_image = cv2.convertScaleAbs(selected_image, alpha=5, beta=-10) 
        
    generated_images_inversed_transformed[i] = modified_image.flatten()

plt.figure(figsize=(25,7))

# Visualize generated images (now with five of them modified)
for i in range(number_of_images):
    plt.subplot(3, 8, i+1) # 3 rows, 4 columns
    plt.imshow(generated_images_inversed_transformed[i].reshape(64,64), cmap='gray')
    plt.xticks([])
    plt.yticks([])

plt.show()

print(f'\nShape of the modified images: {generated_images_inversed_transformed.shape}')

# The modified images have 4096 dimensions - must be reduced to 177 with PCA
score_samples_mod_imgs = gmm.score_samples(pca.transform(generated_images_inversed_transformed))
print(f'\nScore samples output - modified images:\n {score_samples_mod_imgs}')