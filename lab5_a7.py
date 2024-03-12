import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load your data from an Excel file
# Replace 'your_data.xlsx' with the actual file path and column names
file_path = "C:\\Users\\mahad\\OneDrive\\Desktop\\SEM 4 Courses\\ML\\Legality\\Datasets\\English_Abstractive_Embeddings_Fasttext.xlsx"
data = pd.read_excel(file_path)

# Assuming 'X' contains your feature vectors and 'y' contains your class labels
X = data.drop(columns=['Judgement Status'])  # Assuming 'Judgement Status' is the column containing class labels
y = data['Judgement Status']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the kNN classifier
knn_classifier = KNeighborsClassifier()

# Specify the hyperparameter grid to search
param_dist = {'n_neighbors': range(1, 20)}

# Use RandomizedSearchCV for hyperparameter tuning
random_search = RandomizedSearchCV(knn_classifier, param_distributions=param_dist, n_iter=10, cv=5, scoring='accuracy', random_state=42)
random_search.fit(X_train, y_train)

# Print the best hyperparameters
print("Best Parameters: ", random_search.best_params_)

# Get the best kNN model with the tuned hyperparameters
best_knn_model = random_search.best_estimator_

# Evaluate the model on the test set
accuracy = best_knn_model.score(X_test, y_test)
print("Test Accuracy: {:.2f}%".format(accuracy * 100))
