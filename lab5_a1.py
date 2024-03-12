import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# Load the dataset
file_path = "C:\\Users\\mahad\\OneDrive\\Desktop\\SEM 4 Courses\\ML\\Legality\\Datasets\\English_Abstractive_Embeddings_Fasttext.xlsx"
data = pd.read_excel(file_path)

# Assuming 'X' contains your feature vectors and 'y' contains your class labels
X = data.drop(columns=['Judgement Status'])  # Assuming 'Judgement Status' is the column containing class labels
y = data['Judgement Status']

# Perform the train-test split with 70% of the data for training and 30% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a kNN classifier with k=3
neigh = KNeighborsClassifier(n_neighbors=3)

# Train the classifier using the training set
neigh.fit(X_train, y_train)

# Predictions for training and test data
train_predictions = neigh.predict(X_train)
test_predictions = neigh.predict(X_test)

# Confusion matrix for training data
train_conf_matrix = confusion_matrix(y_train, train_predictions)

# Confusion matrix for test data
test_conf_matrix = confusion_matrix(y_test, test_predictions)

# Performance metrics for training data
train_precision = precision_score(y_train, train_predictions, average='macro')
train_recall = recall_score(y_train, train_predictions, average='macro')
train_f1_score = f1_score(y_train, train_predictions, average='macro')

# Performance metrics for test data
test_precision = precision_score(y_test, test_predictions, average='macro')
test_recall = recall_score(y_test, test_predictions, average='macro')
test_f1_score = f1_score(y_test, test_predictions, average='macro')

# Print confusion matrix and performance metrics
print("Confusion Matrix for Training Data:")
print(train_conf_matrix)
print("\nConfusion Matrix for Test Data:")
print(test_conf_matrix)

print("\nPerformance Metrics for Training Data:")
print("Precision:", train_precision)
print("Recall:", train_recall)
print("F1-Score:", train_f1_score)

print("\nPerformance Metrics for Test Data:")
print("Precision:", test_precision)
print("Recall:", test_recall)
print("F1-Score:", test_f1_score)
