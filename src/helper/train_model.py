import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

dataset_directory = './datasets' 
vectorizer = CountVectorizer()
classifier = MultinomialNB()

# Function to preprocess the data
def preprocess_data(text: str) -> str:
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation (assuming only basic punctuation)
    text = text.replace('.', '').replace(',', '').replace('?', '').replace('!', '')
    # Remove stop words if necessary
    stop_words = ENGLISH_STOP_WORDS
    text = ' '.join([word for word in text.split() if word not in stop_words])
    # Add any other preprocessing steps as needed
    return text

# Read the .txt file and process the data
def read_txt_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Preprocess the data
    processed_data = [preprocess_data(line.strip()) for line in lines]
    
    return processed_data

# Read and process the data from the .txt files
def read_data_from_files(directory):
    data = []
    labels = []
    
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        aspect = file_name.split('.')[0]  # Extract the aspect from the file name
        
        # Read and preprocess the data from the file
        processed_data = read_txt_file(file_path)
        
        # Add the processed data and labels to the main lists
        data.extend(processed_data)
        labels.extend([aspect] * len(processed_data))
    
    return data, labels

def train_model (dataset_directory: str):
    # Load and preprocess the data from the .txt files
    data, labels = read_data_from_files(dataset_directory)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=42)

    # Create feature vectors using the bag-of-words model
    X_train_vectors = vectorizer.fit_transform(X_train)
    X_test_vectors = vectorizer.transform(X_test)

    # Train a Naive Bayes classifier
    classifier.fit(X_train_vectors, y_train)

    # Predict the labels for the test set
    y_pred = classifier.predict(X_test_vectors)

    # Evaluate the classifier
    print(classification_report(y_test, y_pred))

train_model(dataset_directory)

def predict (question: str) -> str:
    preprocessed_input = preprocess_data(question)
    input_vector = vectorizer.transform([preprocessed_input])
    predicted_label = classifier.predict(input_vector)
    return predicted_label

