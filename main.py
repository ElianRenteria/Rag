from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sqlite3
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.llms import Ollama
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize the Flask app
app = Flask(__name__)
CORS(app)

# Initialize the tinyllama model
llm = Ollama(model="tinyllama")

# Function to chunk text into smaller pieces
def chun(text, chunk_size=500):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    sentence_number = 1

    for sentence in sentences:
        current_chunk.append((sentence, sentence_number))
        if len(' '.join([s[0] for s in current_chunk])) > chunk_size:
            chunks.append(current_chunk)
            current_chunk = []
        sentence_number += 1

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

# Function to search database and retrieve relevant text
def search_database(query, documents):
    # Filter out empty or stop word documents
    filtered_documents = [doc for doc in documents if
                          doc[0].strip() and doc[0].strip().lower() not in stopwords.words('english')]

    if not filtered_documents:
        return None, None, None

    texts = [doc[0] for doc in filtered_documents]
    doc_names = [doc[1] for doc in filtered_documents]
    sentence_numbers = [doc[2] for doc in filtered_documents]

    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Fit vectorizer to documents
    X = vectorizer.fit_transform(texts)

    # Transform query to TF-IDF representation
    query_vec = vectorizer.transform([query])

    # Calculate cosine similarity between query and documents
    cosine_similarities = cosine_similarity(query_vec, X)

    # Get index of most relevant document
    doc_index = cosine_similarities.argmax()

    # Return the most relevant text, its name, and sentence number
    return texts[doc_index], doc_names[doc_index], sentence_numbers[doc_index]

# Function to generate response using tinyllama model
def generate_response(query, context, context_info, use_database):
    if use_database:
        prompt = (f"External data in the form of documents, specifically '{context_info}', is provided here that may or may not be relevant to the question. "
                  f"You may determine if the context will be needed or useful to answer the question. "
                  f"Respond to the following query: '{query}', only respond with a direct answer."
                  f"Context: {context}")
    else:
        prompt = f"Respond to the following query: '{query}'."

    response = llm.invoke(prompt)
    return response

# Function to parse HTML content from a URL
def parse_html_content(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        text_content = soup.get_text(separator=' ')
        return text_content
    except Exception as e:
        print("Error parsing HTML content:", e)
        return None

# Function to preprocess text
def preprocess_text(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    preprocessed_text = ' '.join(lemmatized_tokens)
    return preprocessed_text

# Function to create database and setup documents table
def create_database():
    conn = sqlite3.connect('data.db')
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS documents (id INTEGER PRIMARY KEY, text TEXT, name TEXT, sentence_number TEXT)")
    conn.commit()
    conn.close()

# Function to add document to the database
def add_document_to_database(text, name, sentence_number):
    conn = sqlite3.connect('data.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO documents (text, name, sentence_number) VALUES (?, ?, ?)", (text, name, sentence_number))
    conn.commit()
    conn.close()

@app.route('/rag', methods=['POST'])
def rag():
    data = request.json
    user_query = data.get("query", "")

    if "/bye" in user_query:
        return jsonify({"response": "Goodbye!"})

    if user_query.startswith("--url"):
        parts = user_query.split(" ", 2)
        url = parts[1]
        document_name = parts[2] if len(parts) > 2 else "Unnamed Document"

        html_content = parse_html_content(url)

        if html_content:
            preprocessed_text = preprocess_text(html_content)
            text_chunks = chun(preprocessed_text)

            for chunk in text_chunks:
                chunk_text = ' '.join([s[0] for s in chunk])
                sentence_numbers = ', '.join([str(s[1]) for s in chunk])
                add_document_to_database(chunk_text, document_name, sentence_numbers)

            return jsonify({"response": f"URL content from {url} has been added to the database under the name '{document_name}'."})

    else:
        conn = sqlite3.connect('data.db')
        cursor = conn.cursor()
        cursor.execute("SELECT text, name, sentence_number FROM documents")
        documents = cursor.fetchall()
        conn.close()

        relevant_text, doc_name, sentence_number = search_database(user_query, documents)

        if relevant_text:
            response = generate_response(user_query, relevant_text, doc_name, use_database=True)
            cited_response = f"{response} (Cited from document '{doc_name}', Sentence {sentence_number})"
            return jsonify({"response": cited_response})
        else:
            response = generate_response(user_query, "", "", use_database=False)
            return jsonify({"response": f"{response} (Generated from LLM's own knowledge)"})

# Route for the frontend
@app.route('/')
def index():
    return send_from_directory('', 'index.html')

if __name__ == "__main__":
    create_database()
    app.run(port=8111)
