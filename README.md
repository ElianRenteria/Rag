The app is a simple Question-Answering (QA) system built on the Retrieve-and-Generate (RAG) architecture. It utilizes the Ollama TinyLLama language model for generating responses. 
Users can interact with the app by sending HTTP POST requests to the /rag endpoint. 
The app accepts JSON data containing a query, which can be a question or a request to fetch content from a specified URL using the --url parameter. 
The backend processes the query by either fetching content from the URL and storing it in the database or searching the existing database for relevant information. 
If relevant information is found in the database, the app generates a response using the TinyLLama model and provides citations from the stored documents. 
If no relevant information is found, the model generates a response based on its own knowledge. The app also features a simple frontend UI accessible at the domain elianrenteria.me, 
allowing users to interact with the QA system through a web interface.

This setup allows users to ask questions, fetch content from URLs to add to the database, and receive responses with citations if relevant information is found.
