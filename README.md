# Advanced programming
assignment 1 advanced programming.  Team work by Togzhan Oral and Yelnura Akhmetova. 

# Chat with ollama - RAG integration

## Overview

This project demonstrates a **Retrieval-Augmented Generation (RAG)** pipeline using ChromaDB and Ollama models. The application allows users to interact with stored documents in a ChromaDB collection, ask questions, and receive contextually relevant answers powered by Ollama.

## Installation

Follow these steps to get your development environment set up:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/chat-with-ollama.git
    ```

2. Navigate to the project directory:
    ```bash
    cd chat-with-ollama
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Ensure you have the necessary API keys and environment setup (if applicable).

## Usage

To run the application locally:

1. Start the Streamlit app:
    ```bash
    streamlit run src/app.py
    ```

2. The app will open in your browser where you can interact with it by:

    - Selecting a model (e.g., "llama3.2", "phi3", "mistral").
    - Adding new documents to ChromaDB.
    - Asking questions based on the documents stored in the ChromaDB collection.

## Features

- **Add Documents to ChromaDB**: Upload new documents into a ChromaDB collection.
- **Ask Questions**: Query the collection, and get answers using the RAG pipeline.
- **Show Data**: View the stored documents in ChromaDB.

## Examples

- **Add a New Document**:
    - Enter your document in the provided text area, then click "Add Document" to store it in the database.

- **Ask a Question**:
    - Type in a question related to the documents you added. The app will retrieve relevant information and generate an answer using the RAG pipeline.


 Screenshot of the chat with Ollama: 
 <img width="1425" alt="image" src="https://github.com/user-attachments/assets/09c116ae-590d-4c15-9ab2-8e63a252f888" />

      




