from model.language_model import generate_response
from backend.embedding import search_embeddings, store_embeddings
import numpy as np

def handle_query(query, sources):
    # Convert sources to embeddings and retrieve relevant documents
    index = store_embeddings(sources)
    indices, _ = search_embeddings(index, query)

    # Debugging prints
    print("Indices:", indices)
    print("Types of indices:", [type(i) for i in indices])  # Inspect types

    # Flatten the indices if they are in a 2D structure
    indices = np.array(indices).flatten()  # Convert to 1D array
    indices = [int(i) for i in indices if isinstance(i, (int, float)) and i >= 0]  # Ensure valid indices

    if not indices:
        return "No relevant documents found."
    
    # Use the retrieved document snippets to generate a response
    relevant_docs = [sources[i] for i in indices]
    response = generate_response(query, relevant_docs)
    return response
