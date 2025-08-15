📚 Bennabi RAG Chatbot
A (RAG) chatbot powered by the Mistral LLM (model="mistral-small"). It is designed to answer questions about three books by Malek Bennabi by combining document retrieval with generative AI.

Features
📖 Retrieves relevant excerpts directly from the books using a hybrid retrieval system combining FAISS vector search and BM25 keyword search with a cross-encoder re-ranker for improved accuracy.

💡 Generates contextual answers based on Bennabi’s ideas when explicit information is not available in the texts.

🎯 Uses LangChain chains including document loaders, retrieval chains, and prompt templates customized per book.

Built with Streamlit interface.
