import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder
from langchain.schema import BaseRetriever
from langchain.chains import create_retrieval_chain
from pathlib import Path
from langchain_mistralai import ChatMistralAI

# Prevent threading issues with sentence-transformers
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")


load_dotenv()
os.environ["Mistral_API_KEY"] = os.getenv("Mistral_API_KEY")

# Configuration des livres disponibles
BOOKS_CONFIG = {
    "bennabi_renaissance": {
        "title": "Les conditions de la renaissance",
        "author": "Malek Bennabi",
        "description": "Analyse philosophique de la renaissance civilisationnelle",
        "pdf_file": "Les_conditions_de_la_renaissance.pdf",
        "icon": "🌟",
        "prompt_context": "Tu incarnes Malek Bennabi (1905-1973), penseur algérien et auteur de \"Les conditions de la renaissance\"."
    },
    "bennabi_vocation": {
        "title": "Vocation de l'Islam",
        "author": "Malek Bennabi", 
        "description": "Réflexion sur le rôle et la mission de l'Islam dans le monde moderne",
        "pdf_file": "vocation-de-lislam.pdf",
        "icon": "📕",
        "prompt_context": "Tu incarnes Malek Bennabi (1905-1973), penseur algérien et auteur de \"Vocation de l'Islam\"."
    },
    "Le problème des idées dans le monde musulman": {
        "title": "Le problème des idées dans le monde musulman",
        "author": "Malek Bennabi",
        "description": "Analyse des idées et de leur impact sur la société musulmane",
        "pdf_file": "probleme.idees_.pdf",
        "icon": "📘",
        "prompt_context": "Tu incarnes Malek Bennabi (1905-1973), penseur algérien et auteur de 'Le problème des idées dans le monde musulman'."
    }
}

# Initialisation du modèle LLM
@st.cache_resource
def get_llm():
    return ChatMistralAI(
        mistral_api_key=os.environ["Mistral_API_KEY"],
        model="mistral-small",
        temperature=0.1
    )

@st.cache_resource
def get_cross_encoder():
    return CrossEncoder("BAAI/bge-reranker-base", device="cpu")

# Fonction de traitement des PDFs
@st.cache_resource
def load_and_process_pdf(book_key: str):
    import re, os
    book_config = BOOKS_CONFIG[book_key]
    def safe_path(s):
        return re.sub(r'[^a-zA-Z0-9_]', '_', s)
    vector_store_path = f"{safe_path(book_key)}_vector"
    pdf_file = book_config["pdf_file"]
    os.makedirs(vector_store_path, exist_ok=True)

    # Chargement et découpage du PDF
    loader = PyPDFLoader(pdf_file)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500,
        chunk_overlap=300,
        separators=["\n\n", "\n", " ", ". ", "? ", "! ", "; "],
        length_function=len,
    )
    final_documents = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()

    # Utiliser l'index existant si disponible
    if Path(f"{vector_store_path}.faiss").exists():
        st.info(f"📥 Chargement de l'index existant pour {book_config['title']}...")
        vector_store = FAISS.load_local(
            vector_store_path,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        return vector_store, final_documents

    # Sinon, créer et sauvegarder l'index
    st.info(f"🔄 Création de l'index vectoriel pour {book_config['title']}...")
    vector_store = FAISS.from_documents(final_documents, embeddings)
    vector_store.save_local(vector_store_path)
    st.success(f"✅ Index créé et sauvegardé pour {book_config['title']}")
    return vector_store, final_documents

# Création du prompt spécifique au livre
def create_book_prompt(book_key: str, response_style: str = "normal"):
    book_config = BOOKS_CONFIG[book_key]
    
    # Define different response styles
    style_instructions = {
        "normal": "# RÉPONSE (en français soutenu, style académique):",
        "short": "# RÉPONSE (en français soutenu, style académique, MAXIMUM 80 MOTS, réponse concise et directe):",
        "detailed": "# RÉPONSE (en français soutenu, style académique, analyse détaillée avec exemples et développements approfondis):"
    }
    
    # Adjust structure based on style
    if response_style == "short":
        structure_section = """
    # STRUCTURE DE TA RÉPONSE (COURTE ET DIRECTE)
    1. **Définition** (1-2 phrases) - Concept central selon Bennabi
    2. **Explication** (2-3 phrases) - Points clés avec citation courte si nécessaire
    """
    elif response_style == "detailed":
        structure_section = """
    # STRUCTURE DE TA RÉPONSE (DÉTAILLÉE)
    1. **Introduction** - Identifie le concept central de la question selon Bennabi
    2. **Analyse principale** - Développe l'explication en t'appuyant sur les passages pertinents
    3. **Citations directes** - Utilise citations directes pour les passages importants
    4. **Contextualisation** - Relie le concept aux autres théories de Bennabi si pertinent
    5. **Conclusion** - Synthèse claire de la position de Bennabi sur cette question
    6. **Implications** - Conséquences et applications des idées présentées
    """
    else:  # normal
        structure_section = """
    # STRUCTURE DE TA RÉPONSE
    1. **Introduction** - Identifie le concept central de la question selon Bennabi
    2. **Analyse principale** - Développe l'explication en t'appuyant sur les passages pertinents
    3. **Citations directes** - Utilise citations directes pour les passages importants
    4. **Contextualisation** - Relie le concept aux autres théories de Bennabi si pertinent
    5. **Conclusion** - Synthèse claire de la position de Bennabi sur cette question
    """
    
    return ChatPromptTemplate.from_template(f"""
    # ROLE ET CONTEXTE HISTORIQUE
    {book_config['prompt_context']} Tu as vécu l'époque coloniale, l'indépendance algérienne, et tu as développé des théories sur les causes de la stagnation du monde musulman et les conditions nécessaires pour sa renaissance.

    # METHODOLOGIE D'ANALYSE
    1. **Analyse conceptuelle précise** - Identifie les concepts-clés liés à la question dans le contexte fourni
    2. **Raisonnement causal structuré** - Distingue clairement: causes primaires, mécanismes, effets
    3. **Contextualisation historique** - Situe les idées dans leur contexte historique/social spécifique 
    4. **Citation textuelle** - Utilise des citations directes pour appuyer tes arguments
    5. **Objectivité analytique** - Présente les idées telles qu'exprimées dans le livre, même si controversées

    # CONCEPTS FONDAMENTAUX À RECONNAÎTRE
    - **Colonisabilité** - État psychologique/social rendant une société susceptible d'être colonisée
    - **Homme post-Mohammadien** - L'homme musulman après l'âge d'or de la civilisation islamique
    - **Civilisation** - Plus qu'un simple développement matériel, inclut dimensions spirituelles et culturelles
    - **Cycle vital des civilisations** - Naissance, apogée, déclin des civilisations selon des lois historiques
    - **Idées mortes/mortelles/vivantes** - Classification des idées selon leur impact social
    - **Efficacité sociale** - Capacité d'une société à transformer ses idées en réalités concrètes

    {structure_section}

    # LIMITES ET PRÉCISION
    - Si plusieurs interprétations existent, présente-les: "Selon le contexte, deux lectures sont possibles..."
    - Si une information précise n'est pas dans le contexte: "Le texte ne développe pas explicitement ce point, mais selon la logique générale de Bennabi..."
    - Si une question est totalement absente du contexte: "Cette question n'est pas traitée dans les passages fournis."

    # CONTEXTE FOURNI
    {{context}}

    # QUESTION
    {{input}}

    {style_instructions[response_style]}
    """)

# Fonction de réordonnancement des résultats
def rerank_with_crossencoder(query, docs, top_k=6):
    cross_encoder = get_cross_encoder()
    pairs = [[query, doc.page_content] for doc in docs]
    scores = cross_encoder.predict(pairs)
    scored_docs = list(zip(scores, docs))
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored_docs[:top_k]]

class HybridRerankRetriever(BaseRetriever):
    retriever: BaseRetriever

    def _get_relevant_documents(self, query: str, **kwargs):
        docs = self.retriever.get_relevant_documents(query, **kwargs)
        return rerank_with_crossencoder(query, docs)

    async def _aget_relevant_documents(self, query: str, **kwargs):
        docs = await self.retriever.aget_relevant_documents(query, **kwargs)
        return rerank_with_crossencoder(query, docs)

# Interface Streamlit
st.title("📚 Malek Bennabi Chatbot , Ask Questions about his books")
st.caption("Choisissez un livre pour commencer à discuter")

# Sélection du livre
st.header("📖 Sélectionnez un livre")

# Création des colonnes pour les livres
cols = st.columns(len(BOOKS_CONFIG))

for i, (book_key, book_config) in enumerate(BOOKS_CONFIG.items()):
    with cols[i]:
        st.subheader(f"{book_config['icon']} {book_config['author']}")
        st.write(f"**{book_config['title']}**")
        st.write(book_config['description'])
        
        if Path(book_config['pdf_file']).exists():
            if st.button(
                "💬 Discuter avec ce livre", 
                key=book_key,
                help=f"Cliquez pour charger et discuter avec {book_config['title']}"
            ):
                st.session_state.selected_book = book_key
        else:
            st.button(
                "📄 PDF manquant", 
                key=f"{book_key}_missing",
                disabled=True,
                help=f"Le fichier {book_config['pdf_file']} n'est pas trouvé"
            )

# Interface de chat avec le livre sélectionné
if "selected_book" in st.session_state and st.session_state.selected_book:
    selected_book_key = st.session_state.selected_book
    book_config = BOOKS_CONFIG[selected_book_key]
    
    st.divider()
    st.header(f"📚 Chat avec {book_config['author']} - {book_config['title']}")
    
    # Chargement des données du livre
    vector_key = f"{selected_book_key}_vector"
    docs_key = f"{selected_book_key}_docs"
    bm25_key = f"{selected_book_key}_bm25"
    chat_history_key = f"{selected_book_key}_chat_history"
    
    if vector_key not in st.session_state:
        with st.spinner(f"🔄 Préparation du livre {book_config['title']}..."):
            st.session_state[vector_key], st.session_state[docs_key] = load_and_process_pdf(selected_book_key)

    # Initialisation du système de récupération
    llm = get_llm()
    # Get response style from session state or set default
    response_style = st.session_state.get(f"style_{selected_book_key}", "normal")
    prompt = create_book_prompt(selected_book_key, response_style)
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Configuration du système de récupération
    faiss_retriever = st.session_state[vector_key].as_retriever(
        search_type='mmr',
        search_kwargs={
            "k": 4,
            "fetch_k": 8,
            "lambda_mult": 0.7
        }
    )

    # Mise en cache du BM25
    if bm25_key not in st.session_state:
        with st.spinner("🔍 Construction de l'index BM25..."):
            st.session_state[bm25_key] = BM25Retriever.from_documents(st.session_state[docs_key])
            st.session_state[bm25_key].k = 6

    bm25retriever = st.session_state[bm25_key]
    hybrid_retriever = EnsembleRetriever(
        retrievers=[faiss_retriever, bm25retriever], 
        weights=[0.7, 0.3]
    )

    hybrid_retriever = HybridRerankRetriever(retriever=hybrid_retriever)
    retrieval_chain = create_retrieval_chain(hybrid_retriever, document_chain)

    # Interface de conversation
    st.subheader("💬 Discussion")
    st.subheader("💬 Discussion")

    # Insert “Suggested Questions” buttons for onboarding (suggestion #9)
    st.write("### ❓ Questions suggérées")
    suggested = [
        "Quelle est la notion de colonisabilité selon Bennabi ?",
        "Comment Bennabi définit-il la civilisation ?",
        "Quelles sont les étapes du cycle vital des civilisations ?"
    ]
    for idx, q in enumerate(suggested):
        if st.button(q, key=f"suggest_{selected_book_key}_{idx}"):
            # store the clicked suggestion in session_state and rerun
            st.session_state[f"{selected_book_key}_suggested"] = q
            st.rerun()

    # When handling input, pick up either a suggested question or chat_input
    suggested_q = st.session_state.pop(f"{selected_book_key}_suggested", None)
    user_question = suggested_q or st.chat_input(
        f"Posez une question à {book_config['author']}...",
        key=f"user_input_{selected_book_key}"
    )

    # Bouton de retour
    if st.button("⬅️ Retour à la sélection de livres"):
        if "selected_book" in st.session_state:
            del st.session_state.selected_book
        st.rerun()

    # Initialisation de l'historique
    if chat_history_key not in st.session_state:
        st.session_state[chat_history_key] = []
    
    # Affichage de l'historique
    for i, (question, answer) in enumerate(st.session_state[chat_history_key]):
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant", avatar=book_config['icon']):
            st.write(answer)
            
            # Boutons de feedback pour le dernier message
            if i == len(st.session_state[chat_history_key]) - 1:
                feedback_col1, feedback_col2 = st.columns([1, 6])
                with feedback_col1:
                    st.write("Utile?")
                with feedback_col2:
                    feedback_cols = st.columns([1, 1, 8])
                    with feedback_cols[0]:
                        if st.button("👍", key=f"yes_{i}"):
                            st.session_state.setdefault("feedback", []).append((question, "positive"))
                            st.success("Merci!")
                    with feedback_cols[1]:
                        if st.button("👎", key=f"no_{i}"):
                            st.session_state.setdefault("feedback", []).append((question, "negative"))
    
    # Saisie de la question
    if user_question:
        # Affichage du message utilisateur
        with st.chat_message("user"):
            st.write(user_question)
            
        # Indicateur de réponse en cours
        with st.chat_message("assistant", avatar=book_config['icon']):
            message_placeholder = st.empty()
            message_placeholder.info("Recherche dans le livre...")
            
            try:
                # Mode citation exacte
                exact_mode = any(keyword in user_question.lower() for keyword in ["citation", "extrait", "mot pour mot", "texte exact"])
                if exact_mode:
                    docs = hybrid_retriever.get_relevant_documents(user_question)
                    message_placeholder.write("Extrait exact trouvé :")
                    for doc in docs[:2]:
                        message_placeholder.write(doc.page_content)
                    
                    citation_response = "\n\n".join([doc.page_content for doc in docs[:2]])
                    st.session_state[chat_history_key].append((user_question, citation_response))
                    st.rerun()
                
                # Utilisation du contexte des conversations récentes
                if st.session_state[chat_history_key]:
                    recent_history = st.session_state[chat_history_key][-2:]
                    context_str = "\n".join([f"Q: {q}\nR: {a[:150]}..." for q, a in recent_history])
                    enhanced_question = f"Contexte des questions précédentes:\n{context_str}\n\nNouvelle question: {user_question}"
                    response = retrieval_chain.invoke({"input": enhanced_question})
                else:
                    response = retrieval_chain.invoke({"input": user_question})
                
                # Affichage de la réponse
                message_placeholder.write(response["answer"])
                
                # Mise à jour de l'historique
                st.session_state[chat_history_key].append((user_question, response["answer"]))
                
                # Affichage des sources
                with st.expander("📖 Sources"):
                    for i, doc in enumerate(response["context"]):
                        relevance = "🔥" if i < 2 else "📄"
                        st.write(f"**{relevance} Source {i+1} - Page approx. {doc.metadata.get('page', 'Unknown')}:**")
                        st.write(doc.page_content)
                        st.write("---")
                
            except Exception as e:
                message_placeholder.error(f"Erreur: {e}")
                message_placeholder.info("Essayez de reformuler votre question")
                st.stop()

    # Option d'exportation de la conversation
    export_col1, export_col2 = st.columns([1, 5])
    with export_col1:
        if st.session_state[chat_history_key] and st.button("💾 Exporter", key=f"export_{selected_book_key}"):
            conversation_text = f"# Conversation avec {book_config['author']} - {book_config['title']}\n\n"
            conversation_text += "\n\n".join([
                f"## Question\n{q}\n\n## Réponse\n{a}" 
                for q, a in st.session_state[chat_history_key]
            ])
            with export_col2:
                st.download_button(
                    "📥 Télécharger (.md)",
                    conversation_text,
                    f"conversation_{book_config['author']}_{book_config['title']}.md"
                )

    # Response style selector
    response_style = st.radio(
        "🎯 Style de réponse:",
        ["normal", "short", "detailed"],
        format_func=lambda x: {
            "normal": "📝 Normal",
            "short": "⚡ Concis", 
            "detailed": "📚 Détaillé"
        }[x],
        horizontal=True,
        key=f"style_{selected_book_key}"
    )

else:
    # Instructions quand aucun livre n'est sélectionné
    st.info("👆 Sélectionnez un livre ci-dessus pour commencer à discuter")
    
    st.subheader("🚀 Fonctionnalités")
    st.write("""
    - **Chat intelligent** avec le contenu des livres
    - **Citations exactes** en utilisant les mots-clés 'citation' ou 'extrait'
    - **Sources vérifiables** pour chaque réponse
    - **Historique de conversation** par livre
    - **Recherche hybride** (sémantique + mots-clés)
    """)
    
    st.subheader("💡 Comment utiliser")
    st.write("""
    1. Cliquez sur un livre disponible
    2. Attendez le chargement de l'index (première fois seulement)
    3. Posez vos questions dans la zone de texte
    4. Explorez les sources pour vérifier les réponses
    """)
    
    st.subheader("📚 Livres disponibles")
    available_books = [config['title'] for config in BOOKS_CONFIG.values() if Path(config['pdf_file']).exists()]
    missing_books = [config['title'] for config in BOOKS_CONFIG.values() if not Path(config['pdf_file']).exists()]
    
    if available_books:
        st.success(f"✅ Livres prêts: {', '.join(available_books)}")
    if missing_books:
        st.warning(f"📄 Fichiers PDF manquants: {', '.join(missing_books)}")

# Barre latérale
with st.sidebar:
    st.subheader("🔍 Recherche globale")
    global_search = st.text_input("Rechercher dans tous les livres:")
    if global_search:
        st.write("Résultats de recherche:")
        results_found = False
        
        for book_key, book_config in BOOKS_CONFIG.items():
            if Path(book_config['pdf_file']).exists():
                import re
                def safe_path(s):
                    return re.sub(r'[^a-zA-Z0-9_]', '_', s)
                    
                vector_key = f"{safe_path(book_key)}_vector"
                
                if vector_key in st.session_state:
                    try:
                        results = st.session_state[vector_key].similarity_search(global_search, k=2)
                        results_found = True
                        
                        with st.expander(f"📘 {book_config['title']}"):
                            for i, doc in enumerate(results):
                                st.write(f"**Extrait {i+1}:**")
                                st.write(doc.page_content[:300] + "...")
                                if st.button(f"💬 Discuter", key=f"search_{book_key}_{i}"):
                                    st.session_state.selected_book = book_key
                                    st.rerun()
                    except Exception as e:
                        st.write(f"Erreur: {str(e)}")
        
        if not results_found:
            st.info("Aucun livre indexé trouvé. Veuillez d'abord ouvrir un livre pour l'indexer.")
    
    # Mode sombre
    st.divider()
    if st.checkbox("🌙 Mode sombre"):
        st.markdown("""
        <style>
        .stApp {background-color: #0e1117; color: #fafafa;}
        .stMarkdown {color: #fafafa;}
        </style>
        """, unsafe_allow_html=True)