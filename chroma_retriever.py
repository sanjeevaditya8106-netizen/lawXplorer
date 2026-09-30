import chromadb
from sentence_transformers import SentenceTransformer

class LawXplorerChromaStore:
    """Manages persistent vector indexing and precedent retrieval using ChromaDB."""

    def __init__(self, persist_dir: str = "./lawxplorer_chroma_db", collection_name: str = "legal_precedents"):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Indian Legal Precedents Store", "hnsw:space": "cosine"}
        )
        
        if self.collection.count() == 0:
            self._seed_default_precedents()

    def _seed_default_precedents(self):
        seed_data = [
            {
                "id": "case_001",
                "text": "Right to privacy is declared an intrinsic part of the right to life and personal liberty under Article 21.",
                "metadata": {
                    "case_name": "K.S. Puttaswamy v. Union of India (2017)",
                    "citation": "(2017) 10 SCC 1",
                    "citator_status": "ACTIVE PRECEDENT",
                    "topic": "Fundamental Rights"
                }
            },
            {
                "id": "case_002",
                "text": "Procedure established by law under Article 21 must be just, fair, and reasonable, not arbitrary.",
                "metadata": {
                    "case_name": "Maneka Gandhi v. Union of India (1978)",
                    "citation": "(1978) 1 SCC 248",
                    "citator_status": "ACTIVE PRECEDENT",
                    "topic": "Fundamental Rights"
                }
            },
            {
                "id": "case_003",
                "text": "Narrow interpretation of personal liberty as distinct silos of rights.",
                "metadata": {
                    "case_name": "A.K. Gopalan v. State of Madras (1950)",
                    "citation": "AIR 1950 SC 27",
                    "citator_status": "OVERRULED by Maneka Gandhi (1978)",
                    "topic": "Fundamental Rights"
                }
            },
            {
                "id": "case_004",
                "text": "Post-contractual negative covenants in employment and restraint of trade are void under Section 27 of the Indian Contract Act.",
                "metadata": {
                    "case_name": "Percept D'Mark v. Zaheer Khan (2006)",
                    "citation": "AIR 2006 SC 3426",
                    "citator_status": "ACTIVE PRECEDENT",
                    "topic": "Contract & Commercial Law"
                }
            }
        ]
        self.add_precedents(seed_data)

    def add_precedents(self, items: list[dict]):
        ids = [item["id"] for item in items]
        documents = [item["text"] for item in items]
        metadatas = [item["metadata"] for item in items]
        embeddings = self.embed_model.encode(documents, normalize_embeddings=True).tolist()

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

    def query_precedents(self, query_text: str, n_results: int = 3, topic_filter: str = None) -> list[dict]:
        query_embedding = self.embed_model.encode([query_text], normalize_embeddings=True).tolist()
        where_filter = {"topic": topic_filter} if topic_filter and topic_filter != "Other" else None

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where=where_filter
        )

        formatted = []
        if results and results["documents"]:
            for i in range(len(results["documents"][0])):
                dist = results["distances"][0][i] if results.get("distances") else 0.0
                formatted.append({
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "similarity_score": round(1.0 - dist, 4)
                })
        return formatted
