from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class ResumeMatcher:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_text(self, text: str) -> np.ndarray:
        if not text or not text.strip():
            return np.zeros((1, 384))
        return self.model.encode([text], normalize_embeddings=True)

    def similarity(self, resume_text: str, jd_text: str) -> float:
        resume_emb = self.embed_text(resume_text)
        jd_emb = self.embed_text(jd_text)
        sim = cosine_similarity(resume_emb, jd_emb)[0][0]
        return float(sim * 100)

    def extract_bullets(self, text: str, min_len: int = 25):
        parts = [p.strip(" -•\n\r\t") for p in text.split("\n") if len(p.strip()) >= min_len]
        return parts

    def compare_bullets(self, resume_text: str, jd_text: str, top_k: int = 5):
        resume_items = self.extract_bullets(resume_text)
        jd_items = self.extract_bullets(jd_text)

        if not resume_items or not jd_items:
            return [], []

        resume_embs = self.model.encode(resume_items, normalize_embeddings=True)
        jd_embs = self.model.encode(jd_items, normalize_embeddings=True)

        sim_matrix = cosine_similarity(resume_embs, jd_embs)

        strengths = []
        gaps = []

        for i, res_b in enumerate(resume_items):
            j_best = int(np.argmax(sim_matrix[i]))
            score = float(sim_matrix[i, j_best])
            strengths.append((res_b, jd_items[j_best], score))

        strengths.sort(key=lambda x: x[2], reverse=True)
        strengths = strengths[:top_k]

        for j, jd_b in enumerate(jd_items):
            i_best = int(np.argmax(sim_matrix[:, j]))
            score = float(sim_matrix[i_best, j])
            if score < 0.6:
                gaps.append((jd_b, score))

        gaps.sort(key=lambda x: x[1])
        gaps = gaps[:top_k]

        return strengths, gaps
