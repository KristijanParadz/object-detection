import numpy as np
import random


class GlobalIDManager:
    def __init__(self, similarity_threshold):
        self.similarity_threshold = similarity_threshold
        self.global_tracks = {}
        self.global_id_to_class = {}
        self.next_global_id = 1

    def match_or_create(self, embedding, class_id):
        if class_id not in self.global_tracks:
            self.global_tracks[class_id] = {}

        best_g_id = None
        best_sim = -1.0
        for g_id, (g_emb, g_color) in self.global_tracks[class_id].items():
            sim = float(np.dot(embedding, g_emb))
            if sim > best_sim:
                best_sim = sim
                best_g_id = g_id

        if best_g_id is not None and best_sim >= self.similarity_threshold:
            return best_g_id

        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )
        new_g_id = self.next_global_id
        self.next_global_id += 1

        self.global_tracks[class_id][new_g_id] = (embedding, color)
        self.global_id_to_class[new_g_id] = class_id

        return new_g_id

    def get_color(self, global_id):
        if global_id not in self.global_id_to_class:
            return (255, 255, 255)
        cls_id = self.global_id_to_class[global_id]
        if cls_id in self.global_tracks and global_id in self.global_tracks[cls_id]:
            return self.global_tracks[cls_id][global_id][1]
        return (255, 255, 255)

    def update_embedding(self, global_id, new_embedding, alpha=0.7):
        if global_id not in self.global_id_to_class:
            return
        cls_id = self.global_id_to_class[global_id]
        if cls_id not in self.global_tracks or global_id not in self.global_tracks[cls_id]:
            return

        old_emb, color = self.global_tracks[cls_id][global_id]
        blended = alpha * old_emb + (1.0 - alpha) * new_embedding
        norm = np.linalg.norm(blended)
        if norm > 0:
            blended /= norm
        self.global_tracks[cls_id][global_id] = (blended, color)

    def reset(self):
        self.global_tracks.clear()
        self.global_id_to_class.clear()
        self.next_global_id = 1
