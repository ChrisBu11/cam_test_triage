import time
import numpy as np

class FacePresenceTracker:
    def __init__(self, similarity_threshold=0.6):
        # Maps integer ID -> {'embedding': ndarray, 'total_time': float, 'last_seen': float}
        self.database = {}
        self.next_id = 1
        self.similarity_threshold = similarity_threshold

    def match_or_register(self, current_embedding) -> tuple[int, float]:
        """
        Compares an embedding to the database. 
        Returns (assigned_id, total_time_in_frame).
        """
        if current_embedding is None:
            return 0, 0.0

        matched_id = None
        max_similarity = -1.0

        # Compute cosine similarity against all known faces
        for face_id, data in self.database.items():
            known_emb = data['embedding']
            # Cosine similarity formula
            sim = np.dot(current_embedding, known_emb) / (np.linalg.norm(current_embedding) * np.linalg.norm(known_emb))
            
            if sim > self.similarity_threshold and sim > max_similarity:
                max_similarity = sim
                matched_id = face_id

        now = time.time()

        if matched_id is not None:
            # Face found! Update database tracking state
            time_delta = now - self.database[matched_id]['last_seen']
            # Only add to time if the gap is small (e.g. didn't leave the room and come back)
            if time_delta < 15.0: 
                self.database[matched_id]['total_time'] += time_delta
            
            self.database[matched_id]['last_seen'] = now
            # Update embedding slightly to account for changing angles/lighting
            self.database[matched_id]['embedding'] = 0.9 * self.database[matched_id]['embedding'] + 0.1 * current_embedding
        else:
            # Brand new face! Register them
            matched_id = self.next_id
            self.database[matched_id] = {
                'embedding': current_embedding,
                'total_time': 0.0,
                'last_seen': now,
                'first_seen': now
            }
            self.next_id += 1

        return matched_id, self.database[matched_id]['total_time']

    def update_passive_time(self, active_id):
        """Ticks the clock forward for the active face between recognition checks."""
        now = time.time()
        if active_id in self.database:
            time_delta = now - self.database[active_id]['last_seen']
            if time_delta < 2.0: # Ensure they are actively sitting in frame
                self.database[active_id]['total_time'] += time_delta
            self.database[active_id]['last_seen'] = now
            return self.database[active_id]['total_time']
        return 0.0
