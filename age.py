import logging

class DEXAgeEstimator:
    def __init__(self):
        # No model initialization needed since InsightFace already loaded it!
        logging.info("DEXAgeEstimator initialized using existing InsightFace backend.")

    def predict(self, face_object) -> int:
        """
        Extracts the age from the InsightFace face object.
        Falls back safely if the attribute is missing.
        """
        try:
            # InsightFace's buffalo_l genderage model stores age as an integer directly on the face object
            if hasattr(face_object, 'age'):
                return int(face_object.age)
            
            # If face_object is just a cropped image matrix instead of an InsightFace Face object
            if hasattr(face_object, 'shape') and len(face_object.shape) >= 2:
                logging.warning("DEXAgeEstimator received an image crop instead of an InsightFace object. Returning default age.")
                return 30
                
            return 30
        except Exception as e:
            logging.error(f"Failed to read age from face object: {e}")
            return 30
