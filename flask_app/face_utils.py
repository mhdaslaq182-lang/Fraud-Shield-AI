import pickle
import numpy as np
import os
import base64
from PIL import Image
import io

def _fr():
    import face_recognition
    return face_recognition

FACES_DB = os.path.join(os.path.dirname(__file__), "..", "faces", "encodings.pkl")

def load_db():
    if os.path.exists(FACES_DB):
        return pickle.load(open(FACES_DB, "rb"))
    return {}

def save_db(db):
    os.makedirs(os.path.dirname(FACES_DB), exist_ok=True)
    pickle.dump(db, open(FACES_DB, "wb"))

def register_face_from_image(username, image_data):
    try:
        if "base64," in image_data:
            image_data = image_data.split("base64,")[1]
        img_bytes = base64.b64decode(image_data)
        img       = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_array = np.array(img)
        locations = _fr().face_locations(img_array)
        encodings = _fr().face_encodings(img_array, locations)
        if not encodings:
            return False, "No face detected in image"
        db = load_db()
        db[username] = encodings[0]
        save_db(db)
        return True, "Face registered successfully"
    except Exception as e:
        return False, str(e)

def verify_face_from_image(username, image_data):
    try:
        db = load_db()
        if username not in db:
            return False, "Face not registered for this user"
        known = db[username]
        if "base64," in image_data:
            image_data = image_data.split("base64,")[1]
        img_bytes = base64.b64decode(image_data)
        img       = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_array = np.array(img)
        locations = _fr().face_locations(img_array)
        encodings = _fr().face_encodings(img_array, locations)
        if not encodings:
            return False, "No face detected in camera"
        match      = _fr().compare_faces([known], encodings[0], tolerance=0.6)
        distance   = _fr().face_distance([known], encodings[0])[0]
        confidence = round((1 - distance) * 100, 1)
        if match[0]:
            return True, f"Identity verified ({confidence}% confidence)"
        else:
            return False, f"Face does not match ({confidence}% confidence)"
    except Exception as e:
        return False, str(e)

def is_face_registered(username):
    db = load_db()
    return username in db
