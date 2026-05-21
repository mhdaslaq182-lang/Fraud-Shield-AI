# =============================================
#   Face Recognition Login Module
#   Sri Lanka Banking Fraud Detection 🇱🇰
#   face_auth.py
# =============================================

import face_recognition
import cv2
import numpy as np
import pickle
import os
from datetime import datetime

FACES_DB  = "faces/encodings.pkl"
FACES_LOG = "faces/access_log.txt"
os.makedirs("faces", exist_ok=True)


def load_db():
    """Load saved face encodings."""
    if os.path.exists(FACES_DB):
        return pickle.load(open(FACES_DB, "rb"))
    return {}


def save_db(db):
    """Save face encodings."""
    pickle.dump(db, open(FACES_DB, "wb"))


def log_access(username: str, status: str):
    """Log every access attempt."""
    entry = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] " \
            f"User: {username} — {status}\n"
    with open(FACES_LOG, "a") as f:
        f.write(entry)
    print(f"  📝 Logged: {entry.strip()}")


# ── Register New User Face ──
def register_face(username: str):
    """
    Opens webcam and captures face for registration.
    Press SPACE to capture, Q to quit.
    """
    db  = load_db()
    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        print("  ❌ Camera not found!")
        return False

    print(f"\n{'='*50}")
    print(f"  📸 REGISTERING FACE: {username}")
    print(f"{'='*50}")
    print("  Look at the camera.")
    print("  Press SPACE to capture.")
    print("  Press Q to quit.")
    print(f"{'='*50}")

    registered = False

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        # Show face detection box
        rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb)

        for (top, right, bottom, left) in locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, "Face Detected", (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(frame, f"Registering: {username}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 0), 2)
        cv2.putText(frame, "SPACE=Capture  Q=Quit",
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Face Registration — Sri Lanka Banking", frame)
        key = cv2.waitKey(1)

        if key == 32:  # SPACE
            if locations:
                encodings = face_recognition.face_encodings(rgb, locations)
                if encodings:
                    db[username] = encodings[0]
                    save_db(db)
                    log_access(username, "REGISTERED ✅")
                    print(f"\n  ✅ Face registered for '{username}'!")
                    registered = True
                    break
            else:
                print("  ⚠️  No face detected. Try again.")

        elif key == ord('q'):
            print("  ❌ Registration cancelled.")
            break

    cam.release()
    cv2.destroyAllWindows()
    return registered


# ── Verify User Face ──
def verify_face(username: str, tolerance: float = 0.5) -> bool:
    """
    Opens webcam and verifies user identity.
    Returns True if face matches registered face.
    """
    db = load_db()

    if username not in db:
        print(f"  ❌ User '{username}' not registered.")
        print(f"     Run: register_face('{username}') first.")
        return False

    known_encoding = db[username]
    cam            = cv2.VideoCapture(0)

    if not cam.isOpened():
        print("  ❌ Camera not found!")
        return False

    print(f"\n{'='*50}")
    print(f"  🔐 VERIFYING IDENTITY: {username}")
    print(f"{'='*50}")
    print("  Look at the camera...")
    print(f"{'='*50}")

    verified     = False
    attempts     = 0
    max_attempts = 60  # ~3 seconds

    while attempts < max_attempts:
        ret, frame = cam.read()
        if not ret:
            break

        rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, locations)

        for (top, right, bottom, left), encoding in zip(locations, encodings):
            match    = face_recognition.compare_faces(
                [known_encoding], encoding, tolerance=tolerance
            )
            distance = face_recognition.face_distance([known_encoding], encoding)[0]

            if match[0]:
                color  = (0, 255, 0)
                label  = f"✅ VERIFIED ({1-distance:.0%})"
                verified = True
            else:
                color  = (0, 0, 255)
                label  = f"❌ NOT MATCHED ({1-distance:.0%})"

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, label, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        status_text = "✅ VERIFIED" if verified else f"Scanning... {attempts+1}/{max_attempts}"
        color_text  = (0, 255, 0) if verified else (0, 165, 255)
        cv2.putText(frame, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_text, 2)
        cv2.putText(frame, f"User: {username}",
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Face Verification — Sri Lanka Banking", frame)
        cv2.waitKey(1)

        if verified:
            import time
            time.sleep(1)
            break

        attempts += 1

    cam.release()
    cv2.destroyAllWindows()

    if verified:
        log_access(username, "ACCESS GRANTED ✅")
        print(f"  ✅ Identity confirmed: {username}")
        print(f"  🏦 Welcome to Sri Lanka Banking Fraud Detection System!")
    else:
        log_access(username, "ACCESS DENIED ❌")
        print(f"  🚨 Identity FAILED for: {username}")
        print(f"  ❌ Access denied!")

    return verified


# ── List Registered Users ──
def list_users():
    """Show all registered users."""
    db = load_db()
    print(f"\n{'='*40}")
    print(f"  👥 REGISTERED USERS")
    print(f"{'='*40}")
    if db:
        for i, name in enumerate(db.keys(), 1):
            print(f"  {i}. {name}")
        print(f"\n  Total: {len(db)} user(s)")
    else:
        print("  No users registered yet.")
    print(f"{'='*40}")


# ── Remove User ──
def remove_user(username: str):
    """Remove a registered user."""
    db = load_db()
    if username in db:
        del db[username]
        save_db(db)
        log_access(username, "REMOVED 🗑️")
        print(f"  ✅ User '{username}' removed.")
    else:
        print(f"  ❌ User '{username}' not found.")


# ── Show Access Log ──
def show_log():
    """Display access history."""
    print(f"\n{'='*55}")
    print(f"  📋 FACE ACCESS LOG")
    print(f"{'='*55}")
    if os.path.exists(FACES_LOG):
        with open(FACES_LOG, "r") as f:
            lines = f.readlines()
        for line in lines[-20:]:  # show last 20
            print(f"  {line.strip()}")
        print(f"\n  Total entries: {len(lines)}")
    else:
        print("  No access log yet.")
    print(f"{'='*55}")


if __name__ == "__main__":
    print("="*50)
    print("  🔐 FACE RECOGNITION MODULE TEST")
    print("  Sri Lanka Banking Fraud Detection 🇱🇰")
    print("="*50)
    print("\nCommands:")
    print("  1. Register : register_face('your_name')")
    print("  2. Verify   : verify_face('your_name')")
    print("  3. List     : list_users()")
    print("  4. Log      : show_log()")
    print("\nRunning list_users()...")
    list_users()

    print("\n  To register your face run:")
    print("  python3 -c \"from face_auth import register_face; register_face('admin')\"")
    print("\n  To verify your face run:")
    print("  python3 -c \"from face_auth import verify_face; verify_face('admin')\"")
