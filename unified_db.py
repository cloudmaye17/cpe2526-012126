#!/usr/bin/env python3
"""
Unified Database Manager for Face Recognition Attendance System
Microcity College of Business and Technology

All-in-one database management:
- Setup database
- Add/manage students
- Capture/delete training images
- Train model
- View database
- Delete specific student images
"""

import sqlite3
import os
import sys
import cv2
import numpy as np
import pickle
from datetime import datetime
import shutil

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


# ==================== DATABASE SETUP ====================

def setup_database():
    """Create database with all required tables"""
    print("\n" + "=" * 70)
    print("SETTING UP DATABASE")
    print("=" * 70)
    
    os.makedirs("database", exist_ok=True)
    os.makedirs("database/training_images", exist_ok=True)
    
    conn = sqlite3.connect('database/faces.db')
    cursor = conn.cursor()
    
    # Persons table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS persons (
            person_id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT UNIQUE NOT NULL,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            course TEXT,
            year_level TEXT,
            section TEXT,
            email TEXT,
            phone TEXT,
            date_enrolled TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'active',
            photo_count INTEGER DEFAULT 0
        )
    ''')
    
    # Faces table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS faces (
            face_id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER NOT NULL,
            face_encoding BLOB,
            image_path TEXT,
            capture_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            quality_score REAL,
            FOREIGN KEY (person_id) REFERENCES persons(person_id)
        )
    ''')
    
    # Attendance logs table with datetime column
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance_logs (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER NOT NULL,
            time_in TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            confidence REAL,
            camera_id TEXT,
            location TEXT,
            status TEXT DEFAULT 'present',
            accuracy_score REAL,
            factors TEXT,
            FOREIGN KEY (person_id) REFERENCES persons(person_id)
        )
    ''')
    
    # System settings table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS system_settings (
            setting_key TEXT PRIMARY KEY,
            setting_value TEXT,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Training history table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS training_history (
            training_id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            total_persons INTEGER,
            total_faces INTEGER,
            accuracy REAL,
            model_path TEXT
        )
    ''')
    
    conn.commit()
    
    # Insert default settings
    cursor.execute('''
        INSERT OR IGNORE INTO system_settings (setting_key, setting_value)
        VALUES ('recognition_threshold', '55')
    ''')
    
    cursor.execute('''
        INSERT OR IGNORE INTO system_settings (setting_key, setting_value)
        VALUES ('min_faces_per_person', '10')
    ''')
    
    conn.commit()
    conn.close()
    
    print("✓ Database setup complete!")
    print("✓ Tables created: persons, faces, attendance_logs, system_settings, training_history")
    return True


# ==================== STUDENT MANAGEMENT ====================

def add_student(student_info):
    """Add a new student to database"""
    conn = sqlite3.connect('database/faces.db')
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO persons 
            (student_id, first_name, last_name, course, year_level, section, email, phone)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            student_info['student_id'],
            student_info['first_name'],
            student_info['last_name'],
            student_info['course'],
            student_info['year_level'],
            student_info['section'],
            student_info.get('email', ''),
            student_info.get('phone', '')
        ))
        
        conn.commit()
        person_id = cursor.lastrowid
        
        folder_name = f"Person_{person_id}_{student_info['first_name']}_{student_info['last_name']}"
        folder_path = os.path.join("database", "training_images", folder_name)
        os.makedirs(folder_path, exist_ok=True)
        
        print(f"\n✓ Student added successfully!")
        print(f"  Person ID: {person_id}")
        print(f"  Name: {student_info['first_name']} {student_info['last_name']}")
        print(f"  Student ID: {student_info['student_id']}")
        print(f"  Folder: {folder_path}")
        
        return person_id, folder_path
        
    except sqlite3.IntegrityError:
        print(f"\n✗ Error: Student ID '{student_info['student_id']}' already exists!")
        return None, None
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return None, None
    finally:
        conn.close()


def add_group_members():
    """Add all group members"""
    group_members = [
        {
            'student_id': '2210003M',
            'first_name': 'Eisley',
            'last_name': 'Atienza',
            'course': 'Computer Engineering',
            'year_level': '4th Year',
            'section': 'A',
            'email': 'eisley.atienza@microcity.edu.ph',
            'phone': '0967-676-6767'
        },
        {
            'student_id': '2210014M',
            'first_name': 'Jahanna',
            'last_name': 'Castro',
            'course': 'Computer Engineering',
            'year_level': '4th Year',
            'section': 'A',
            'email': 'jahnna.castro@microcity.edu.ph',
            'phone': '0938-674-6084'
        },
        {
            'student_id': '2210012M',
            'first_name': 'Diotanico',
            'last_name': 'Padilla',
            'course': 'Computer Engineering',
            'year_level': '4th Year',
            'section': 'A',
            'email': 'diotanico.padilla@microcity.edu.ph',
            'phone': '0938-722-4395'
        },
        {
            'student_id': '2210017M',
            'first_name': 'Betty Maye',
            'last_name': 'Eugenio',
            'course': 'Computer Engineering',
            'year_level': '4th Year',
            'section': 'A',
            'email': 'bettymayeeugenio0526@gmail.com',
            'phone': '0994-207-9840'
        },
        {
            'student_id': '2210016M',
            'first_name': 'Ace Joshua',
            'last_name': 'Abad',
            'course': 'Computer Engineering',
            'year_level': '4th Year',
            'section': 'A',
            'email': 'aceabad2300@gmail.com',
            'phone': '0919-386-7357'
        },
        {
            'student_id': '2210018M',
            'first_name': 'Richard Enock',
            'last_name': 'Catalan',
            'course': 'Computer Engineering',
            'year_level': '4th Year',
            'section': 'A',
            'email': 'enock.catalan@microcity.edu.ph',
            'phone': '0912-345-6789'
        },
        {
            'student_id': '2210010M',
            'first_name': 'Randel Krishna',
            'last_name': 'Bugay',
            'course': 'Computer Engineering',
            'year_level': '4th Year',
            'section': 'A',
            'email': 'randel.bugay@microcity.edu.ph',
            'phone': '0960-409-8354'
        },
        {
            'student_id': '2210013M',
            'first_name': 'Cyrrenz',
            'last_name': 'Paguirigan',
            'course': 'Computer Engineering',
            'year_level': '4th Year',
            'section': 'A',
            'email': 'cy.paguirigan@microcity.edu.ph',
            'phone': '0969-123-6967'
        },
    ]
    
    print("\n" + "=" * 70)
    print("ADDING GROUP MEMBERS")
    print("=" * 70)
    
    success_count = 0
    for member in group_members:
        result = add_student(member)
        if result[0] is not None:
            success_count += 1
    
    print(f"\n✓ Successfully added {success_count}/{len(group_members)} students")


def add_single_student():
    """Interactive single student addition"""
    print("\n" + "=" * 70)
    print("ADD SINGLE STUDENT")
    print("=" * 70)
    
    student_info = {}
    print("\nEnter student information:")
    student_info['student_id'] = input("Student ID: ").strip()
    student_info['first_name'] = input("First Name: ").strip()
    student_info['last_name'] = input("Last Name: ").strip()
    student_info['course'] = input("Course: ").strip()
    student_info['year_level'] = input("Year Level: ").strip()
    student_info['section'] = input("Section: ").strip()
    student_info['email'] = input("Email (optional): ").strip()
    student_info['phone'] = input("Phone (optional): ").strip()
    
    add_student(student_info)


def list_students():
    """Get list of all active students"""
    conn = sqlite3.connect('database/faces.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT person_id, student_id, first_name, last_name, photo_count, status
        FROM persons
        WHERE status = 'active'
        ORDER BY person_id
    ''')
    
    students = cursor.fetchall()
    conn.close()
    return students


# ==================== IMAGE CAPTURE ====================

def capture_images_for_student(person_id, first_name, last_name, num_images=20):
    """Capture training images for a student"""
    folder_name = f"Person_{person_id}_{first_name}_{last_name}"
    folder_path = os.path.join("database", "training_images", folder_name)
    os.makedirs(folder_path, exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("✗ ERROR: Cannot open camera!")
            return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    if face_cascade.empty():
        print("✗ ERROR: Could not load face detector!")
        cap.release()
        return
    
    count = 0
    saved_count = 0
    
    print(f"\n{'=' * 70}")
    print(f"CAPTURING IMAGES FOR: {first_name} {last_name}")
    print(f"{'=' * 70}")
    print(f"Target: {num_images} images")
    print("\nInstructions:")
    print("  - Position face in the green rectangle")
    print("  - Images captured automatically")
    print("  - Press SPACE to manually capture")
    print("  - Press 'Q' to quit")
    print(f"{'=' * 70}\n")
    
    try:
        while saved_count < num_images:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5,
                minSize=(150, 150), maxSize=(400, 400)
            )
            
            display = frame.copy()
            
            for (x, y, w, h) in faces:
                color = (0, 255, 0) if len(faces) == 1 else (0, 165, 255)
                cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
                cv2.putText(display, f"{saved_count}/{num_images}",
                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            cv2.putText(display, f"Capturing: {first_name} {last_name}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(display, f"Progress: {saved_count}/{num_images}",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            if len(faces) == 0:
                cv2.putText(display, "No face detected",
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            elif len(faces) > 1:
                cv2.putText(display, "Multiple faces detected",
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            
            cv2.imshow("Capture Training Images", display)
            
            key = cv2.waitKey(1) & 0xFF
            
            count += 1
            if count % 30 == 0 and len(faces) == 1:
                x, y, w, h = faces[0]
                face_roi = frame[y:y + h, x:x + w]
                face_roi = cv2.resize(face_roi, (200, 200))
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"{first_name}_{last_name}_{saved_count + 1}_{timestamp}.jpg"
                filepath = os.path.join(folder_path, filename)
                
                cv2.imwrite(filepath, face_roi)
                saved_count += 1
                print(f"✓ Captured image {saved_count}/{num_images}")
                
                try:
                    conn = sqlite3.connect('database/faces.db')
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO faces (person_id, image_path)
                        VALUES (?, ?)
                    ''', (person_id, filepath))
                    conn.commit()
                    conn.close()
                except Exception as e:
                    print(f"  Warning: Database error: {e}")
            
            elif key == ord(' ') and len(faces) == 1:
                x, y, w, h = faces[0]
                face_roi = frame[y:y + h, x:x + w]
                face_roi = cv2.resize(face_roi, (200, 200))
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"{first_name}_{last_name}_{saved_count + 1}_{timestamp}.jpg"
                filepath = os.path.join(folder_path, filename)
                
                cv2.imwrite(filepath, face_roi)
                saved_count += 1
                print(f"✓ Manual capture {saved_count}/{num_images}")
                
                try:
                    conn = sqlite3.connect('database/faces.db')
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO faces (person_id, image_path)
                        VALUES (?, ?)
                    ''', (person_id, filepath))
                    conn.commit()
                    conn.close()
                except Exception as e:
                    print(f"  Warning: Database error: {e}")
            
            elif key == ord('q') or key == 27:
                print("\nCapture cancelled")
                break
    
    except KeyboardInterrupt:
        print("\nCapture interrupted")
    except Exception as e:
        print(f"\n✗ Error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    try:
        conn = sqlite3.connect('database/faces.db')
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE persons SET photo_count = ? WHERE person_id = ?
        ''', (saved_count, person_id))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Warning: Could not update photo count: {e}")
    
    print(f"\n{'=' * 70}")
    print(f"Capture complete! Saved {saved_count} images")
    print(f"{'=' * 70}\n")


# ==================== DELETE IMAGES ====================

def delete_student_images(person_id, first_name, last_name):
    """Delete all captured images for a specific student"""
    print(f"\n{'=' * 70}")
    print(f"DELETE IMAGES FOR: {first_name} {last_name}")
    print(f"{'=' * 70}")
    
    conn = sqlite3.connect('database/faces.db')
    cursor = conn.cursor()
    
    # Get all image paths for this student
    cursor.execute('''
        SELECT face_id, image_path FROM faces WHERE person_id = ?
    ''', (person_id,))
    
    images = cursor.fetchall()
    
    if not images:
        print(f"\n✓ No images found for {first_name} {last_name}")
        conn.close()
        return
    
    print(f"\nFound {len(images)} images for {first_name} {last_name}")
    confirm = input("Are you sure you want to delete ALL images? (yes/no): ").strip().lower()
    
    if confirm != 'yes':
        print("✗ Deletion cancelled")
        conn.close()
        return
    
    deleted_files = 0
    deleted_db = 0
    
    for face_id, image_path in images:
        # Delete physical file
        if os.path.exists(image_path):
            try:
                os.remove(image_path)
                deleted_files += 1
            except Exception as e:
                print(f"  Warning: Could not delete {image_path}: {e}")
        
        # Delete database record
        try:
            cursor.execute('DELETE FROM faces WHERE face_id = ?', (face_id,))
            deleted_db += 1
        except Exception as e:
            print(f"  Warning: Could not delete record {face_id}: {e}")
    
    # Update photo count
    cursor.execute('''
        UPDATE persons SET photo_count = 0 WHERE person_id = ?
    ''', (person_id,))
    
    conn.commit()
    conn.close()
    
    print(f"\n✓ Deleted {deleted_files} image files")
    print(f"✓ Deleted {deleted_db} database records")
    print(f"✓ Ready to capture new images for {first_name} {last_name}")


# ==================== TRAIN MODEL ====================

def train_model():
    """Train the face recognition model"""
    print("\n" + "=" * 70)
    print("TRAINING FACE RECOGNITION MODEL")
    print("=" * 70)
    
    conn = sqlite3.connect('database/faces.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT f.person_id, f.image_path, p.first_name, p.last_name
        FROM faces f
        JOIN persons p ON f.person_id = p.person_id
        WHERE p.status = 'active'
        ORDER BY f.person_id
    ''')
    
    data = cursor.fetchall()
    
    if not data:
        print("\n✗ No training data found!")
        conn.close()
        return False
    
    print(f"\nFound {len(data)} images in database")
    print("Loading and preprocessing images...")
    
    faces = []
    labels = []
    names = {}
    
    loaded_count = 0
    error_count = 0
    person_counts = {}
    
    for person_id, image_path, first_name, last_name in data:
        full_name = f"{first_name} {last_name}"
        names[person_id] = full_name
        person_counts[person_id] = person_counts.get(person_id, 0) + 1
        
        if not os.path.exists(image_path):
            error_count += 1
            continue
        
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                error_count += 1
                continue
            
            img = cv2.resize(img, (200, 200))
            img = cv2.equalizeHist(img)
            
            faces.append(img)
            labels.append(person_id)
            loaded_count += 1
            
        except Exception as e:
            print(f"  ✗ Error loading {image_path}: {e}")
            error_count += 1
    
    print(f"\n✓ Loaded: {loaded_count} images")
    if error_count > 0:
        print(f"✗ Failed: {error_count} images")
    
    print(f"\nTraining data summary:")
    for person_id, count in sorted(person_counts.items()):
        status = "✓" if count >= 10 else "⚠"
        print(f"  {status} {names[person_id]}: {count} images")
    
    if len(faces) < 5:
        print("\n✗ Not enough training data (need at least 5 images)")
        conn.close()
        return False
    
    print("\nTraining LBPH model...")
    
    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1, neighbors=8, grid_x=8, grid_y=8
    )
    
    try:
        recognizer.train(faces, np.array(labels))
        print("✓ Training complete!")
        
        model_path = 'database/trained_model.yml'
        recognizer.write(model_path)
        print(f"✓ Model saved: {model_path}")
        
        mapping_path = 'database/name_mapping.pkl'
        with open(mapping_path, 'wb') as f:
            pickle.dump(names, f)
        print(f"✓ Name mapping saved: {mapping_path}")
        
        cursor.execute('''
            INSERT INTO training_history (total_persons, total_faces, model_path)
            VALUES (?, ?, ?)
        ''', (len(set(labels)), len(faces), model_path))
        conn.commit()
        
        print("\n" + "=" * 70)
        print("TRAINING SUCCESSFUL!")
        print(f"Total persons: {len(set(labels))}")
        print(f"Total images: {len(faces)}")
        print("=" * 70)
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        conn.close()
        return False


# ==================== VIEW DATABASE ====================

def view_all_students():
    """Display all registered students"""
    print("\n" + "=" * 70)
    print("REGISTERED STUDENTS")
    print("=" * 70)
    
    conn = sqlite3.connect('database/faces.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT person_id, student_id, first_name || ' ' || last_name,
               course, year_level, section, photo_count, status
        FROM persons ORDER BY person_id
    ''')
    
    students = cursor.fetchall()
    
    if students:
        print("\n{:<5} {:<15} {:<25} {:<20} {:<8} {:<8} {:<8}".format(
            "ID", "Student ID", "Name", "Course", "Year", "Photos", "Status"
        ))
        print("-" * 100)
        
        for row in students:
            print("{:<5} {:<15} {:<25} {:<20} {:<8} {:<8} {:<8}".format(*row))
        
        print(f"\nTotal: {len(students)} students")
    else:
        print("\nNo students found")
    
    conn.close()


def view_attendance_logs():
    """Display recent attendance logs"""
    print("\n" + "=" * 70)
    print("ATTENDANCE LOGS")
    print("=" * 70)
    
    conn = sqlite3.connect('database/faces.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT 
            a.log_id,
            p.first_name || ' ' || p.last_name as name,
            a.time_in,
            ROUND(a.confidence, 1) as conf,
            a.status
        FROM attendance_logs a
        JOIN persons p ON a.person_id = p.person_id
        ORDER BY a.time_in DESC
        LIMIT 30
    ''')
    
    logs = cursor.fetchall()
    
    if logs:
        print("\n{:<8} {:<25} {:<25} {:<8} {:<10}".format(
            "Log ID", "Student", "Time In", "Conf%", "Status"
        ))
        print("-" * 85)
        
        for row in logs:
            print("{:<8} {:<25} {:<25} {:<8} {:<10}".format(*row))
        
        cursor.execute('SELECT COUNT(*) FROM attendance_logs')
        total = cursor.fetchone()[0]
        print(f"\nShowing latest 30 of {total} total records")
    else:
        print("\nNo attendance logs found")
    
    conn.close()


def view_database_stats():
    """Display database statistics"""
    print("\n" + "=" * 70)
    print("DATABASE STATISTICS")
    print("=" * 70)
    
    conn = sqlite3.connect('database/faces.db')
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM persons")
    print(f"\nStudents: {cursor.fetchone()[0]}")
    
    cursor.execute("SELECT COUNT(*) FROM faces")
    print(f"Face Images: {cursor.fetchone()[0]}")
    
    cursor.execute("SELECT COUNT(*) FROM attendance_logs")
    print(f"Attendance Logs: {cursor.fetchone()[0]}")
    
    if os.path.exists('database/faces.db'):
        db_size = os.path.getsize('database/faces.db') / (1024 * 1024)
        print(f"Database Size: {db_size:.2f} MB")
    
    if os.path.exists('database/trained_model.yml'):
        print("\n✓ Trained model exists")
    else:
        print("\n⚠ No trained model found")
    
    conn.close()


# ==================== MAIN MENU ====================

def main_menu():
    """Main interactive menu"""
    
    while True:
        print("\n" + "=" * 70)
        print(" FACE RECOGNITION DATABASE MANAGER ".center(70, "="))
        print(" Microcity College of Business and Technology ".center(70))
        print("=" * 70)
        print("\n DATABASE SETUP")
        print("  1. Setup Database (First Time Only)")
        print("\n STUDENT MANAGEMENT")
        print("  2. Add All Group Members")
        print("  3. Add Single Student")
        print("  4. View All Students")
        print("\n IMAGE MANAGEMENT")
        print("  5. Capture Images for Student")
        print("  6. Delete Student Images (Re-capture)")
        print("\n MODEL TRAINING")
        print("  7. Train Recognition Model")
        print("\n VIEW DATA")
        print("  8. View Attendance Logs")
        print("  9. View Database Statistics")
        print("\n  0. Exit")
        print("=" * 70)
        
        try:
            choice = input("\nEnter choice: ").strip()
            
            if choice == '1':
                setup_database()
            
            elif choice == '2':
                if not os.path.exists('database/faces.db'):
                    print("\n✗ Database not found! Run option 1 first")
                else:
                    add_group_members()
            
            elif choice == '3':
                if not os.path.exists('database/faces.db'):
                    print("\n✗ Database not found! Run option 1 first")
                else:
                    add_single_student()
            
            elif choice == '4':
                if not os.path.exists('database/faces.db'):
                    print("\n✗ Database not found! Run option 1 first")
                else:
                    view_all_students()
            
            elif choice == '5':
                if not os.path.exists('database/faces.db'):
                    print("\n✗ Database not found! Run option 1 first")
                else:
                    students = list_students()
                    if not students:
                        print("\n✗ No students found! Add students first")
                    else:
                        print("\nRegistered Students:")
                        for i, (pid, sid, fname, lname, count, status) in enumerate(students, 1):
                            print(f"  {i}. {fname} {lname} (ID: {sid}) - {count} photos")
                        
                        try:
                            idx = int(input("\nSelect student number: ")) - 1
                            if 0 <= idx < len(students):
                                pid, sid, fname, lname, count, status = students[idx]
                                num_imgs = int(input(f"\nNumber of images to capture (default 20): ") or "20")
                                capture_images_for_student(pid, fname, lname, num_imgs)
                            else:
                                print("✗ Invalid selection")
                        except ValueError:
                            print("✗ Invalid input")
            
            elif choice == '6':
                if not os.path.exists('database/faces.db'):
                    print("\n✗ Database not found! Run option 1 first")
                else:
                    students = list_students()
                    if not students:
                        print("\n✗ No students found!")
                    else:
                        print("\nSelect student to DELETE images:")
                        for i, (pid, sid, fname, lname, count, status) in enumerate(students, 1):
                            print(f"  {i}. {fname} {lname} - {count} photos")
                        
                        try:
                            idx = int(input("\nSelect student number: ")) - 1
                            if 0 <= idx < len(students):
                                pid, sid, fname, lname, count, status = students[idx]
                                delete_student_images(pid, fname, lname)
                            else:
                                print("✗ Invalid selection")
                        except ValueError:
                            print("✗ Invalid input")
            
            elif choice == '7':
                if not os.path.exists('database/faces.db'):
                    print("\n✗ Database not found! Run option 1 first")
                else:
                    train_model()
            
            elif choice == '8':
                if not os.path.exists('database/faces.db'):
                    print("\n✗ Database not found! Run option 1 first")
                else:
                    view_attendance_logs()
            
            elif choice == '9':
                if not os.path.exists('database/faces.db'):
                    print("\n✗ Database not found! Run option 1 first")
                else:
                    view_database_stats()
            
            elif choice == '0':
                print("\n" + "=" * 70)
                print("Thank you for using the Database Manager!")
                print("Microcity College of Business and Technology")
                print("=" * 70 + "\n")
                break
            
            else:
                print("\n✗ Invalid choice! Please select 0-9")
            
            input("\nPress Enter to continue...")
        
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\n✗ Error: {e}")
            input("\nPress Enter to continue...")


if __name__ == "__main__":
    main_menu()