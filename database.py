import sqlite3
import hashlib
from typing import Dict, List
import pandas as pd
class UserDB:
    def __init__(self):
        self.db_file = "users.db"
        self.uploads_dir = "user_uploads"  # Thư mục lưu file uploads
        self.create_tables()
        self.create_uploads_directory()
    
    def create_tables(self):
        """Create user and file tables"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        # Existing user table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                full_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # New table for file uploads
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                filename TEXT NOT NULL,
                original_filename TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_size INTEGER,
                upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT,
                FOREIGN KEY (username) REFERENCES users (username)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_uploads_directory(self):
        """Create uploads directory if not exists"""
        import os
        if not os.path.exists(self.uploads_dir):
            os.makedirs(self.uploads_dir)
            # Create user subdirectories as needed
    
    def save_user_file(self, username: str, uploaded_file, metadata: Dict = None) -> str:
        """Save uploaded file to filesystem and metadata to database"""
        import os
        import hashlib
        import json
        from datetime import datetime
        
        try:
            # Create user-specific directory
            user_dir = os.path.join(self.uploads_dir, username)
            if not os.path.exists(user_dir):
                os.makedirs(user_dir)
            
            # Generate unique filename to avoid conflicts
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_hash = hashlib.md5(uploaded_file.name.encode()).hexdigest()[:8]
            safe_filename = f"{timestamp}_{file_hash}_{uploaded_file.name}"
            file_path = os.path.join(user_dir, safe_filename)
            
            # Save file to filesystem
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Save metadata to database
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            metadata_json = json.dumps(metadata) if metadata else None
            
            cursor.execute('''
                INSERT INTO user_files (username, filename, original_filename, file_path, file_size, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (username, safe_filename, uploaded_file.name, file_path, uploaded_file.size, metadata_json))
            
            conn.commit()
            conn.close()
            
            return file_path
            
        except Exception as e:
            print(f"Error saving file: {str(e)}")
            return None
    
    def get_user_files(self, username: str) -> List[Dict]:
        """Get all files for a specific user"""
        import json
        import os
        
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT filename, original_filename, file_path, file_size, upload_time, metadata
            FROM user_files 
            WHERE username = ?
            ORDER BY upload_time DESC
        ''', (username,))
        
        files = []
        for row in cursor.fetchall():
            filename, original_filename, file_path, file_size, upload_time, metadata_json = row
            
            # Check if file still exists
            if os.path.exists(file_path):
                metadata = json.loads(metadata_json) if metadata_json else {}
                files.append({
                    'filename': filename,
                    'original_filename': original_filename,
                    'file_path': file_path,
                    'file_size': file_size,
                    'upload_time': upload_time,
                    'metadata': metadata
                })
        
        conn.close()
        return files
    
    def delete_user_file(self, username: str, filename: str) -> bool:
        """Delete a specific file for user"""
        import os
        
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        # Get file path first
        cursor.execute('''
            SELECT file_path FROM user_files 
            WHERE username = ? AND filename = ?
        ''', (username, filename))
        
        result = cursor.fetchone()
        if result:
            file_path = result[0]
            
            # Delete from filesystem
            if os.path.exists(file_path):
                os.remove(file_path)
            
            # Delete from database
            cursor.execute('''
                DELETE FROM user_files 
                WHERE username = ? AND filename = ?
            ''', (username, filename))
            
            conn.commit()
            conn.close()
            return True
        
        conn.close()
        return False
    
    def load_user_file(self, file_path: str) -> pd.DataFrame:
        """Load file from filesystem back to DataFrame"""
        try:
            file_extension = file_path.split('.')[-1].lower()
            
            if file_extension == 'csv':
                return pd.read_csv(file_path)
            elif file_extension in ['xlsx', 'xls']:
                return pd.read_excel(file_path)
            else:
                return None
        except Exception as e:
            print(f"Error loading file: {str(e)}")
            return None

    def __init__(self, db_path: str = "users.db"):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                full_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tạo user admin mặc định
        admin_password = self.hash_password("admin123")
        c.execute('''
            INSERT OR IGNORE INTO users (username, password, full_name) 
            VALUES (?, ?, ?)
        ''', ("admin", admin_password, "Administrator"))
        
        conn.commit()
        conn.close()
    
    def hash_password(self, password: str) -> str:
        return hashlib.sha256(password.encode()).hexdigest()
    
    def verify_user(self, username: str, password: str) -> bool:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        hashed_password = self.hash_password(password)
        c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, hashed_password))
        result = c.fetchone()
        conn.close()
        return result is not None
    
    def get_user_info(self, username: str):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT username, full_name FROM users WHERE username=?", (username,))
        result = c.fetchone()
        conn.close()
        if result:
            return {'username': result[0], 'full_name': result[1]}
        return None
    def create_user(self, username: str, password: str, full_name: str = "") -> bool:
        """Tạo user mới"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            hashed_password = self.hash_password(password)
            c.execute(
                "INSERT INTO users (username, password, full_name) VALUES (?, ?, ?)",
                (username, hashed_password, full_name)
            )
            conn.commit()
            conn.close()
            return True
        except sqlite3.IntegrityError:
            return False  # Username đã tồn tại

