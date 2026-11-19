"""
SQLite Database Manager for AI-Based Crop Health Monitoring System
Stores scan histories, predictions, severity metrics, and advisory logs.
"""

import sqlite3
import datetime
import os
from pathlib import Path
from config import DATABASE_PATH

class DatabaseManager:
    def __init__(self, db_path=None):
        self.db_path = str(db_path or DATABASE_PATH)
        # Ensure parent directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.init_database()

    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_database(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS scan_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    crop_name TEXT NOT NULL,
                    disease_name TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    severity_level TEXT NOT NULL,
                    lesion_percentage REAL NOT NULL,
                    image_filename TEXT,
                    alert_level TEXT NOT NULL,
                    treatment_summary TEXT,
                    notes TEXT,
                    action_status TEXT DEFAULT 'Pending'
                )
            """)
            conn.commit()

    def insert_scan(self, crop_name, disease_name, confidence, severity_level, 
                    lesion_percentage, image_filename, alert_level, treatment_summary, 
                    notes="", action_status="Pending"):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO scan_history (
                    timestamp, crop_name, disease_name, confidence, 
                    severity_level, lesion_percentage, image_filename, 
                    alert_level, treatment_summary, notes, action_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp, crop_name, disease_name, round(confidence, 2),
                severity_level, round(lesion_percentage, 2), image_filename,
                alert_level, treatment_summary, notes, action_status
            ))
            conn.commit()
            return cursor.lastrowid

    def get_all_scans(self, limit=100, crop_filter=None, status_filter=None):
        query = "SELECT * FROM scan_history WHERE 1=1"
        params = []
        if crop_filter and crop_filter != "All":
            query += " AND crop_name = ?"
            params.append(crop_filter)
        if status_filter and status_filter != "All":
            query += " AND action_status = ?"
            params.append(status_filter)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_scan_by_id(self, scan_id):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM scan_history WHERE id = ?", (scan_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def update_action_status(self, scan_id, new_status, notes=None):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if notes is not None:
                cursor.execute("""
                    UPDATE scan_history 
                    SET action_status = ?, notes = ?
                    WHERE id = ?
                """, (new_status, notes, scan_id))
            else:
                cursor.execute("""
                    UPDATE scan_history 
                    SET action_status = ?
                    WHERE id = ?
                """, (new_status, scan_id))
            conn.commit()
            return cursor.rowcount > 0

    def delete_scan(self, scan_id):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM scan_history WHERE id = ?", (scan_id,))
            conn.commit()
            return cursor.rowcount > 0

    def clear_all_history(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM scan_history")
            conn.commit()

    def get_analytics_summary(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM scan_history")
            total_scans = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM scan_history WHERE alert_level = 'Safe'")
            healthy_scans = cursor.fetchone()[0]

            infected_scans = total_scans - healthy_scans

            cursor.execute("""
                SELECT disease_name, COUNT(*) as count 
                FROM scan_history 
                GROUP BY disease_name 
                ORDER BY count DESC
            """)
            disease_distribution = [dict(row) for row in cursor.fetchall()]

            cursor.execute("""
                SELECT severity_level, COUNT(*) as count 
                FROM scan_history 
                GROUP BY severity_level
            """)
            severity_distribution = [dict(row) for row in cursor.fetchall()]

            cursor.execute("""
                SELECT crop_name, COUNT(*) as count 
                FROM scan_history 
                GROUP BY crop_name
            """)
            crop_distribution = [dict(row) for row in cursor.fetchall()]

            cursor.execute("""
                SELECT SUBSTR(timestamp, 1, 10) as scan_date, COUNT(*) as count 
                FROM scan_history 
                GROUP BY scan_date 
                ORDER BY scan_date ASC 
                LIMIT 14
            """)
            daily_scans = [dict(row) for row in cursor.fetchall()]

            return {
                "total_scans": total_scans,
                "healthy_scans": healthy_scans,
                "infected_scans": infected_scans,
                "disease_distribution": disease_distribution,
                "severity_distribution": severity_distribution,
                "crop_distribution": crop_distribution,
                "daily_scans": daily_scans
            }
