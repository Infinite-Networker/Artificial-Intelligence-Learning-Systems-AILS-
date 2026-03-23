"""
AILS Database Manager Module
MySQL and NoSQL integration for persistent data storage.
Created by Cherry Computer Ltd.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple


class AILSDatabaseManager:
    """
    AILS MySQL Database Manager.
    Handles connection, table creation, CRUD operations, and batch inserts.

    Example:
        db = AILSDatabaseManager(host="localhost", user="root",
                                  password="pass", database="AILS_data")
        if db.connect():
            db.create_table("reviews", "id INT AUTO_INCREMENT PRIMARY KEY, "
                            "text TEXT, sentiment VARCHAR(10)")
            db.insert_many("reviews", ["text", "sentiment"],
                           [("Great product!", "positive")])
            db.close()
    """

    def __init__(self, host: str = "localhost", user: str = "root",
                 password: str = "", database: str = "AILS_data",
                 port: int = 3306):
        self.config = {
            "host": host,
            "user": user,
            "password": password,
            "database": database,
            "port": port,
        }
        self.connection = None
        self.logger = logging.getLogger("AILS.Database")

    def connect(self) -> bool:
        """Establish a database connection."""
        try:
            import mysql.connector
            self.connection = mysql.connector.connect(**self.config)
            if self.connection.is_connected():
                self.logger.info(
                    f"✅ Connected to MySQL database '{self.config['database']}'"
                )
                return True
        except Exception as e:
            self.logger.error(f"❌ Database connection failed: {e}")
        return False

    def create_database(self, db_name: str) -> None:
        """Create a database if it doesn't exist."""
        cursor = self.connection.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_name}")
        self.logger.info(f"Database '{db_name}' ensured.")

    def create_table(self, table_name: str, schema: str) -> None:
        """
        Create a table if it doesn't already exist.

        Args:
            table_name: Name of the table.
            schema: SQL column definitions string.
        """
        cursor = self.connection.cursor()
        cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({schema})")
        self.connection.commit()
        self.logger.info(f"Table '{table_name}' is ready.")

    def insert_one(self, table_name: str,
                   columns: List[str], values: Tuple) -> int:
        """
        Insert a single record.

        Returns:
            The last inserted row ID.
        """
        cursor = self.connection.cursor()
        placeholders = ", ".join(["%s"] * len(columns))
        col_str = ", ".join(columns)
        sql = f"INSERT INTO {table_name} ({col_str}) VALUES ({placeholders})"
        cursor.execute(sql, values)
        self.connection.commit()
        return cursor.lastrowid

    def insert_many(self, table_name: str,
                    columns: List[str],
                    records: List[Tuple]) -> None:
        """
        Batch insert multiple records using executemany (optimized).

        Args:
            table_name: Target table.
            columns: List of column names.
            records: List of value tuples.
        """
        cursor = self.connection.cursor()
        placeholders = ", ".join(["%s"] * len(columns))
        col_str = ", ".join(columns)
        sql = f"INSERT INTO {table_name} ({col_str}) VALUES ({placeholders})"
        cursor.executemany(sql, records)
        self.connection.commit()
        self.logger.info(
            f"✅ Batch inserted {len(records)} records into '{table_name}'."
        )

    def fetch_all(self, table_name: str,
                  condition: Optional[str] = None,
                  limit: Optional[int] = None) -> List[Tuple]:
        """
        Retrieve all records from a table.

        Args:
            table_name: Table to query.
            condition: Optional WHERE clause (e.g., "sentiment='positive'").
            limit: Optional LIMIT clause.

        Returns:
            List of row tuples.
        """
        cursor = self.connection.cursor()
        query = f"SELECT * FROM {table_name}"
        if condition:
            query += f" WHERE {condition}"
        if limit:
            query += f" LIMIT {limit}"
        cursor.execute(query)
        return cursor.fetchall()

    def fetch_as_dict(self, table_name: str,
                      condition: Optional[str] = None) -> List[Dict]:
        """Retrieve records as list of dictionaries."""
        cursor = self.connection.cursor(dictionary=True)
        query = f"SELECT * FROM {table_name}"
        if condition:
            query += f" WHERE {condition}"
        cursor.execute(query)
        return cursor.fetchall()

    def update(self, table_name: str,
               set_clause: str, condition: str) -> int:
        """
        Update records matching a condition.

        Returns:
            Number of affected rows.
        """
        cursor = self.connection.cursor()
        sql = f"UPDATE {table_name} SET {set_clause} WHERE {condition}"
        cursor.execute(sql)
        self.connection.commit()
        return cursor.rowcount

    def delete(self, table_name: str, condition: str) -> int:
        """
        Delete records matching a condition.

        Returns:
            Number of deleted rows.
        """
        cursor = self.connection.cursor()
        sql = f"DELETE FROM {table_name} WHERE {condition}"
        cursor.execute(sql)
        self.connection.commit()
        return cursor.rowcount

    def count(self, table_name: str,
              condition: Optional[str] = None) -> int:
        """Return the count of records in a table."""
        cursor = self.connection.cursor()
        query = f"SELECT COUNT(*) FROM {table_name}"
        if condition:
            query += f" WHERE {condition}"
        cursor.execute(query)
        result = cursor.fetchone()
        return result[0] if result else 0

    def close(self) -> None:
        """Close the database connection safely."""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            self.logger.info("Database connection closed.")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class AILSNoSQLManager:
    """
    AILS NoSQL Manager — MongoDB integration.
    For flexible, schema-less document storage.

    Example:
        db = AILSNoSQLManager(uri="mongodb://localhost:27017", database="AILS")
        db.insert("articles", {"title": "AI News", "content": "..."})
    """

    def __init__(self, uri: str = "mongodb://localhost:27017",
                 database: str = "AILS"):
        try:
            from pymongo import MongoClient
            self.client = MongoClient(uri)
            self.db = self.client[database]
            self.logger = logging.getLogger("AILS.NoSQL")
        except ImportError:
            raise ImportError("pymongo is required: pip install pymongo")

    def insert(self, collection: str, document: Dict) -> str:
        """Insert a single document and return its ID."""
        result = self.db[collection].insert_one(document)
        return str(result.inserted_id)

    def insert_many(self, collection: str,
                    documents: List[Dict]) -> List[str]:
        """Batch insert documents."""
        result = self.db[collection].insert_many(documents)
        return [str(id_) for id_ in result.inserted_ids]

    def find(self, collection: str,
             query: Optional[Dict] = None,
             limit: int = 0) -> List[Dict]:
        """Find documents matching a query."""
        cursor = self.db[collection].find(query or {}).limit(limit)
        return list(cursor)

    def update_one(self, collection: str,
                   query: Dict, update: Dict) -> int:
        """Update first matching document. Returns modified count."""
        result = self.db[collection].update_one(query, {"$set": update})
        return result.modified_count

    def delete_many(self, collection: str, query: Dict) -> int:
        """Delete documents matching a query."""
        result = self.db[collection].delete_many(query)
        return result.deleted_count

    def close(self) -> None:
        self.client.close()
