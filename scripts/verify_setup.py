#!/usr/bin/env python3
import os
import sys

# ✅ Add backend directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

def verify_setup():
    checks = []

    # === Check Python packages ===
    try:
        import fastapi
        checks.append(("✓", "FastAPI installed"))
    except Exception:
        checks.append(("✗", "FastAPI missing"))

    try:
        import chromadb
        checks.append(("✓", "ChromaDB installed"))
    except Exception:
        checks.append(("✗", "ChromaDB missing"))

    # === Check PostgreSQL connection ===
    try:
        from database.postgres_manager import DatabaseManager
        db = DatabaseManager()
        db.connect()
        checks.append(("✓", "Database connection successful"))
        db.close()
    except Exception as e:
        checks.append(("✗", f"Database connection failed: {e}"))

    # === Check Vector Store ===
    try:
        from database.vector_store import VectorStoreManager
        vs = VectorStoreManager()
        checks.append(("✓", "Vector store initialized"))
    except Exception as e:
        checks.append(("✗", f"Vector store initialization failed: {e}"))

    # === Print Results ===
    print("\n=== Setup Verification ===\n")
    for symbol, message in checks:
        print(f"{symbol} {message}")
    print("\n")

if __name__ == "__main__":
    verify_setup()
