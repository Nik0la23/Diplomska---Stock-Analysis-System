"""
Database Initialization Script
Creates SQLite database with complete schema for stock analysis system.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.database.db_manager import init_database, verify_database, get_connection
from src.utils.logger import setup_logger
import sqlite3

# Setup logging
logger = setup_logger(__name__)


def main():
    """
    Initialize the SQLite database with full schema.
    
    Steps:
    1. Create data directory if needed
    2. Execute schema.sql to create all tables, views, indexes
    3. Verify database structure
    4. Print summary
    """
    print("=" * 80)
    print("LangGraph Stock Analysis - Database Setup")
    print("=" * 80)
    
    # Database path
    db_path = Path("data/stock_prices.db")
    
    # Create data directory
    db_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Database location: {db_path.absolute()}")
    
    # Check if database already exists
    if db_path.exists():
        print(f"⚠️  Database already exists. Schema will be updated if needed.")
    
    # Initialize database
    print("\n🔧 Initializing database schema...")
    try:
        init_database(
            schema_path="src/database/schema.sql",
            db_path=str(db_path)
        )
        print("✅ Schema initialization complete")
    except Exception as e:
        print(f"❌ Failed to initialize database: {str(e)}")
        sys.exit(1)
    
    # Verify database structure
    print("\n🔍 Verifying database structure...")
    verification = verify_database(str(db_path))
    
    if verification['status'] == 'OK':
        print("✅ Database verification successful\n")
        
        # Print tables
        print(f"📊 Created {verification['table_count']} tables:")
        for table in verification['tables']:
            print(f"   - {table}")
        
        # Print views
        print(f"\n👁️  Created {verification['view_count']} views:")
        for view in verification['views']:
            print(f"   - {view}")
        
        # Print indexes
        print(f"\n⚡ Created {len(verification['indexes'])} indexes for performance")
        
        # Verify critical tables for Node 8
        critical_tables = ['news_outcomes', 'source_reliability', 'news_articles']
        print(f"\n🎓 Thesis Innovation - Node 8 Critical Tables:")
        for table in critical_tables:
            if table in verification['tables']:
                print(f"   ✅ {table}")
            else:
                print(f"   ❌ {table} - MISSING!")
        
        # Test basic query
        print(f"\n🧪 Testing database connectivity...")
        try:
            with get_connection(str(db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
                count = cursor.fetchone()[0]
                print(f"   ✅ Connection successful - {count} tables accessible")
        except Exception as e:
            print(f"   ❌ Connection test failed: {str(e)}")
        
        print("\n" + "=" * 80)
        print("✅ DATABASE SETUP COMPLETE")
        print("=" * 80)
        print("\nNext steps:")
        print("1. Copy .env.example to .env and add your API keys")
        print("2. Install dependencies: pip install -r requirements.txt")
        print("3. Start building nodes following NODE_BUILD_GUIDE.md")
        print("4. Run tests: pytest tests/ -v")
        print("=" * 80)
        
    else:
        print(f"❌ Database verification failed: {verification.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
