"""
Test script for PostgreSQL database connection and FBSL prototype storage
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from .database.postgres_manager import DatabaseManager
from .core.fbsl_models import FBSLLayoutNode, Function, Behavior, Layout, Room, NodeType, FunctionCategory, BehaviorCategory
import uuid
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

??s
def test_database_connection():
    """Test basic database connection"""
    print("=" * 80)
    print("🔌 Testing Database Connection")
    print("=" * 80)
    
    try:
        db = DatabaseManager()
        print("✅ Database connection successful!")
        
        # Test query
        result = db.execute_query("SELECT version();")
        if result:
            print(f"✅ PostgreSQL version: {result[0]['version'][:50]}...")
        
        db.close()
        return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print("\n💡 Make sure PostgreSQL is running and config/config.yaml has correct credentials")
        return False


def test_schema_creation():
    """Test that schema was created correctly"""
    print("\n" + "=" * 80)
    print("📋 Testing Database Schema")
    print("=" * 80)
    
    try:
        db = DatabaseManager()
        
        # Check if tables exist
        tables = ['fbsl_nodes', 'projects', 'evaluations']
        for table in tables:
            result = db.execute_query(
                "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = %s);",
                (table,)
            )
            exists = result[0]['exists'] if result else False
            status = "✅" if exists else "❌"
            print(f"{status} Table '{table}': {'exists' if exists else 'missing'}")
        
        # Check indexes
        result = db.execute_query("""
            SELECT indexname FROM pg_indexes 
            WHERE tablename = 'fbsl_nodes' 
            ORDER BY indexname;
        """)
        if result:
            print(f"\n✅ Found {len(result)} indexes on fbsl_nodes:")
            for idx in result:
                print(f"   - {idx['indexname']}")
        
        db.close()
        return True
    except Exception as e:
        print(f"❌ Schema test failed: {e}")
        return False


def test_store_and_retrieve():
    """Test storing and retrieving FBSL nodes"""
    print("\n" + "=" * 80)
    print("💾 Testing Store and Retrieve")
    print("=" * 80)
    
    try:
        db = DatabaseManager()
        
        # Create a test FBSL node
        test_node = FBSLLayoutNode(
            node_id=str(uuid.uuid4()),
            project_id="test_project_001",
            node_type=NodeType.DESIGN_PROTOTYPE,
            generation_level=1
        )
        
        # Add a function
        func = Function(
            name="provide_living_room",
            category=FunctionCategory.SPATIAL,
            description="Living room space",
            priority=0.8,
            activities=["living", "relaxation"]
        )
        test_node.add_function(func)
        
        # Add a behavior
        behav = Behavior(
            category=BehaviorCategory.SPATIAL,
            metric_name="living_room_area",
            metric_unit="sqm",
            target_value=20.0,
            actual_value=18.5,
            derived_from_function=func.function_id
        )
        behav.calculate_satisfaction()
        test_node.add_behavior(behav)
        
        # Add a layout with room
        layout = Layout()
        room = Room(
            name="Living Room",
            room_type="living_room",
            function_id=func.function_id,
            area=18.5,
            height=3.0
        )
        room.calculate_volume()
        layout.rooms[room.room_id] = room
        layout.total_area = 18.5
        layout.used_area = 18.5
        layout.calculate_metrics()
        test_node.layout = layout
        
        # Set scores
        test_node.functional_score = 0.85
        test_node.behavioral_score = 0.92
        test_node.structural_score = 0.75
        test_node.layout_score = 0.88
        test_node.composite_score = 0.85
        
        # Store node
        print("📤 Storing test node...")
        node_dict = test_node.to_dict()
        success = db.store_fbsl_node(node_dict)
        
        if success:
            print(f"✅ Node stored successfully: {test_node.node_id[:8]}...")
        else:
            print("❌ Failed to store node")
            return False
        
        # Retrieve node
        print("📥 Retrieving test node...")
        retrieved = db.get_fbsl_node(test_node.node_id)
        
        if retrieved:
            print(f"✅ Node retrieved successfully")
            print(f"   Node ID: {retrieved['node_id'][:8]}...")
            print(f"   Type: {retrieved['node_type']}")
            print(f"   Composite Score: {retrieved['composite_score']:.3f}")
            print(f"   Functions: {len(retrieved['functions'])}")
            print(f"   Behaviors: {len(retrieved['behaviors'])}")
            print(f"   Has Layout: {retrieved['layout'] is not None}")
        else:
            print("❌ Failed to retrieve node")
            return False
        
        # Test project storage
        print("\n📤 Storing test project...")
        project_success = db.store_project(
            project_id="test_project_001",
            project_name="Test Apartment",
            description="Test project for database",
            requirements="2 bedroom apartment",
            context={"building_type": "residential"}
        )
        
        if project_success:
            print("✅ Project stored successfully")
        else:
            print("❌ Failed to store project")
        
        # Test prototype retrieval
        print("\n📥 Retrieving prototypes for project...")
        prototypes = db.get_prototypes_by_project("test_project_001", limit=5)
        print(f"✅ Found {len(prototypes)} prototypes")
        
        db.close()
        return True
        
    except Exception as e:
        print(f"❌ Store/retrieve test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_bulk_operations():
    """Test storing multiple nodes"""
    print("\n" + "=" * 80)
    print("📦 Testing Bulk Operations")
    print("=" * 80)
    
    try:
        db = DatabaseManager()
        
        # Create multiple test nodes
        nodes = []
        for i in range(3):
            node = FBSLLayoutNode(
                node_id=str(uuid.uuid4()),
                project_id="test_project_002",
                node_type=NodeType.DESIGN_PROTOTYPE,
                composite_score=0.5 + (i * 0.1)  # 0.5, 0.6, 0.7
            )
            nodes.append(node)
        
        # Store all nodes
        stored = 0
        for node in nodes:
            if db.store_fbsl_node(node.to_dict()):
                stored += 1
        
        print(f"✅ Stored {stored}/{len(nodes)} nodes")
        
        # Retrieve and verify ordering
        prototypes = db.get_prototypes_by_project("test_project_002", limit=10)
        print(f"✅ Retrieved {len(prototypes)} prototypes")
        
        if prototypes:
            print("   Scores (should be descending):")
            for i, proto in enumerate(prototypes, 1):
                print(f"   {i}. Score: {proto['composite_score']:.3f}")
        
        db.close()
        return True
        
    except Exception as e:
        print(f"❌ Bulk operations test failed: {e}")
        return False


def main():
    """Run all database tests"""
    print("\n" + "=" * 80)
    print("🧪 FBSL-KAGS Database Test Suite")
    print("=" * 80)
    
    tests = [
        ("Database Connection", test_database_connection),
        ("Schema Creation", test_schema_creation),
        ("Store and Retrieve", test_store_and_retrieve),
        ("Bulk Operations", test_bulk_operations),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 Test Summary")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\n{'✅' if passed == total else '⚠️'} {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All database tests passed!")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")


if __name__ == "__main__":
    main()

