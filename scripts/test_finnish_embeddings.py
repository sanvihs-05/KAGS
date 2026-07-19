import sys
sys.path.append('backend')

from database.vector_store import VectorStoreManager

def test_finnish_embeddings():
    print("=== Testing Finnish Floor Plan Embeddings ===\n")
    
    # Initialize
    vs = VectorStoreManager()
    
    if vs.finnish_embeddings is None:
        print("✗ No Finnish embeddings loaded")
        return
    
    # Get statistics
    stats = vs.finnish_embeddings.get_plan_statistics()
    print(f"✓ Total annotations: {stats['total_annotations']}")
    print(f"✓ Total plans: {stats['total_plans']}")
    print(f"✓ Embedding types: {stats['embedding_types']}")
    print(f"\n✓ Room type distribution:")
    for room_type, count in list(stats.get('room_type_distribution', {}).items())[:10]:
        english = vs.translate_room_type(room_type)
        print(f"  • {room_type} ({english}): {count}")
    
    # Test room type search
    print(f"\n=== Testing Room Type Search ===")
    test_rooms = ['mh', 'oh', 'kh', 'keittiö']
    
    for room in test_rooms:
        results = vs.search_finnish_rooms(room, top_k=3)
        english = vs.translate_room_type(room)
        print(f"\n{room} ({english}): {len(results)} results")
        for i, result in enumerate(results[:2], 1):
            print(f"  {i}. Plan {result['plan_id']}: {result['original_text']}")
    
    # Test similarity search
    print(f"\n=== Testing Similarity Search ===")
    test_queries = [
        "bedroom with natural light",
        "kitchen near living room",
        "bathroom"
    ]
    
    for query in test_queries:
        query_emb = vs.embedding_model.encode(query)
        results = vs.search_similar_finnish_spaces(query_emb, top_k=3)
        print(f"\nQuery: '{query}'")
        print(f"Results: {len(results)}")
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['room_type']} (similarity: {result['similarity']:.3f})")
            print(f"     Plan: {result['plan_id']}, Text: {result['original_text']}")
    
    print("\n✓ All tests completed!")

if __name__ == "__main__":
    test_finnish_embeddings()