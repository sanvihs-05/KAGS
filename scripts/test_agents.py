import sys
from pathlib import Path
backend_path = Path(__file__).parent.parent / 'backend'
sys.path.insert(0, str(backend_path))

# Global vector store instance
_vector_store = None

def get_vector_store():
    """Get or create vector store instance (singleton pattern)"""
    global _vector_store
    if _vector_store is None:
        from database.vector_store import VectorStoreManager
        _vector_store = VectorStoreManager()
    return _vector_store

def test_ollama():
    """Test Ollama connection"""
    print("🔍 Testing Ollama connection...")
    try:
        from ollama import Client
        client = Client()
        
        response = client.list()
        print(f"✅ Ollama is running")
        
        # Try direct generation test
        test_response = client.chat(
            model='llama3.1:8b',
            messages=[{'role': 'user', 'content': 'Say hello in one word'}],
            options={'temperature': 0.0}
        )
        print(f"✅ llama3.1:8b is working!")
        print(f"✅ Generation test: {test_response['message']['content'][:50]}")
        return True
            
    except Exception as e:
        print(f"❌ Ollama test failed: {e}")
        return False

def test_embeddings():
    """Test Finnish embeddings"""
    print("\n🔍 Testing Finnish embeddings...")
    try:
        vs = get_vector_store()  # Use singleton
        
        if vs.finnish_embeddings:
            print(f"✅ Finnish embeddings loaded")
            print(f"   • Annotations: {len(vs.finnish_embeddings.metadata):,}")
            print(f"   • Embedding types: {list(vs.finnish_embeddings.embeddings.keys())}")
            
            # Test encoding
            query = "bedroom"
            emb = vs.finnish_embeddings.encode_query(query)
            print(f"✅ Query encoding works: {emb.shape}")
            
            # Test search
            results = vs.finnish_embeddings.search_similar_by_embedding(emb, top_k=3)
            print(f"✅ Search works: {len(results)} results found")
            for i, r in enumerate(results, 1):
                print(f"   {i}. {r['room_type']} (similarity: {r['similarity']:.3f})")
            return True
        else:
            print(f"❌ Finnish embeddings not loaded")
            return False
    except Exception as e:
        print(f"❌ Embeddings test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_encoder():
    """Test encoder agent"""
    print("\n🔍 Testing Encoder Agent...")
    try:
        from agents.encoder_agent import EncoderAgent
        
        vs = get_vector_store()  # Use singleton
        encoder = EncoderAgent(vs)
        
        # Simple test
        test_input = "2 bedroom apartment with kitchen and bathroom"
        print(f"Encoding: '{test_input}'")
        node = encoder.encode_requirements(test_input)
        
        print(f"✅ Encoder works!")
        print(f"   • Node ID: {node.node_id[:16]}...")
        print(f"   • Functions: {len(node.functions)}")
        
        if node.functions:
            for func_id, func in list(node.functions.items())[:3]:
                print(f"     - {func.name} (priority: {func.priority:.2f})")
        else:
            print(f"     ⚠️  No functions created (check Finnish mapper)")
        
        print(f"   • Behaviors: {len(node.behaviors)}")
        if node.behaviors:
            for behav_id, behav in list(node.behaviors.items())[:2]:
                print(f"     - {behav.metric_name}: {behav.target_value} {behav.metric_unit}")
        
        print(f"   • Structures: {len(node.structures)}")
        
        return True
    except Exception as e:
        print(f"❌ Encoder test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_generalizer():
    """Test generalizer agent"""
    print("\n🔍 Testing Generalizer Agent...")
    try:
        from agents.encoder_agent import EncoderAgent
        from agents.generalizer_agent import GeneralizerAgent
        
        vs = get_vector_store()  # Use singleton
        encoder = EncoderAgent(vs)
        generalizer = GeneralizerAgent()
        
        # Create a problem node
        problem_node = encoder.encode_requirements("2 bedroom apartment with living room")
        
        # Generate alternatives
        alternatives = generalizer.decompose_problem(problem_node, max_alternatives=3)
        
        print(f"✅ Generalizer works!")
        print(f"   • Generated {len(alternatives)} alternatives")
        
        for i, alt in enumerate(alternatives, 1):
            print(f"   {i}. {alt.metadata.get('variant_type', 'N/A')}")
        
        return True
    except Exception as e:
        print(f"❌ Generalizer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("="*60)
    print("🚀 FBSL-KAGS Quick System Check")
    print("="*60)
    
    ollama_ok = test_ollama()
    embeddings_ok = test_embeddings()
    encoder_ok = test_encoder() if embeddings_ok else False
    generalizer_ok = test_generalizer() if encoder_ok else False
    
    print("\n" + "="*60)
    print("📊 Summary:")
    print(f"  Ollama: {'✅' if ollama_ok else '❌'}")
    print(f"  Embeddings: {'✅' if embeddings_ok else '❌'}")
    print(f"  Encoder: {'✅' if encoder_ok else '❌'}")
    print(f"  Generalizer: {'✅' if generalizer_ok else '❌'}")
    print("="*60)
    
    if all([ollama_ok, embeddings_ok, encoder_ok, generalizer_ok]):
        print("\n🎉 ALL SYSTEMS OPERATIONAL!")
        print("\n✅ Ready for production use")
        print("\nNext steps:")
        print("  1. Run full agent tests: python scripts/test_agents.py")
        print("  2. Test complete pipeline: python scripts/test_full_pipeline.py")
        print("  3. Start API server: cd backend && python main.py")
    else:
        print("\n⚠️  Some systems need attention")