"""
Diagnostic: Check Vector Search Quality
========================================
Run from GroundZero directory:
    python check_search.py "France"
    python check_search.py "Albert Einstein"
"""

import sys
sys.path.insert(0, '.')

from storage import KnowledgeBase
from config import Settings

def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "France"
    
    print(f"\n🔍 Searching for: '{query}'\n")
    print("=" * 60)
    
    settings = Settings()
    kb = KnowledgeBase(data_dir=settings.data_dir, dimension=settings.embedding_dimension)
    
    # Search
    results = kb.search(query, limit=10, min_score=0.0)
    
    if not results:
        print("❌ No results found!")
        print("\nPossible issues:")
        print("  - Query not matching any content")
        print("  - Embeddings not working correctly")
        return
    
    print(f"Found {len(results)} results:\n")
    
    for i, r in enumerate(results, 1):
        title = r.get('source_title', 'Unknown')[:50]
        relevance = r.get('relevance', 0)
        content_preview = r.get('content', '')[:100].replace('\n', ' ')
        
        # Check if query term appears in content
        query_lower = query.lower()
        content_lower = r.get('content', '').lower()
        has_query = query_lower in content_lower
        
        match_icon = "✅" if has_query else "⚠️"
        
        print(f"{i}. [{relevance:.1%}] {match_icon} {title}")
        print(f"   Content: {content_preview}...")
        print()
    
    # Check if any result actually contains the query
    matching = sum(1 for r in results if query.lower() in r.get('content', '').lower())
    print("=" * 60)
    print(f"📊 Summary: {matching}/{len(results)} results contain '{query}'")
    
    if matching == 0:
        print(f"\n⚠️ None of the top results contain '{query}'!")
        print("   This suggests the embedding search isn't finding relevant content.")
        print(f"   Try: Learn about {query} first by clicking 'Learn from URL'")
        print(f"   URL: https://en.wikipedia.org/wiki/{query.replace(' ', '_')}")


if __name__ == "__main__":
    main()