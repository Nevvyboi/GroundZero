"""
GroundZero API Routes v2.1
==========================
All API endpoints that connect to loaded components.
Includes voice transcription with file upload support.
"""

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from pathlib import Path
import asyncio
import tempfile
import os

router = APIRouter()


# ============================================================
# GET COMPONENTS - Access from server.py
# ============================================================

def get_components():
    """Get components from server module"""
    try:
        from . import server
        return getattr(server, '_components', {})
    except:
        return {}


def get_settings():
    """Get settings"""
    try:
        from config import Settings
        return Settings()
    except:
        try:
            from config.settings import Settings
            return Settings()
        except:
            return None


def get_transcriber():
    """Get voice transcriber"""
    try:
        from . import server
        return getattr(server, '_transcriber', None)
    except:
        return None


# ============================================================
# REQUEST/RESPONSE MODELS
# ============================================================

class ChatMessage(BaseModel):
    message: str
    use_knowledge: bool = True
    use_neural: bool = True
    stream: bool = False


class ChatResponse(BaseModel):
    response: str
    sources: List[str] = []
    confidence: float = 0.0
    tokens_used: int = 0


class LearnRequest(BaseModel):
    url: Optional[str] = None
    text: Optional[str] = None
    title: Optional[str] = None


class TeachRequest(BaseModel):
    knowledge: str
    source: str = "user"


# ============================================================
# STATS ENDPOINTS - These power the frontend dashboard
# ============================================================

@router.get("/api/stats")
async def get_stats():
    """Get system statistics - powers the frontend Model Stats panel"""
    components = get_components()
    
    # Vector store stats
    vector_count = 0
    if components.get('vector_store'):
        try:
            vs = components['vector_store']
            if hasattr(vs, 'get_stats'):
                vs_stats = vs.get_stats()
                vector_count = vs_stats.get('total_vectors', vs_stats.get('faiss_vectors', 0))
            elif hasattr(vs, 'count'):
                vector_count = vs.count
            elif hasattr(vs, 'ntotal'):
                vector_count = vs.ntotal
        except:
            pass
    
    if vector_count == 0 and components.get('faiss_index'):
        try:
            vector_count = components['faiss_index'].ntotal
        except:
            pass
    
    # Knowledge graph stats
    facts_count = 0
    entities_count = 0
    if components.get('graph_reasoner'):
        try:
            gr = components['graph_reasoner']
            if hasattr(gr, 'get_stats'):
                gr_stats = gr.get_stats()
                facts_count = gr_stats.get('total_facts', 0)
                entities_count = gr_stats.get('unique_subjects', gr_stats.get('total_entities', 0))
        except:
            pass
    
    # Learning engine stats
    articles_learned = 0
    tokens_trained = 0
    sources_count = 0
    if components.get('learner'):
        try:
            le = components['learner']
            if hasattr(le, 'get_stats'):
                le_stats = le.get_stats()
                articles_learned = le_stats.get('total_articles_learned', le_stats.get('total_articles', 0))
                tokens_trained = le_stats.get('total_tokens_trained', le_stats.get('total_tokens', 0))
                sources_count = le_stats.get('total_sources', articles_learned)
        except:
            pass
    
    # Neural network stats
    vocab_size = 0
    words_learned = 0
    params = 0
    model_size = 'unknown'
    if components.get('neural_brain'):
        try:
            nb = components['neural_brain']
            if hasattr(nb, 'get_stats'):
                nb_stats = nb.get_stats()
                vocab_size = nb_stats.get('vocab_size', 0)
                words_learned = nb_stats.get('total_tokens_trained', tokens_trained)
                params = nb_stats.get('model_params', 0)
                model_size = nb_stats.get('model_size', 'medium')
        except:
            pass
    
    return {
        "vocabulary": vocab_size,
        "words": words_learned,
        "knowledge": facts_count,
        "sources": sources_count,
        "sessions": 0,
        "vectors": vector_count,
        "articles_learned": articles_learned,
        "tokens_trained": tokens_trained,
        "entities": entities_count,
        "facts": facts_count,
        "params": params,
        "neural": {
            "parameters": params,
            "vocab_size": vocab_size,
            "total_tokens_trained": tokens_trained,
            "model_size": model_size
        },
        "learning": {
            "total_articles_learned": articles_learned,
            "total_tokens": tokens_trained
        }
    }


@router.get("/api/stats/detailed")
async def get_stats_detailed():
    """Get detailed system statistics"""
    return await get_stats()


@router.get("/api/status")
async def get_status():
    """Get component status"""
    components = get_components()
    
    return {
        "status": "ready",
        "components": {
            "vector_store": components.get('vector_store') is not None or components.get('faiss_index') is not None,
            "knowledge_base": components.get('kb') is not None,
            "knowledge_graph": components.get('graph_reasoner') is not None,
            "neural_brain": components.get('neural_brain') is not None,
            "learning_engine": components.get('learner') is not None,
            "response_generator": components.get('response_generator') is not None,
            "context_brain": components.get('context_brain') is not None,
            "strategic_planner": components.get('strategic_planner') is not None
        }
    }


@router.get("/api/neural/stats")
async def get_neural_stats():
    """Get neural network statistics"""
    components = get_components()
    neural_brain = components.get('neural_brain')
    
    if not neural_brain:
        return {"status": "not_loaded", "params": 0, "vocab_size": 0}
    
    try:
        if hasattr(neural_brain, 'get_stats'):
            stats = neural_brain.get_stats()
            return {
                "status": "ready",
                "params": stats.get('model_params', 0),
                "vocab_size": stats.get('vocab_size', 0),
                "tokens_trained": stats.get('total_tokens_trained', 0),
                "model_size": stats.get('model_size', 'unknown'),
                "device": stats.get('device', 'cpu')
            }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@router.get("/api/knowledge/stats")  
async def get_knowledge_stats():
    """Get knowledge graph statistics"""
    components = get_components()
    graph = components.get('graph_reasoner')
    
    if not graph:
        return {"status": "not_loaded", "facts": 0, "entities": 0}
    
    try:
        if hasattr(graph, 'get_stats'):
            stats = graph.get_stats()
            return {
                "status": "ready",
                "facts": stats.get('total_facts', 0),
                "entities": stats.get('unique_subjects', 0),
                "relations": len(stats.get('facts_by_relation', {}))
            }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@router.get("/api/learning/stats")
async def get_learning_stats():
    """Get learning engine statistics"""
    components = get_components()
    learner = components.get('learner')
    
    if not learner:
        return {"status": "not_loaded", "articles": 0, "tokens": 0, "is_running": False}
    
    try:
        if hasattr(learner, 'get_stats'):
            stats = learner.get_stats()
            return {
                "status": "ready",
                "articles_learned": stats.get('total_articles_learned', 0),
                "tokens_trained": stats.get('total_tokens_trained', 0),
                "is_running": stats.get('is_running', getattr(learner, 'is_running', False)),
                "current_article": stats.get('current_article', None)
            }
        return {
            "status": "ready",
            "is_running": getattr(learner, 'is_running', False)
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "is_running": False}


# ============================================================
# CHAT ENDPOINT
# ============================================================

@router.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatMessage):
    """Chat with the AI"""
    components = get_components()
    response_gen = components.get('response_generator')
    neural_brain = components.get('neural_brain')
    kb = components.get('kb')
    graph = components.get('graph_reasoner')
    
    query = request.message
    
    def is_good_response(text):
        """Check if response is usable (not garbage)"""
        if not text or len(text) < 5:
            return False
        # Check for too many unknown tokens
        if text.count('<|unk|>') > 3 or text.count('<unk>') > 3:
            return False
        # Check for repetitive garbage
        if text.count('|>') > 10:
            return False
        return True
    
    # Try response generator first
    if response_gen and hasattr(response_gen, 'generate'):
        try:
            result = await asyncio.to_thread(response_gen.generate, query)
            
            if isinstance(result, dict):
                text = result.get('response', result.get('text', ''))
            else:
                text = str(result) if result else ''
            
            if is_good_response(text):
                return ChatResponse(
                    response=text,
                    sources=result.get('sources', []) if isinstance(result, dict) else [],
                    confidence=0.8
                )
        except Exception as e:
            print(f"ResponseGenerator error: {e}")
    
    # Fallback: Try neural brain directly
    if neural_brain and hasattr(neural_brain, 'generate'):
        try:
            result = await asyncio.to_thread(neural_brain.generate, query, max_tokens=100)
            text = str(result) if result else ''
            
            if is_good_response(text):
                return ChatResponse(response=text, confidence=0.6)
        except Exception as e:
            print(f"NeuralBrain error: {e}")
    
    # Fallback: Search knowledge graph
    if graph and hasattr(graph, 'query'):
        try:
            results = await asyncio.to_thread(graph.query, query)
            if results and len(results) > 0:
                facts = []
                for fact in results[:5]:
                    if isinstance(fact, tuple) and len(fact) >= 3:
                        facts.append(f"• {fact[0]} {fact[1]} {fact[2]}")
                    elif isinstance(fact, dict):
                        facts.append(f"• {fact.get('subject', '')} {fact.get('relation', '')} {fact.get('object', '')}")
                    else:
                        facts.append(f"• {str(fact)}")
                
                if facts:
                    response = f"Here's what I know about that:\n\n" + "\n".join(facts)
                    return ChatResponse(response=response, confidence=0.5)
        except Exception as e:
            print(f"GraphReasoner error: {e}")
    
    # Fallback: Search knowledge base
    if kb and hasattr(kb, 'search'):
        try:
            # Try different search signatures
            results = None
            for search_call in [
                lambda: kb.search(query, k=5),
                lambda: kb.search(query, limit=5),
                lambda: kb.search(query, n=5),
                lambda: kb.search(query),
            ]:
                try:
                    results = await asyncio.to_thread(search_call)
                    break
                except TypeError:
                    continue
            
            if results and len(results) > 0:
                response_text = f"Based on my knowledge:\n\n"
                for i, r in enumerate(results[:3], 1):
                    if isinstance(r, dict):
                        content = r.get('content', r.get('text', str(r)))[:150]
                    else:
                        content = str(r)[:150]
                    response_text += f"{i}. {content}...\n\n"
                return ChatResponse(response=response_text, confidence=0.4)
        except Exception as e:
            print(f"KnowledgeBase error: {e}")
    
    # Final fallback - simple responses
    query_lower = query.lower()
    if any(w in query_lower for w in ['hello', 'hi', 'hey']):
        return ChatResponse(response="Hello! I'm GroundZero AI. How can I help you today?", confidence=0.9)
    elif any(w in query_lower for w in ['how are you', 'how do you do']):
        return ChatResponse(response="I'm doing well, thank you for asking! I'm here to help answer your questions.", confidence=0.9)
    elif any(w in query_lower for w in ['what can you do', 'help', 'capabilities']):
        return ChatResponse(response="I can answer questions based on my learned knowledge, help with general queries, and learn from new information. Try asking me about topics I've learned!", confidence=0.9)
    elif any(w in query_lower for w in ['thank', 'thanks']):
        return ChatResponse(response="You're welcome! Let me know if you have any other questions.", confidence=0.9)
    else:
        return ChatResponse(
            response=f"I'm still learning about '{query}'. My neural network is training - the more I learn, the better my responses will be! You can teach me by going to the Learning tab.",
            confidence=0.2
        )


@router.post("/api/ask")
async def ask(request: ChatMessage):
    """Alternative chat endpoint"""
    return await chat(request)


# ============================================================
# LEARNING ENDPOINTS
# ============================================================

@router.post("/api/learn")
async def learn_from_url(request: LearnRequest):
    """Learn from a URL"""
    components = get_components()
    learner = components.get('learner')
    
    if not learner:
        raise HTTPException(status_code=503, detail="Learning engine not available")
    
    if not request.url:
        raise HTTPException(status_code=400, detail="URL required")
    
    try:
        if hasattr(learner, 'learn_from_url'):
            result = await asyncio.to_thread(learner.learn_from_url, request.url)
            return {"status": "success", "result": result}
        else:
            raise HTTPException(status_code=501, detail="learn_from_url not implemented")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/teach")
async def teach(request: TeachRequest):
    """Teach the AI new knowledge"""
    components = get_components()
    learner = components.get('learner')
    graph = components.get('graph_reasoner')
    
    if not request.knowledge:
        raise HTTPException(status_code=400, detail="Knowledge text required")
    
    result = {"status": "success", "added": []}
    
    if graph and hasattr(graph, 'add_fact'):
        try:
            graph.add_fact("user_knowledge", "contains", request.knowledge[:200])
            result["added"].append("knowledge_graph")
        except:
            pass
    
    if learner and hasattr(learner, 'learn_text'):
        try:
            learner.learn_text(request.knowledge, source=request.source)
            result["added"].append("learning_engine")
        except:
            pass
    
    return result


@router.post("/api/learning/start")
async def start_learning():
    """Start the learning engine"""
    components = get_components()
    learner = components.get('learner')
    
    if not learner:
        raise HTTPException(status_code=503, detail="Learning engine not available")
    
    try:
        if hasattr(learner, 'start'):
            await asyncio.to_thread(learner.start)
            return {"status": "started"}
        elif hasattr(learner, 'start_learning'):
            await asyncio.to_thread(learner.start_learning)
            return {"status": "started"}
        else:
            return {"status": "no_start_method"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/learning/stop")
async def stop_learning():
    """Stop the learning engine"""
    components = get_components()
    learner = components.get('learner')
    
    if not learner:
        raise HTTPException(status_code=503, detail="Learning engine not available")
    
    try:
        if hasattr(learner, 'stop'):
            learner.stop()
            return {"status": "stopped"}
        elif hasattr(learner, 'stop_learning'):
            learner.stop_learning()
            return {"status": "stopped"}
        else:
            return {"status": "no_stop_method"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# SEARCH ENDPOINTS
# ============================================================

@router.get("/api/search")
async def search(q: str = Query(..., description="Search query"), k: int = 10):
    """Search the knowledge base"""
    components = get_components()
    results = []
    
    # Search knowledge base first (has actual content)
    kb = components.get('kb')
    if kb:
        try:
            kb_results = None
            # Try different search methods
            for method in ['search', 'query', 'find']:
                if hasattr(kb, method):
                    func = getattr(kb, method)
                    # Try different argument styles
                    for call in [
                        lambda: func(q, k=k),
                        lambda: func(q, limit=k),
                        lambda: func(q, n=k),
                        lambda: func(q),
                        lambda: func(query=q, k=k),
                    ]:
                        try:
                            kb_results = call()
                            break
                        except TypeError:
                            continue
                    if kb_results:
                        break
            
            if kb_results:
                for item in kb_results[:k]:
                    content = ""
                    if isinstance(item, dict):
                        content = item.get('content') or item.get('text') or item.get('chunk') or str(item)
                    elif isinstance(item, tuple):
                        content = str(item[0]) if item else str(item)
                    elif hasattr(item, 'content'):
                        content = item.content
                    elif hasattr(item, 'text'):
                        content = item.text
                    else:
                        content = str(item)
                    
                    if content and len(content) > 10:
                        results.append({
                            "type": "knowledge",
                            "content": content[:500],
                            "source": "knowledge_base"
                        })
        except Exception as e:
            print(f"KB search error: {e}")
    
    # Search knowledge graph for facts
    graph = components.get('graph_reasoner')
    if graph and hasattr(graph, 'query'):
        try:
            graph_results = graph.query(q)
            for fact in (graph_results or [])[:k]:
                if isinstance(fact, tuple) and len(fact) >= 3:
                    content = f"{fact[0]} → {fact[1]} → {fact[2]}"
                elif isinstance(fact, dict):
                    content = f"{fact.get('subject', '')} → {fact.get('relation', '')} → {fact.get('object', '')}"
                else:
                    content = str(fact)
                
                if content and len(content) > 5:
                    results.append({
                        "type": "fact",
                        "content": content,
                        "source": "knowledge_graph"
                    })
        except Exception as e:
            print(f"Graph search error: {e}")
    
    # Search vector store with content retrieval
    vector_store = components.get('vector_store')
    if vector_store and hasattr(vector_store, 'search'):
        try:
            # Get embedding function
            embed_func = None
            try:
                from storage.vector_store import simple_embed
                embed_func = lambda t: simple_embed(t, dim=256)
            except:
                try:
                    from core.embeddings import get_embedding
                    embed_func = get_embedding
                except:
                    pass
            
            if embed_func:
                query_vec = embed_func(q)
                vec_results = vector_store.search(query_vec, k=k)
                
                for r in vec_results:
                    content = ""
                    # Try to get actual content from metadata
                    if hasattr(r, 'metadata') and r.metadata:
                        meta = r.metadata
                        if isinstance(meta, dict):
                            content = meta.get('content') or meta.get('text') or meta.get('chunk') or meta.get('content_preview', '')
                        elif hasattr(meta, 'content'):
                            content = meta.content
                        elif hasattr(meta, 'text'):
                            content = meta.text
                    
                    # If no content in metadata, try to retrieve from KB by ID
                    if not content and kb and hasattr(kb, 'get'):
                        try:
                            item = kb.get(r.id)
                            if item:
                                if isinstance(item, dict):
                                    content = item.get('content') or item.get('text') or ''
                                else:
                                    content = str(item)
                        except:
                            pass
                    
                    if content and len(content) > 10:
                        results.append({
                            "type": "vector",
                            "content": content[:500],
                            "score": round(1 - (r.distance if hasattr(r, 'distance') else 0), 3),
                            "source": "vector_store"
                        })
                    elif r.id:
                        # Just show ID if we can't get content
                        results.append({
                            "type": "vector_id",
                            "content": f"Vector ID: {r.id}",
                            "source": "vector_store"
                        })
        except Exception as e:
            print(f"Vector search error: {e}")
    
    # Deduplicate results by content
    seen = set()
    unique_results = []
    for r in results:
        content_key = r.get('content', '')[:100]
        if content_key not in seen:
            seen.add(content_key)
            unique_results.append(r)
    
    return {"query": q, "results": unique_results[:k], "total": len(unique_results)}


@router.post("/api/search")
async def search_post(query: str, k: int = 10):
    """Search via POST"""
    return await search(q=query, k=k)


# ============================================================
# TIMELINE ENDPOINT
# ============================================================

@router.get("/api/timeline")
async def get_timeline(limit: int = 50):
    """Get model timeline events"""
    components = get_components()
    neural_brain = components.get('neural_brain')
    
    events = []
    
    if neural_brain and hasattr(neural_brain, 'get_timeline'):
        try:
            timeline = neural_brain.get_timeline()
            events = timeline[:limit] if timeline else []
        except:
            pass
    
    return {"events": events, "total": len(events)}


# ============================================================
# VOICE ENDPOINTS  
# ============================================================

@router.post("/api/voice/transcribe")
async def transcribe_voice(audio: UploadFile = File(...)):
    """Transcribe uploaded audio file"""
    transcriber = get_transcriber()
    
    if not transcriber:
        return {"error": "Voice transcription not available. Install whisper: pip install openai-whisper", "text": ""}
    
    try:
        # Save uploaded file temporarily
        content = await audio.read()
        
        if len(content) < 1000:
            return {"error": "Audio too short", "text": ""}
        
        # Determine file extension from content type
        ext = '.webm'
        if audio.content_type:
            if 'ogg' in audio.content_type:
                ext = '.ogg'
            elif 'wav' in audio.content_type:
                ext = '.wav'
            elif 'mp3' in audio.content_type:
                ext = '.mp3'
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            # Transcribe using whisper
            if hasattr(transcriber, 'transcribe'):
                result = await asyncio.to_thread(transcriber.transcribe, tmp_path)
                
                if isinstance(result, dict):
                    text = result.get('text', '')
                    # Clean up the text
                    text = text.strip()
                    return {
                        "text": text,
                        "language": result.get('language', 'en'),
                        "success": True
                    }
                else:
                    return {"text": str(result).strip(), "success": True}
            else:
                return {"error": "Transcriber has no transcribe method", "text": ""}
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except:
                pass
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e), "text": "", "success": False}


@router.websocket("/ws/voice")
async def voice_websocket(websocket: WebSocket):
    """WebSocket for streaming voice transcription"""
    await websocket.accept()
    
    transcriber = get_transcriber()
    
    if not transcriber:
        await websocket.send_json({"error": "Voice transcription not available"})
        await websocket.close()
        return
    
    await websocket.send_json({"status": "ready", "message": "Send audio data"})
    
    audio_buffer = bytearray()
    
    try:
        while True:
            data = await websocket.receive_bytes()
            audio_buffer.extend(data)
            
            # Process when we have enough data (about 2 seconds)
            if len(audio_buffer) > 32000:
                try:
                    # Save to temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmp:
                        tmp.write(bytes(audio_buffer))
                        tmp_path = tmp.name
                    
                    try:
                        if hasattr(transcriber, 'transcribe'):
                            result = await asyncio.to_thread(transcriber.transcribe, tmp_path)
                            text = result.get('text', '') if isinstance(result, dict) else str(result)
                            if text.strip():
                                await websocket.send_json({"text": text.strip()})
                    finally:
                        os.unlink(tmp_path)
                    
                    audio_buffer.clear()
                except Exception as e:
                    await websocket.send_json({"error": str(e)})
                    
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass


# ============================================================
# MODEL ENDPOINTS
# ============================================================

@router.get("/api/model/info")
async def get_model_info():
    """Get model information"""
    components = get_components()
    neural_brain = components.get('neural_brain')
    
    if not neural_brain:
        return {"status": "not_loaded"}
    
    try:
        if hasattr(neural_brain, 'get_stats'):
            stats = neural_brain.get_stats()
            return {
                "status": "ready",
                "size": stats.get('model_size', 'unknown'),
                "params": stats.get('model_params', 0),
                "vocab_size": stats.get('vocab_size', 0),
                "device": stats.get('device', 'cpu')
            }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@router.post("/api/model/change-size")
async def change_model_size(new_size: str, backfill: bool = True):
    """Change model size"""
    components = get_components()
    neural_brain = components.get('neural_brain')
    
    if not neural_brain:
        return {"success": False, "error": "Neural brain not loaded"}
    
    try:
        if hasattr(neural_brain, 'change_size'):
            result = await asyncio.to_thread(neural_brain.change_size, new_size, backfill=backfill)
            return {"success": True, "result": result}
        else:
            return {"success": False, "error": "change_size not implemented"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================
# FRONTEND ROUTES
# ============================================================

@router.get("/")
async def index():
    """Serve frontend"""
    static_path = Path(__file__).parent.parent / "static" / "index.html"
    if static_path.exists():
        return FileResponse(static_path)
    return {"message": "GroundZero API", "docs": "/docs"}


@router.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy"}