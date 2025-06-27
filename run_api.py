#!/usr/bin/env python3
"""
Startup script for Drug Agentic Chatbot API
"""

import uvicorn
import logging
import sys
import os

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import app

def main():
    """Main entry point"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🚀 Starting Drug Agentic Chatbot API...")
    print("📚 Access API documentation at: http://localhost:8000/docs")
    print("🔍 Health check at: http://localhost:8000/health")
    print("💬 Q&A endpoint: http://localhost:8000/qa")
    print("📡 Streaming Q&A: http://localhost:8000/qa/stream")
    print("📊 Indexing endpoint: http://localhost:8000/index")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True,
        reload=True  # Set to False in production
    )

if __name__ == "__main__":
    main()
