#!/usr/bin/env python3
"""
Test script for Drug Agentic Chatbot API
"""

import requests
import json
import time
import sys
import os

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

API_BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test health check endpoint"""
    print("🔍 Testing health check...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        result = response.json()
        
        if response.status_code == 200:
            print(f"✅ Health check passed: {result['message']}")
            return True
        else:
            print(f"❌ Health check failed: {result}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_indexing():
    """Test indexing endpoint"""
    print("\n📚 Testing indexing...")
    
    try:
        payload = {
            "csv_file_path": "data/knowledge_base.csv"
        }
        
        response = requests.post(f"{API_BASE_URL}/index", json=payload)
        result = response.json()
        
        if response.status_code == 200 and result['success']:
            print(f"✅ Indexing successful: {result['message']}")
            print(f"📊 Documents indexed: {result['document_count']}")
            return True
        else:
            print(f"❌ Indexing failed: {result}")
            return False
    except Exception as e:
        print(f"❌ Indexing error: {e}")
        return False

def test_qa():
    """Test Q&A endpoint"""
    print("\n💬 Testing Q&A...")
    
    try:
        payload = {
            "query": "Tác dụng phụ của paracetamol là gì?",
            "user_id": "test_user_123",
            "conversation_id": "test_conv_456"
        }
        
        response = requests.post(f"{API_BASE_URL}/qa", json=payload)
        result = response.json()
        
        if response.status_code == 200 and result['success']:
            print(f"✅ Q&A successful")
            print(f"📝 Answer: {result['answer'][:100]}...")
            print(f"⏱️ Processing time: {result.get('processing_time', 'N/A')}s")
            return True
        else:
            print(f"❌ Q&A failed: {result}")
            return False
    except Exception as e:
        print(f"❌ Q&A error: {e}")
        return False

def test_streaming_qa():
    """Test streaming Q&A endpoint"""
    print("\n📡 Testing streaming Q&A...")
    
    try:
        payload = {
            "query": "Paracetamol có tương tác với thuốc nào không?",
            "user_id": "test_user_streaming",
            "conversation_id": "test_conv_streaming"
        }
        
        response = requests.post(
            f"{API_BASE_URL}/qa/stream", 
            json=payload,
            stream=True
        )
        
        if response.status_code == 200:
            print("✅ Streaming Q&A started")
            print("📺 Streaming response:")
            
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        data = json.loads(line_str[6:])
                        if data['type'] == 'chunk':
                            print(data['content'], end='', flush=True)
                        elif data['type'] == 'complete':
                            print("\n✅ Streaming completed")
                            break
                        elif data['type'] == 'error':
                            print(f"\n❌ Streaming error: {data['message']}")
                            return False
            return True
        else:
            print(f"❌ Streaming failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Streaming error: {e}")
        return False

def test_status():
    """Test status endpoint"""
    print("\n📊 Testing status...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/status")
        result = response.json()
        
        if response.status_code == 200:
            print(f"✅ Status check passed")
            print(f"🔧 API Status: {result['api_status']}")
            print(f"⏰ Timestamp: {result['timestamp']}")
            return True
        else:
            print(f"❌ Status check failed: {result}")
            return False
    except Exception as e:
        print(f"❌ Status error: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Starting API Tests for Drug Agentic Chatbot")
    print("=" * 50)
    
    # Wait for API to be ready
    print("⏳ Waiting for API to be ready...")
    time.sleep(2)
    
    tests = [
        ("Health Check", test_health_check),
        ("Status Check", test_status),
        ("Indexing", test_indexing),
        ("Q&A", test_qa),
        ("Streaming Q&A", test_streaming_qa)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
        
        time.sleep(1)  # Small delay between tests
    
    # Summary
    print(f"\n{'='*50}")
    print("📋 TEST SUMMARY")
    print(f"{'='*50}")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<30} {status}")
        if result:
            passed += 1
    
    print(f"\n📊 Results: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! API is working correctly.")
        return 0
    else:
        print("⚠️ Some tests failed. Check the logs above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
