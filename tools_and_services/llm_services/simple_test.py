#!/usr/bin/env python3
"""
Test đơn giản cho LLMService - chỉ test một service cụ thể
"""

import time
import os
import sys
import asyncio
from loguru import logger

# Add current directory to path for importing
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import with absolute path to avoid relative import issues
from llm_services import LLMService

def simple_test(service_name='answer', prompt_args=None):
    """Test đơn giản cho một service"""
    
    # Default test data
    if prompt_args is None:
        test_data = {
            'answer': ("Thuốc chống viêm và giảm đau dịch sang tiếng Anh là gì?", "", [{}]),
            'general': ("Thuốc paracetamol có tác dụng gì và cách sử dụng ra sao?", {}),
            'structured_query_generator': ("Tôi muốn tìm thuốc điều trị cảm cúm, dùng paracetamol có được không?",),
            'reflection': ("Câu hỏi: Codein có tác dụng gì? Liều dùng như nào", "Codein là thuốc giảm đau. Codein có thể gây nghiện."),
        }
        prompt_args = test_data.get(service_name, test_data['answer'])
    
    logger.info(f"Testing {service_name} service...")
    
    try:
        llm_service = LLMService()

        # Actual test runs
        num_runs = 1
        total_time = 0
        total_tokens = 0
        
        for i in range(num_runs):
            logger.info(f"Run {i+1}/{num_runs}")
            
            start_time = time.time()
            response = llm_service.generate_response(*prompt_args)
            end_time = time.time()
            
            response_time = end_time - start_time
            # Ước tính tokens (1 token ≈ 4 ký tự tiếng Việt)
            # response_tokens = len(response) // 4
            
            total_time += response_time
            # total_tokens += response_tokens
            
            logger.info(f"Response time: {response_time:.2f}s")
            # logger.info(f"Response tokens: {response_tokens}")
            # logger.info(f"Tokens/second: {response_tokens/response_time:.2f}")
            logger.info(f"Response preview: {response}...")
            print("-" * 50)
        
        # Calculate averages
        avg_time = total_time / num_runs
        avg_tokens = total_tokens / num_runs
        avg_tokens_per_sec = avg_tokens / avg_time
        
        print("\n" + "="*60)
        print("SUMMARY RESULTS")
        print("="*60)
        print(f"Service: {service_name}")
        print(f"Number of runs: {num_runs}")
        print(f"Average response time: {avg_time:.3f}s")
        print(f"Average response tokens: {avg_tokens:.1f}")
        print(f"Average tokens/second: {avg_tokens_per_sec:.2f}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        raise

async def simple_test_streaming(service_name='general', prompt_args=None):
    """Test streaming response functionality"""
    
    # Default test data
    if prompt_args is None:
        test_data = {
            'answer': ("Thuốc chống viêm có tác dụng như thế nào?", "", [{}]),
            'general': ("Thuốc paracetamol có tác dụng gì và cách sử dụng ra sao?", {}),
            'structured_query_generator': ("Tôi muốn tìm thuốc điều trị đau đầu",),
        }
        prompt_args = test_data.get(service_name, test_data['general'])
    
    try:
        print(f"Testing streaming for service: {service_name}")
        print(f"Prompt args: {prompt_args}")
        print("-" * 60)
        
        # Initialize service
        service = LLMService()
        
        # Test streaming response
        print("Streaming Response:")
        print("-" * 30)
        
        start_time = time.time()
        token_count = 0
        
        async for chunk in service.generate_stream_response(service_name, *prompt_args):
            if chunk:
                print(chunk, end='', flush=True)
                token_count += len(chunk.split())
                
        end_time = time.time()
        
        print(f"\n\n{'-' * 30}")
        print(f"Streaming completed in {end_time - start_time:.3f}s")
        print(f"Approximate tokens: {token_count}")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"Error during streaming test: {e}")
        raise

if __name__ == "__main__":
    # Test với service 'general'
    # print("="*80)
    # print("TESTING NORMAL RESPONSE")
    # print("="*80)
    # simple_test('general')

    # Test streaming response
    print("\n" + "="*80)
    print("TESTING STREAMING RESPONSE")
    print("="*80)
    asyncio.run(simple_test_streaming('general'))

    # print("\n" + "="*80)
    # print("TESTING ANSWER SERVICE")
    # print("="*80)
    # simple_test('answer')

    # print("\n" + "="*80)
    # print("TESTING STRUCTURED QUERY GENERATOR SERVICE")
    # print("="*80)
    # simple_test('structured_query_generator')

    # print("\n" + "="*80)
    # print("TESTING REFLECTION SERVICE")
    # print("="*80)
    # simple_test('reflection')
