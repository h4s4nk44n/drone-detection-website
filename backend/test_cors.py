#!/usr/bin/env python3
"""Simple script to test CORS headers"""

import requests
import sys

def test_cors(base_url):
    """Test CORS headers on different endpoints"""
    
    endpoints = ['/test', '/']
    
    for endpoint in endpoints:
        url = f"{base_url}{endpoint}"
        print(f"\n🔄 Testing CORS for: {url}")
        
        try:
            # Test OPTIONS request (preflight)
            options_response = requests.options(url, timeout=10)
            print(f"   OPTIONS Status: {options_response.status_code}")
            
            cors_headers = [
                'Access-Control-Allow-Origin',
                'Access-Control-Allow-Methods',
                'Access-Control-Allow-Headers'
            ]
            
            for header in cors_headers:
                value = options_response.headers.get(header, 'NOT FOUND')
                print(f"   {header}: {value}")
            
            # Test GET request
            get_response = requests.get(url, timeout=10)
            print(f"   GET Status: {get_response.status_code}")
            print(f"   GET CORS Origin: {get_response.headers.get('Access-Control-Allow-Origin', 'NOT FOUND')}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    # Test locally first
    print("🧪 Testing CORS configuration...")
    
    # You can test against your deployed URL
    base_url = "https://drone-detection-686868741947.europe-west1.run.app"
    test_cors(base_url)
    
    print("\n✅ CORS test complete!") 