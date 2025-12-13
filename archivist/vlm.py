import requests
import os
import json
import base64
import cv2

class VLMClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        # Local VSS Engine typically doesn't require an API key for internal calls, 
        # but we can pass a dummy one if needed.
        self.api_key = "dummy-key" 

    def _encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def upload_asset(self, file_path):
        """Uploads a video/image asset to the VSS Engine."""
        url = f"{self.base_url}/files"
        print(f"Uploading {file_path} to {url}...")
        try:
            with open(file_path, 'rb') as f:
                files = {'file': f}
                response = requests.post(url, files=files, timeout=600)
            
            response.raise_for_status()
            result = response.json()
            print(f"Upload result: {result}")
            return result.get('id')
        except Exception as e:
            print(f"Error uploading asset: {e}")
            return None

    def analyze_video(self, video_path, query="Describe this industrial scene in detail. Focus on safety violations and worker activities."):
        """
        Uploads the video to the local VSS Engine and requests analysis.
        """
        print(f"Analyzing {video_path} locally...")
        
        try:
            # 1. Upload Video
            asset_id = self.upload_asset(video_path)
            if not asset_id:
                return "Error: Could not upload video."
            
            # 2. Construct Payload
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            # The VSS Engine expects 'video_uuid' (mapping to id_list) for video analysis
            payload = {
                "model": "cosmos-reason1",
                "messages": [
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                "video_uuid": [asset_id], # Passing as list to be safe, or single ID if supported
                "max_tokens": 1024,
                "temperature": 0.2
            }
            
            # 3. Inference
            response = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=payload, timeout=300)
            
            response.raise_for_status()
            result = response.json()
            
            return result['choices'][0]['message']['content']
            
        except Exception as e:
            print(f"Error in Local VLM analysis: {e}")
            return None
        payload = {
            "model": "vila", 
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this industrial scene in detail. Focus on safety violations and worker activities."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 300
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except Exception as e:
            print(f"VLM Error: {e}")
            return f"Error analyzing video: {e}"
