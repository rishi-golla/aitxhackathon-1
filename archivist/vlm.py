import requests
import os
import json

class VLMClient:
    """Client for the local VSS Engine (NVIDIA Visual Search & Summary)."""

    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url

    def upload_asset(self, file_path):
        """
        Uploads a video/image asset to the VSS Engine.
        Returns the asset ID for use in subsequent API calls.
        """
        url = f"{self.base_url}/files"
        print(f"Uploading {file_path} to {url}...")

        try:
            with open(file_path, 'rb') as f:
                # VSS Engine requires purpose=vision and media_type for uploads
                files = {'file': (os.path.basename(file_path), f)}
                data = {'purpose': 'vision', 'media_type': 'video'}
                response = requests.post(url, files=files, data=data, timeout=600)

            response.raise_for_status()
            result = response.json()
            asset_id = result.get('id')
            print(f"Upload successful. Asset ID: {asset_id}")
            return asset_id
        except requests.exceptions.RequestException as e:
            print(f"Error uploading asset: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response: {e.response.text}")
            return None

    def analyze_video(self, video_path, query="Describe this industrial scene in detail. Focus on safety hazards, PPE usage, and worker activities."):
        """
        Uploads a video to the VSS Engine and requests analysis using the Cosmos-Reason1 VLM.

        Args:
            video_path: Path to the video file
            query: The prompt/question to ask about the video

        Returns:
            String description of the video content, or None on error
        """
        print(f"Analyzing {video_path}...")

        try:
            # 1. Upload Video to get asset ID
            asset_id = self.upload_asset(video_path)
            if not asset_id:
                return "Error: Could not upload video."

            # 2. Use /summarize endpoint with enable_chat for Q&A capability
            # This is the correct way to analyze uploaded videos in VSS Engine
            headers = {"Content-Type": "application/json"}

            payload = {
                "model": "cosmos-reason1",
                "id": asset_id,
                "prompt": query,
                "caption_summarization_prompt": "Generate a detailed description of the video content focusing on workplace safety.",
                "summary_aggregation_prompt": "Provide a comprehensive safety analysis based on the video observations.",
                "enable_chat": True,
                "max_tokens": 1024,
                "temperature": 0.2
            }

            print(f"Sending summarization request for asset {asset_id}...")
            response = requests.post(
                f"{self.base_url}/summarize",
                headers=headers,
                json=payload,
                timeout=300
            )

            response.raise_for_status()
            result = response.json()

            # Extract the summary from the response
            if 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0].get('message', {}).get('content', '')
            elif 'summary' in result:
                return result['summary']
            else:
                print(f"Unexpected response format: {result}")
                return str(result)

        except requests.exceptions.RequestException as e:
            print(f"Error in VLM analysis: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response: {e.response.text}")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None

    def health_check(self):
        """Check if the VSS Engine is ready."""
        try:
            response = requests.get(f"{self.base_url}/health/ready", timeout=5)
            return response.status_code == 200
        except:
            return False
