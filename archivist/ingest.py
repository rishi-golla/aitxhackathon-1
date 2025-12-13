import os
import json
from .vlm import VLMClient
from .vector_store import VectorStore

def run_ingest(dataset_path):
    print(f"Starting ingestion for dataset at {dataset_path}")
    
    vlm = VLMClient()
    store = VectorStore()
    
    if not os.path.exists(dataset_path):
        print(f"Dataset path {dataset_path} does not exist.")
        return

    files = [f for f in os.listdir(dataset_path) if f.endswith('.mp4')]
    
    for video_file in files:
        full_path = os.path.join(dataset_path, video_file)
        print(f"Processing {video_file}...")
        
        # Analyze the video using VILA Cloud API
        description = vlm.analyze_video(full_path)
        
        if description:
            print(f"Description: {description}")
            # Store the description in our index
            store.add_document(video_file, description)
        else:
            print(f"Failed to analyze {video_file}")
            
    # Save the index
    store.save()
    print("Ingestion complete.")

if __name__ == "__main__":
    run_ingest("data")
