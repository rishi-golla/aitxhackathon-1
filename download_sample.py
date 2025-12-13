import os
from huggingface_hub import hf_hub_download
import shutil

def download_sample_video():
    print("Downloading sample video from builddotai/Egocentric-10K...")
    print("Note: You must be logged in to Hugging Face (huggingface-cli login) and have accepted the dataset terms.")
    
    try:
        # Download a specific video file from the dataset
        # We'll try to grab a small sample if possible, or a specific file we know exists from the docs
        # The docs mention: factory_001/workers/worker_001/factory001_worker001_part00.tar
        # But downloading a whole tar might be big. Let's try to find a direct MP4 if accessible or just guide the user.
        
        # Since it's WebDataset (TAR files), we have to download a TAR.
        # Let's download one shard.
        local_path = hf_hub_download(
            repo_id="builddotai/Egocentric-10K",
            filename="factory_001/workers/worker_001/factory001_worker001_part00.tar",
            repo_type="dataset"
        )
        
        print(f"Downloaded shard to: {local_path}")
        
        # Extract one MP4 from the tar
        import tarfile
        with tarfile.open(local_path, "r") as tar:
            # Find first MP4
            mp4_member = next((m for m in tar.getmembers() if m.name.endswith(".mp4")), None)
            if mp4_member:
                print(f"Extracting {mp4_member.name}...")
                tar.extract(mp4_member, path=".")
                # Rename to a simple name
                extracted_path = mp4_member.name
                if os.path.exists("test_footage.mp4"):
                    os.remove("test_footage.mp4")
                os.rename(extracted_path, "test_footage.mp4")
                print("Success! Sample video saved as 'test_footage.mp4'")
            else:
                print("No MP4 found in the shard.")
                
    except Exception as e:
        print(f"Error downloading: {e}")
        print("Please ensure you have access to the dataset at https://huggingface.co/datasets/builddotai/Egocentric-10K")

if __name__ == "__main__":
    download_sample_video()
