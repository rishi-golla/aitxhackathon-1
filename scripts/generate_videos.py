import os
import re
import time
import argparse
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

def parse_prompts(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    prompts = []
    # Split by sections (### Number. Title)
    sections = re.split(r'^### \d+\. ', content, flags=re.MULTILINE)
    
    # Skip the first part (header)
    for section in sections[1:]:
        # Extract title (first line)
        title_match = re.match(r'(.*)', section)
        if not title_match:
            continue
        title = title_match.group(1).strip()
        
        # Extract prompt
        # Look for **Prompt:** followed by > "..."
        # Use non-greedy match for the content inside quotes
        prompt_match = re.search(r'\*\*Prompt:\*\*\s*\n>\s*"(.*?)"', section, re.DOTALL)
        if prompt_match:
            prompt_text = prompt_match.group(1).replace('\n', ' ').strip()
            prompts.append({
                'title': title,
                'prompt': prompt_text
            })
            
    return prompts

def generate_video(client, prompt, output_filename):
    max_retries = 5
    retry_delay = 30
    
    for attempt in range(max_retries):
        try:
            print(f"Generating video for: {output_filename} (Attempt {attempt+1}/{max_retries})")
            print(f"Prompt: {prompt}")
            
            operation = client.models.generate_videos(
                model="veo-3.1-generate-preview",
                prompt=prompt,
                config=types.GenerateVideosConfig(
                    aspect_ratio="16:9"
                )
            )
            
            print("Waiting for video generation to complete...")
            while not operation.done:
                time.sleep(10)
                operation = client.operations.get(operation)
                print(".", end="", flush=True)
            print()
                
            if operation.result:
                 # Check for errors if any (though result usually implies success or failure object)
                 pass

            # Download the generated video.
            if operation.response and operation.response.generated_videos:
                generated_video = operation.response.generated_videos[0]
                
                print(f"Downloading video to {output_filename}...")
                client.files.download(file=generated_video.video)
                generated_video.video.save(output_filename)
                print(f"Saved to {output_filename}")
                return True
            else:
                print("No video generated in response.")
                return False

        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                print(f"Rate limit hit. Waiting {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2 # Exponential backoff
            else:
                print(f"Error generating video: {e}")
                return False
    
    print("Max retries exceeded.")
    return False

def main():
    parser = argparse.ArgumentParser(description='Generate videos from prompts using Gemini Veo 3.1')
    parser.add_argument('--api-key', help='Gemini API Key')
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get('GEMINI_API_KEY')
    if not api_key:
        print("Error: API key not found. Please provide --api-key or set GEMINI_API_KEY env var.")
        return

    client = genai.Client(api_key=api_key)

    prompts_file = 'docs/VIDEO_PROMPTS.md'
    output_dir = 'data'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    prompts = parse_prompts(prompts_file)
    print(f"Found {len(prompts)} prompts.")

    for i, item in enumerate(prompts):
        title_slug = item['title'].lower().replace(' ', '_').replace('+', '').replace('(', '').replace(')', '').replace('__', '_')
        filename = f"violation_{i+1}_{title_slug}.mp4"
        output_path = os.path.join(output_dir, filename)
        
        if os.path.exists(output_path):
            print(f"Skipping {filename}, already exists.")
            continue
            
        success = generate_video(client, item['prompt'], output_path)
        if success:
            print(f"Successfully generated {filename}")
        else:
            print(f"Failed to generate {filename}")
        
        # Sleep a bit to avoid rate limits if necessary
        time.sleep(10)

if __name__ == "__main__":
    main()
