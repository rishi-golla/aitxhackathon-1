import requests
import os
import uuid
import sys

invoke_url = "https://ai.api.nvidia.com/v1/vlm/nvidia/vila"
stream = False
query = "Describe the scene"

kApiKey = os.getenv("VILA_API_KEY", "")
assert kApiKey, "Generate API_KEY and export VILA_API_KEY=xxxx"

kNvcfAssetUrl = "https://api.nvcf.nvidia.com/v2/nvcf/assets"
# ext: {mime, media}
kSupportedList = {
    "png": ["image/png", "img"],
    "jpg": ["image/jpg", "img"],
    "jpeg": ["image/jpeg", "img"],
    "mp4": ["video/mp4", "video"],
}

def get_extention(filename):
    _, ext = os.path.splitext(filename)
    ext = ext[1:].lower()
    return ext

def mime_type(ext):
    return kSupportedList[ext][0]
def media_type(ext):
    return kSupportedList[ext][1]

def _upload_asset(media_file, description):
    ext = get_extention(media_file)
    assert ext in kSupportedList
    data_input = open(media_file, "rb")
    headers={
        "Authorization": f"Bearer {kApiKey}",
        "Content-Type": "application/json",
        "accept": "application/json",
    }
    assert_url = kNvcfAssetUrl
    authorize = requests.post(
        assert_url,
        headers = headers,
        json={"contentType": f"{mime_type(ext)}", "description": description},
        timeout=30,
    )
    authorize.raise_for_status()

    authorize_res = authorize.json()
    print(f"uploadUrl: {authorize_res['uploadUrl']}")
    response = requests.put(
        authorize_res["uploadUrl"],
        data=data_input,
        headers={
            "x-amz-meta-nvcf-asset-description": description,
            "content-type": mime_type(ext),
        },
        timeout=300,
    )

    response.raise_for_status()
    if response.status_code == 200:
        print(f"upload asset_id {authorize_res['assetId']} successfully!")
    else:
        print(f"upload asset_id {authorize_res['assetId']} failed.")
    return uuid.UUID(authorize_res["assetId"])

def _delete_asset(asset_id):
    headers = {
        "Authorization": f"Bearer {kApiKey}",
    }
    assert_url = f"{kNvcfAssetUrl}/{asset_id}"
    response = requests.delete(
        assert_url, headers=headers, timeout=30
    )
    response.raise_for_status()

def chat_with_media_nvcf(infer_url, media_files, query: str, stream: bool = False):
    asset_list = []
    ext_list = []
    media_content = ""
    assert isinstance(media_files, list), f"{media_files}"
    print(f"uploading {media_files} into s3")
    has_video = False
    for media_file in media_files:
        ext = get_extention(media_file)
        assert ext in kSupportedList, f"{media_file} format is not supported"
        if media_type(ext) == "video":
            has_video = True
        asset_id = _upload_asset(media_file, "Reference media file")
        asset_list.append(f"{asset_id}")
        ext_list.append(ext)
        media_content += f'<{media_type(ext)} src="data:{mime_type(ext)};asset_id,{asset_id}" />'
    if has_video:
        assert len(media_files) == 1, "Only single video supported."
    asset_seq = ",".join(asset_list)
    print(f"received asset_id list: {asset_seq}")
    headers = {
        "Authorization": f"Bearer {kApiKey}",
        "Content-Type": "application/json",
        "NVCF-INPUT-ASSET-REFERENCES": asset_seq,
        "NVCF-FUNCTION-ASSET-IDS": asset_seq,
        "Accept": "application/json",
    }
    if stream:
        headers["Accept"] = "text/event-stream"
    response = None

    messages = [
        {
            "role": "user",
            "content": f"{query} {media_content}",
        }
    ]
    payload = {
        "max_tokens": 1024,
        "temperature": 0.2,
        "top_p": 0.7,
        "seed": 50,
        "num_frames_per_inference": 8,
        "messages": messages,
        "stream": stream,
        "model": "nvidia/vila",
    }

    response = requests.post(infer_url, headers=headers, json=payload, stream=stream)
    if stream:
        for line in response.iter_lines():
            if line:
                print(line.decode("utf-8"))
    else:
        print(response.json())

    print(f"deleting assets: {asset_list}")
    for asset_id in asset_list:
        _delete_asset(asset_id)

if __name__ == "__main__":
    """ export VILA_API_KEY=xxx.
        python test.py sample.mp4
        python test.py sample1.png sample2.png
    """

    if len(sys.argv) <= 1:
        print("Usage: export VILA_API_KEY=xxx")
        print(f"       python {sys.argv[0]} sample1.png sample2.png ... sample16.png")
        print(f"       python {sys.argv[0]} sample.mp4")
        sys.exit(1)
    media_samples = list(sys.argv[1:])
    chat_with_media_nvcf(invoke_url, media_samples, query, stream)
