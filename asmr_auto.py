import os, json, shutil, subprocess, time
from google import genai
from gradio_client import Client
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

GEMINI_MODEL = "gemini-3.8-flash"
HF_SPACE = "Lightricks/ltx-video-distilled"
HF_API_NAME = "/generate"
N_SCENES = 6
OUT = "work"
SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
HISTORY = "history.json"


def get_idea():
    used = json.load(open(HISTORY)) if os.path.exists(HISTORY) else []
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    ask = (
        "Give me a new, creative idea for a 30-second AI ASMR YouTube Short "
        "(vertical 9:16). The idea should be set in an unusual, unexpected place. "
        f"Avoid these previous ideas: {used[-30:]}. "
        "Return ONLY JSON, no explanation and no markdown, in this format: "
        '{"idea":"short description","title":"catchy English title",'
        '"description":"short description","tags":["asmr","ai"],'
        f'"scenes":["exactly {N_SCENES} English video prompts, each a 5-second scene '
        'with the same visual style and elements, ASMR close-up, satisfying, macro"]}'
    )
    r = None
    for attempt in range(6):
        try:
            r = client.models.generate_content(model=GEMINI_MODEL, contents=ask)
            break
        except Exception as e:
            print(f"gemini attempt {attempt + 1} failed: {e}")
            time.sleep(30)
    if r is None:
        raise RuntimeError("Gemini unavailable after retries")
    txt = r.text.replace("```json", "").replace("```", "").strip()
    data = json.loads(txt)
    used.append(data["idea"])
    json.dump(used, open(HISTORY, "w"), ensure_ascii=False)
    return data


def gen_clips(scenes):
    os.makedirs(OUT, exist_ok=True)
    client = Client(HF_SPACE, token=os.environ.get("HF_TOKEN"))
    paths = []
    for i, p in enumerate(scenes):
        for attempt in range(3):
            try:
                res = client.predict(prompt=p, api_name=HF_API_NAME)
                src = res[0] if isinstance(res, (list, tuple)) else res
                if isinstance(src, dict):
                    src = src.get("video") or src.get("path")
                dst = f"{OUT}/clip{i}.mp4"
                shutil.copy(src, dst)
                paths.append(dst)
                break
            except Exception as e:
                print(f"scene {i} attempt {attempt + 1} failed: {e}")
                time.sleep(20)
    if len(paths) < 2:
        raise RuntimeError("Too few clips generated")
    return paths


def merge(paths):
    lst = f"{OUT}/list.txt"
    with open(lst, "w") as f:
        for p in paths:
            f.write(f"file '{os.path.abspath(p)}'\n")
    out = f"{OUT}/final.mp4"
    vf = "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,fps=30"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst,
         "-vf", vf, "-t", "59", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-an", out],
        check=True,
    )
    return out


def yt_service():
    creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds.valid:
        creds.refresh(Request())
    return build("youtube", "v3", credentials=creds)


def upload(video, d):
    yt = yt_service()
    body = {
        "snippet": {
            "title": (d["title"] + " #Shorts")[:100],
            "description": d["description"] + "\n#Shorts #ASMR #AI",
            "tags": d.get("tags", []) + ["Shorts"],
            "categoryId": "24",
        },
        "status": {"privacyStatus": "public", "selfDeclaredMadeForKids": False},
    }
    req = yt.videos().insert(
        part="snippet,status",
        body=body,
        media_body=MediaFileUpload(video, resumable=True),
    )
    resp = None
    while resp is None:
        _, resp = req.next_chunk()
    print("Uploaded: https://youtube.com/shorts/" + resp["id"])


if __name__ == "__main__":
    data = get_idea()
    print("Idea:", data["idea"])
    clips = gen_clips(data["scenes"])
    final = merge(clips)
    upload(final, data)
