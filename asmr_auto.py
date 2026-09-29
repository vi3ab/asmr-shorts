import os, json, shutil, subprocess, time
from google import genai
from gradio_client import Client
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

GEMINI_MODEL = "gemini-3.8-flash"
HF_SPACE = "Lightricks/ltx-video-distilled"
N_SCENES = 6
TARGET_SECONDS = 30
OUT = "work"
SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
HISTORY = "history.json"


def parse_json(text):
    s, e = text.find("{"), text.rfind("}")
    if s == -1 or e == -1:
        raise ValueError("No JSON found in reply")
    return json.loads(text[s:e + 1])


def candidate_models(client):
    names = [GEMINI_MODEL]
    try:
        for m in client.models.list():
            n = (m.name or "").replace("models/", "")
            acts = getattr(m, "supported_actions", None) or []
            if "flash" in n and (not acts or "generateContent" in acts):
                if not any(k in n for k in ("image", "tts", "live", "audio", "embedding", "native")):
                    if n not in names:
                        names.append(n)
    except Exception as e:
        print("could not list models:", e)
    names = names[:5]
    print("Models to try:", names)
    return names


def get_idea():
    used = json.load(open(HISTORY)) if os.path.exists(HISTORY) else []
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    models = candidate_models(client)
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
    data = None
    for attempt in range(15):
        model = models[attempt % len(models)]
        try:
            r = client.models.generate_content(model=model, contents=ask)
            data = parse_json(r.text or "")
            scenes = []
            for s in data.get("scenes", []):
                if isinstance(s, dict):
                    s = s.get("prompt") or " ".join(str(v) for v in s.values())
                scenes.append(str(s))
            if len(scenes) < 2:
                raise ValueError("Not enough scenes")
            data["scenes"] = scenes
            data.setdefault("title", "Satisfying AI ASMR")
            data.setdefault("description", "")
            data.setdefault("idea", data["title"])
            print("Idea generated with model:", model)
            break
        except Exception as e:
            print(f"gemini attempt {attempt + 1} ({model}) failed: {str(e)[:200]}")
            data = None
            time.sleep(20)
    if data is None:
        raise RuntimeError("Gemini failed after retries")
    used.append(data["idea"])
    json.dump(used, open(HISTORY, "w"), ensure_ascii=False)
    return data


def extract_path(x):
    if isinstance(x, str):
        return x if os.path.exists(x) else None
    if isinstance(x, dict):
        for k in ("video", "path", "value", "name"):
            if k in x:
                r = extract_path(x[k])
                if r:
                    return r
        return None
    if isinstance(x, (list, tuple)):
        for i in x:
            r = extract_path(i)
            if r:
                return r
    return None


def list_endpoints(client):
    info = client.view_api(print_info=False, return_format="dict")
    eps = info.get("named_endpoints", {})
    print("Endpoints found:", list(eps.keys()))
    found = []
    for name, ep in eps.items():
        names = [p.get("parameter_name") for p in ep.get("parameters", [])]
        print(" ", name, "->", names)
        if "prompt" in names:
            low = name.lower()
            if "text" in low or "t2v" in low:
                score = 0
            elif any(k in low for k in ("image", "i2v", "v2v", "video_to")):
                score = 2
            else:
                score = 1
            found.append((score, name, names))
    found.sort(key=lambda t: t[0])
    if not found:
        raise RuntimeError("No endpoint with a 'prompt' parameter found")
    return found


def build_kwargs(names, prompt, full=True):
    kw = {"prompt": prompt}
    if full:
        if "height_ui" in names and "width_ui" in names:
            kw["height_ui"] = 704
            kw["width_ui"] = 512
        if "duration_ui" in names:
            kw["duration_ui"] = 5
    return kw


def try_generate(client, endpoints, prompt):
    for score, name, names in endpoints:
        for full in (True, False):
            try:
                res = client.predict(api_name=name, **build_kwargs(names, prompt, full))
                path = extract_path(res)
                if path:
                    return path
                print("No video path in result:", str(res)[:300])
            except Exception as e:
                print(f"{name} (full={full}) failed: {str(e)[:300]}")
                if "quota" in str(e).lower():
                    time.sleep(30)
    return None


def gen_clips(scenes):
    os.makedirs(OUT, exist_ok=True)
    client = Client(HF_SPACE, token=os.environ.get("HF_TOKEN"))
    endpoints = list_endpoints(client)
    paths = []
    for i, p in enumerate(scenes):
        src = None
        for attempt in range(2):
            src = try_generate(client, endpoints, p)
            if src:
                break
            time.sleep(20)
        if src:
            dst = f"{OUT}/clip{i}.mp4"
            shutil.copy(src, dst)
            paths.append(dst)
            print(f"scene {i} done")
        else:
            print(f"scene {i} failed, skipping")
    if len(paths) < 2:
        raise RuntimeError("Too few clips generated")
    return paths


def duration(path):
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", path],
        capture_output=True, text=True,
    )
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def merge(paths):
    vf = "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,fps=30"
    norm, durs = [], []
    for i, p in enumerate(paths):
        o = f"{OUT}/norm{i}.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-i", p, "-vf", vf, "-c:v", "libx264",
             "-pix_fmt", "yuv420p", "-an", o],
            check=True, capture_output=True,
        )
        norm.append(o)
        durs.append(duration(o))
    seq, total, i = [], 0.0, 0
    while total < TARGET_SECONDS and i < 60:
        seq.append(norm[i % len(norm)])
        total += durs[i % len(norm)]
        i += 1
    lst = f"{OUT}/list.txt"
    with open(lst, "w") as f:
        for p in seq:
            f.write(f"file '{os.path.abspath(p)}'\n")
    out = f"{OUT}/final.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst,
         "-t", str(TARGET_SECONDS), "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-an", out],
        check=True, capture_output=True,
    )
    print("Final video seconds:", duration(out))
    return out


def yt_service():
    creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds.valid:
        creds.refresh(Request())
    return build("youtube", "v3", credentials=creds)


def upload(video, d):
    yt = yt_service()
    tags = [str(t) for t in d.get("tags", [])] + ["Shorts"]
    body = {
        "snippet": {
            "title": d["title"][:88] + " #Shorts",
            "description": d["description"] + "\n#Shorts #ASMR #AI",
            "tags": tags,
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
