"""Recap CLI (slim): Whisper -> Gemini (manual) -> Edge TTS -> smart-sync render -> auto split.

Languages: Burmese (my) / German (de)  ·  Sources: YouTube / TikTok (yt-dlp) or local file

  python recap_slim.py <video|url> -d work/ep1 --lang my
  python recap_slim.py <video|url> -d work/ep1 --lang de --no-split --flip-every 5
  python recap_slim.py <video|url> -d work/ep1 --bgm music.mp3 --bgm-vol 0.3
"""
import argparse
import asyncio
import glob
import hashlib
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import wave
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

TARGET, MAXD = 4.2, 5.6      # Whisper segment merge targets (seconds)
CHUNK = 100                  # lines per Gemini prompt part
SUB_LINES = 2
TRIM = ("silenceremove=start_periods=1:start_threshold=-50dB,areverse,"
        "silenceremove=start_periods=1:start_threshold=-50dB,areverse,apad=pad_dur=0.2")

# ----------------------------------------------------------------------
# prompts
# ----------------------------------------------------------------------
_CORE = (
    "RECAP MODE: first read ALL numbered lines and understand the whole story. Then retell each "
    "line as third-person drama-recap narration, NOT as a translation. Condense: drop filler and "
    "small talk, keep only what moves the story. Each line should be about 50-70% as long as a "
    "literal translation. Never leave a line empty (write one short bridge sentence). Do NOT move "
    "content between lines and do NOT invent events. The source is raw speech-to-text of "
    "characters talking: turn speech into reported speech, no quotation marks, work out who talks "
    "to whom and use names or role words. Speech-to-text misspells names: use ONE spelling per "
    "character. Translate meaning, idioms and emotion, never word by word. Write for a speech "
    "engine: short sentences, a comma at breath points, no ellipsis/emoji/brackets, numbers as "
    "spoken words. Vary connectors and sentence openings. No hooks, no commentary, do not "
    "address the audience. "
)
_MY = (
    "You are a calm Burmese movie/drama recap narrator. Use ONLY Burmese script (no English or "
    "Chinese letters), present tense, third person. Do NOT use ကျွန်တော် or ဗျ. About three lines "
    "out of four end with ပါတယ်။ (never ၏ / သည်။). Use connectors like အဲ့ဒါနဲ့၊ ဒါပေမဲ့၊ "
    "အဲဒီအချိန်မှာပဲ၊ ပြီးတော့၊ ဒါကြောင့်၊ ရုတ်တရက် (never the same one twice in a row) and "
    "speech verbs like လို့ ပြောပါတယ်၊ လို့ မေးပါတယ်၊ လို့ ဖြေပါတယ်၊ လို့ အော်ပါတယ်။ Write names in "
    "Burmese script as the subject (ဝါလ်တာက ...). Unknown name: use a role word (အမေ၊ သူငယ်ချင်း၊ "
    "သူဌေး) or drop the subject, never လူတစ်ယောက်. Numbers as Burmese words (၇၀ → ခုနစ်ဆယ်). "
)
_DE = (
    "You are a native German storyteller retelling a TV drama to a friend. Use ONLY German (Latin "
    "script). Tell it in the present tense (Präsens); Perfekt only for earlier events; no "
    "Präteritum narration. Reported speech with a normal 'dass' clause or main clause, NEVER "
    "Konjunktiv I. Never the same reporting verb twice in a row, 'sagt' at most once every four "
    "lines. Everyday spoken words, a few modal particles (doch, ja, halt, mal), no office German or "
    "Nominalstil. Keep Asian names in Latin/Pinyin spelling; unknown person: a role word (die "
    "Mutter, der Chef, der junge Mann), never 'jemand'. Use er/sie naturally but the name when two "
    "same-gender characters are present. Numbers and symbols as words ('fünfzig Prozent'). "
)
_OUT = ("Output exactly one line per input line as [number] text -- same numbers, same count, no "
        "comments, no headings, no blank lines.\n\n")

LANG = {
    "my": {"role": "You are a Burmese (Myanmar) movie/drama recap narrator. ", "style": _MY,
           "rate": 45, "voice": "my-MM-ThihaNeural", "voices": ("my-MM-ThihaNeural", "my-MM-NilarNeural"),
           "sentence_end": r"(?<=။)", "is_mm": True,
           "fonts": ("padauk", "myanmar", "pyidaung", "notosansmyanmar"),
           "clean": r"[^\u1000-\u109f\uaa60-\uaa7f\ua9e0-\ua9ff\s0-9.,!?\"'%\-]"},
    "de": {"role": "You are a native German storyteller. ", "style": _DE,
           "rate": 10, "voice": "de-DE-KatjaNeural",
           "voices": ("de-DE-KatjaNeural", "de-DE-ConradNeural", "de-DE-AmalaNeural", "de-DE-BerndNeural"),
           "sentence_end": r"(?<=[.!?])\s+", "is_mm": False,
           "fonts": ("dejavusans", "liberation", "notosans", "arial", "freesans"),
           "clean": r"[^A-Za-zÄÖÜäöüß0-9\s.,!?\"'%\-]"},
}


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def sh(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-1200:])
    return r.stdout


def probe_duration(path):
    return float(sh(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                     "-of", "csv=p=0", path]).strip())


def wav_dur(path):
    with wave.open(path) as w:
        return w.getnframes() / float(w.getframerate())


def tick(label, frac, state):
    """print a progress line every ~10%"""
    step = int(frac * 10)
    if step != state.get("s"):
        state["s"] = step
        log(f"  {label} {int(frac * 100)}%")


# ----------------------------------------------------------------------
# download
# ----------------------------------------------------------------------
def extract_url(text):
    text = (text or "").strip()
    if os.path.isfile(text):
        return text
    m = re.search(r"https?://[^\s\"'<>]+", text)
    return m.group(0).rstrip(".,;:!?") if m else text


def ensure_mp4(raw, final, height):
    codec = sh(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                "stream=codec_name", "-of", "csv=p=0", raw]).strip()
    if not codec:
        raise RuntimeError("Video stream မတွေ့ပါ")
    if codec == "h264":
        try:
            sh(["ffmpeg", "-y", "-i", raw, "-c", "copy", "-movflags", "+faststart", final])
            return
        except Exception:
            pass
    sh(["ffmpeg", "-y", "-i", raw, "-vf", f"scale=-2:'min(ih,{height})'", "-c:v", "libx264",
        "-preset", "ultrafast", "-crf", "26", "-pix_fmt", "yuv420p", "-c:a", "aac",
        "-b:a", "96k", "-movflags", "+faststart", final])


def fetch_video(source, final, height):
    source = extract_url(source)
    if not re.match(r"^https?://", source, re.I):
        if not os.path.isfile(source):
            raise FileNotFoundError(f"ဖိုင် မတွေ့ပါ: {source}")
        return ensure_mp4(source, final, height)
    import yt_dlp
    tmp = tempfile.mkdtemp(prefix="dl_")
    try:
        opts = {"format": f"bv*[height<={height}][vcodec^=avc1]+ba[ext=m4a]/b[height<={height}]/bv*+ba/b",
                "outtmpl": os.path.join(tmp, "yt.%(ext)s"), "merge_output_format": "mp4",
                "noplaylist": True, "quiet": True, "no_warnings": True, "retries": 5,
                "concurrent_fragment_downloads": 4}
        cookies = os.environ.get("YT_COOKIES")
        if cookies and os.path.exists(cookies):
            opts["cookiefile"] = cookies
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([source])
        files = [p for p in glob.glob(os.path.join(tmp, "yt.*")) if not p.endswith((".part", ".ytdl"))]
        if not files:
            raise RuntimeError("yt-dlp ဖိုင် မထွက်ပါ")
        ensure_mp4(max(files, key=os.path.getsize), final, height)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ----------------------------------------------------------------------
# whisper
# ----------------------------------------------------------------------
def transcribe(video, wd, model_size, src_lang):
    from faster_whisper import WhisperModel
    vdur = probe_duration(video)
    wav = os.path.join(wd, "audio16k.wav")
    sh(["ffmpeg", "-y", "-err_detect", "ignore_err", "-i", video, "-vn", "-ac", "1", "-ar",
        "16000", "-c:a", "pcm_s16le", wav])
    with wave.open(wav) as wf:
        audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    wm = WhisperModel(model_size, device="cpu", compute_type="int8", cpu_threads=os.cpu_count() or 4)
    lang = None if src_lang == "auto" else src_lang
    raw, st = [], {}
    try:
        from faster_whisper import BatchedInferencePipeline
        segs, info = BatchedInferencePipeline(model=wm).transcribe(
            audio, language=lang, beam_size=1, batch_size=8)
    except Exception:
        segs, info = wm.transcribe(audio, language=lang, vad_filter=True, beam_size=1,
                                   temperature=0.0, condition_on_previous_text=False)
    for s in segs:
        if s.text.strip():
            raw.append((s.start, s.end, s.text.strip()))
        tick("Whisper", min(s.end / vdur, 1.0), st)
    units = []
    for s, e, t in raw:
        if units and (units[-1]["end"] - units[-1]["start"]) < TARGET and (e - units[-1]["start"]) <= MAXD:
            units[-1]["end"], units[-1]["src"] = e, units[-1]["src"] + " " + t
        else:
            units.append({"start": s, "end": e, "src": t, "tr": ""})
    if not units:
        raise RuntimeError("စကားပြောသံ မတွေ့ပါ")
    log(f"  ဘာသာစကား: {info.language} · အပိုင်း {len(units)} ခု")
    return units, vdur


# ----------------------------------------------------------------------
# prompts (Gemini, manual)
# ----------------------------------------------------------------------
def write_prompts(units, lang, wd):
    c = LANG[lang]
    head = c["role"] + _CORE + c["style"] + _OUT
    total = (len(units) + CHUNK - 1) // CHUNK
    for k in range(total):
        lo, hi = k * CHUNK, min((k + 1) * CHUNK, len(units))
        note = (f"(PART {k + 1} of {total} of the same episode. Only lines [{lo}] to [{hi - 1}]. "
                "Keep the SAME numbers, names and style as the earlier parts.)\n\n")
        lines = "\n".join(f"[{i}] {units[i]['src']}" for i in range(lo, hi))
        with open(os.path.join(wd, f"prompt_part{k + 1:02d}.txt"), "w", encoding="utf-8") as f:
            f.write(head + note + lines)
    return total


def parse_numbered(text, n):
    got = {}
    for line in text.splitlines():
        line = line.strip().strip("*").strip()
        m = re.match(r"^[\[\(#]?\s*(\d+)\s*[\]\)\.:\-–]?\s*(.+)$", line)
        if m and int(m.group(1)) < n:
            got[int(m.group(1))] = m.group(2).strip().strip("*").strip()
    return got


# ----------------------------------------------------------------------
# TTS (Edge)
# ----------------------------------------------------------------------
def tts_line(text, voice, rate, mp3, proxy):
    import edge_tts

    async def go():
        kw = {"proxy": proxy} if proxy else {}
        await edge_tts.Communicate(text, voice, rate=rate, **kw).save(mp3)
    asyncio.run(go())


def synthesize(texts, wd, voice, rate, workers, proxy):
    n = len(texts)
    sig = f"edge|{voice}|{rate}"

    def one(i):
        wav, keyf = os.path.join(wd, f"a{i:04d}.wav"), os.path.join(wd, f"a{i:04d}.key")
        key = hashlib.md5(f"{sig}|{texts[i]}".encode()).hexdigest()
        if os.path.exists(wav) and os.path.exists(keyf) and open(keyf).read().strip() == key:
            return wav_dur(wav)
        mp3 = os.path.join(wd, f"a{i:04d}_raw.mp3")
        for attempt in range(4):
            try:
                tts_line(texts[i], voice, rate, mp3, proxy)
                break
            except Exception as ex:
                if attempt == 3:
                    raise RuntimeError(f"Edge TTS မအောင်မြင်ပါ (စာကြောင်း #{i}): {ex}")
        sh(["ffmpeg", "-y", "-i", mp3, "-af", TRIM, "-ar", "44100", "-ac", "1", wav])
        open(keyf, "w").write(key)
        return wav_dur(wav)

    durs, st = [0.0] * n, {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for k, d in enumerate(pool.map(one, range(n))):
            durs[k] = d
            tick("TTS", (k + 1) / n, st)
    return durs


def mix_audio(durs, wd, a, total):
    lst = os.path.join(wd, "alist.txt")
    with open(lst, "w") as f:
        for i in range(len(durs)):
            f.write(f"file '{os.path.join(wd, f'a{i:04d}.wav')}'\n")
    voice = os.path.join(wd, "all.wav")
    sh(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", voice])
    out = os.path.join(wd, "mix.wav")
    fmt = "aresample=44100,aformat=sample_fmts=fltp:channel_layouts=stereo"
    gain = f"loudnorm=I={a.loud:g}:TP=-1.5:LRA=11,volume={a.voice_db:g}dB"
    tail = ["-map", "[o]", "-t", f"{total:.3f}", "-ar", "44100", "-c:a", "pcm_s16le", out]
    if not (a.bgm and os.path.isfile(a.bgm)):
        if a.bgm:
            log(f"⚠️ BGM ဖိုင် မတွေ့ပါ: {a.bgm} — BGM မပါဘဲ ဆက်သွားမယ်")
        sh(["ffmpeg", "-y", "-i", voice, "-filter_complex",
            f"[0:a]{fmt},{gain},alimiter=limit=0.97[o]", *tail])
        return out
    bg = f"[1:a]{fmt},volume={a.bgm_vol:g},atrim=0:{total:.3f}[bg]"

    def graph(opts, end):
        if a.duck:
            return (f"[0:a]{fmt},{gain},asplit=2[v][sc];{bg};"
                    "[bg][sc]sidechaincompress=threshold=0.02:ratio=10:attack=30:release=600[d];"
                    f"[v][d]amix=inputs=2:duration=first:dropout_transition=0{opts}{end}[o]")
        return (f"[0:a]{fmt},{gain}[v];{bg};"
                f"[v][bg]amix=inputs=2:duration=first:dropout_transition=0{opts}{end}[o]")

    def run(fc):
        sh(["ffmpeg", "-y", "-i", voice, "-stream_loop", "-1", "-i", a.bgm,
            "-filter_complex", fc, *tail])
    try:
        run(graph(":normalize=0", ",alimiter=limit=0.97"))
    except RuntimeError:      # ffmpeg < 4.4 has no amix normalize
        run(graph("", ",volume=2,alimiter=limit=0.97"))
    return out


# ----------------------------------------------------------------------
# fonts + subtitles
# ----------------------------------------------------------------------
def find_font(lang, explicit=""):
    if explicit and os.path.exists(explicit):
        return explicit
    env = os.environ.get("RECAP_FONT", "")
    if LANG[lang]["is_mm"] and env and os.path.exists(env):
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    files = []
    for base in (here, os.getcwd(), os.path.join(here, "fonts")):
        for ext in ("ttf", "otf", "ttc"):
            files += glob.glob(os.path.join(base, f"*.{ext}"))
    for ext in ("ttf", "otf", "ttc"):
        files += glob.glob(f"/usr/share/fonts/**/*.{ext}", recursive=True)
        files += glob.glob(f"/Library/Fonts/**/*.{ext}", recursive=True)
        files += glob.glob(f"C:/Windows/Fonts/*.{ext}")
    files.sort(key=lambda p: ("bold" in p.lower() or "italic" in p.lower(), len(p)))
    for key in LANG[lang]["fonts"]:
        for f in files:
            if key in os.path.basename(f).lower():
                return f
    return None


def get_font(path, size):
    try:
        return ImageFont.truetype(path, size, layout_engine=ImageFont.Layout.RAQM)
    except Exception:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            return ImageFont.load_default()


def split_syll(word):
    def start(c):
        return (0x1000 <= c <= 0x1021) or (0x1023 <= c <= 0x1027) or c in (0x1029, 0x102A, 0x103F)
    idx, n = [0], len(word)
    for i in range(1, n):
        if not start(ord(word[i])):
            continue
        nxt = ord(word[i + 1]) if i + 1 < n else 0
        if ord(word[i - 1]) == 0x1039 or nxt in (0x103A, 0x1039):
            continue
        idx.append(i)
    idx.append(n)
    return [word[idx[k]:idx[k + 1]] for k in range(len(idx) - 1)]


def wrap_text(text, font, maxw, is_mm):
    lines, cur = [], ""
    for word in text.split():
        pieces = [word] if font.getlength(word) <= maxw else (split_syll(word) if is_mm else list(word))
        for pi, p in enumerate(pieces):
            cand = cur + (" " if (pi == 0 and cur) else "") + p
            if not cur or font.getlength(cand) <= maxw:
                cur = cand
            else:
                lines.append(cur)
                cur = p
    if cur:
        lines.append(cur)
    return lines


def split_for_sub(text, lang, mc=36):
    c = LANG[lang]
    parts = []
    for s in [s.strip() for s in re.split(c["sentence_end"], text) if s.strip()]:
        if len(s) <= mc:
            parts.append(s)
            continue
        cur = ""
        for word in s.split():
            pieces = [word] if len(word) <= mc else (split_syll(word) if c["is_mm"] else list(word))
            for pi, p in enumerate(pieces):
                sep = " " if (pi == 0 and cur) else ""
                if not cur or len(cur + sep + p) <= mc:
                    cur += sep + p
                else:
                    parts.append(cur)
                    cur = p
        if cur:
            parts.append(cur)
    return parts or [text]


def build_overlay(text, W, H, font_path, fs, pos_y, alpha, is_mm):
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    pad_x, pad_y = int(fs * 0.6), int(fs * 0.5)
    box_h = min(SUB_LINES * int(fs * 1.4) + 2 * pad_y, H)
    box_y = max(0, min(int(pos_y / 100 * max(0, H - box_h)), max(0, H - box_h)))
    cur_fs = fs
    font = get_font(font_path, cur_fs)
    lines = wrap_text(text, font, W - 2 * pad_x, is_mm)
    while len(lines) > SUB_LINES and cur_fs > 12:
        cur_fs = max(12, int(cur_fs * 0.92))
        font = get_font(font_path, cur_fs)
        lines = wrap_text(text, font, W - 2 * pad_x, is_mm)
    if not lines:
        return None
    d.rounded_rectangle([0, box_y, W, box_y + box_h], radius=int(fs * 0.66), fill=(0, 0, 0, alpha))
    lh, sw = int(cur_fs * 1.4), max(1, round(cur_fs / 15))
    ty = box_y + (box_h - len(lines) * lh) // 2
    for ln in lines:
        d.text(((W - font.getlength(ln)) / 2, ty), ln, font=font, fill=(255, 255, 255, 255),
               stroke_width=sw, stroke_fill=(0, 0, 0, 255))
        ty += lh
    arr = np.array(img)
    rows = np.where(arr[:, :, 3].any(axis=1))[0]
    if len(rows) == 0:
        return None
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    a = arr[y0:y1, :, 3:4].astype(np.float32) / 255.0
    bgr = arr[y0:y1, :, [2, 1, 0]].astype(np.float32)
    return y0, y1, 1.0 - a, bgr * a


def blend_overlay(fr, ov):
    y0, y1, inv, pre = ov
    fr[y0:y1] = np.clip(fr[y0:y1].astype(np.float32) * inv + pre, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------
# render
# ----------------------------------------------------------------------
def detect_crop(src):
    r = subprocess.run(["ffmpeg", "-ss", "1", "-i", src, "-t", "8", "-vf", "cropdetect=24:2:0",
                        "-f", "null", "-"], capture_output=True, text=True)
    m = re.findall(r"crop=(\d+):(\d+):(\d+):(\d+)", r.stderr)
    return tuple(int(x) for x in m[-1]) if m else None


def flip_at(a, t):
    """mirror on/off at time t. --flip-every N: 0..N mirror, N..2N normal, ..."""
    if not a.flip:
        return False
    if a.flip_every <= 0:
        return True
    return int(t // a.flip_every) % 2 == 0


def build_plan(units, durs, vdur, max_speed):
    n = len(units)
    bounds = [0.0] + [(units[i]["end"] + units[i + 1]["start"]) / 2 for i in range(n - 1)] + [vdur]
    plan, cum = [], 0.0
    for i in range(n):
        w, d = max(bounds[i + 1] - bounds[i], 0.2), max(durs[i], 0.05)
        w_use = d * max_speed if w / d > max_speed else w
        plan.append((cum, d, bounds[i], w_use / d))
        cum += d
    return plan, cum


def build_subs(texts, durs, lang):
    subs, t0 = [], 0.0
    for i, text in enumerate(texts):
        parts = split_for_sub(text, lang)
        tot, cur = sum(len(p) for p in parts) or 1, t0
        for p in parts:
            dd = durs[i] * len(p) / tot
            subs.append((cur, cur + dd, p))
            cur += dd
        t0 += durs[i]
    return subs


def render_final(src, audio, out, plan, total, a, subs, font_path, is_mm):
    cap = cv2.VideoCapture(src)
    src_w, src_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sfps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stotal = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 10 ** 9
    c = detect_crop(src) if a.crop else None
    if c and c[1] < src_h * 0.95 and c[0] > 0 and c[1] > 0:
        cw, ch, cx, cy = c
        cw, ch = min(cw, src_w - cx), min(ch, src_h - cy)
    else:
        cw, ch, cx, cy = src_w, src_h, 0, 0
    H = a.height // 2 * 2
    W = int(round(cw * H / ch / 2)) * 2
    interp = cv2.INTER_AREA if H < ch else cv2.INTER_LINEAR
    fps_out = a.fps
    nfr = max(1, int(round(total * fps_out)))
    fs = max(16, int(W * a.sub_size / 100))

    q, stop = queue.Queue(maxsize=32), threading.Event()

    def put(x):
        while not stop.is_set():
            try:
                q.put(x, timeout=0.5)
                return
            except queue.Full:
                continue

    def producer():
        cur, last, ui = -1, None, 0
        try:
            for j in range(nfr):
                if stop.is_set():
                    return
                T = j / fps_out
                while ui < len(plan) - 1 and T >= plan[ui + 1][0]:
                    ui += 1
                cs, _d, s0, spd = plan[ui]
                idx = min(int((s0 + (T - cs) * spd) * sfps + 0.5), stotal - 1)
                if idx > cur + 90:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    cur = idx - 1
                new = False
                while cur < idx:
                    if not cap.grab():
                        break
                    cur += 1
                    new = True
                if new:
                    ok, f = cap.retrieve()
                    if ok:
                        f = f[cy:cy + ch, cx:cx + cw]
                        if f.shape[1] != W or f.shape[0] != H:
                            f = cv2.resize(f, (W, H), interpolation=interp)
                        last = f
                if last is None:
                    return
                put(last)
        finally:
            put(None)

    kf = []
    if a.split and a.split_min > 0:   # keyframe at every split boundary -> `-c copy` cuts are exact
        kf = ["-force_key_frames", f"expr:gte(t,n_forced*{a.split_min * 60.0:g})"]
    log_path = out + ".log"
    log_fh = open(log_path, "wb")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{W}x{H}", "-r", str(fps_out), "-i", "-", "-i", audio,
           "-map", "0:v", "-map", "1:a:0", "-c:v", "libx264", "-preset", a.preset, "-crf", "27",
           *kf, "-pix_fmt", "yuv420p", "-threads", "0", "-c:a", "aac", "-b:a", "128k",
           "-movflags", "+faststart", "-t", f"{total:.3f}", out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log_fh)
    th = threading.Thread(target=producer, daemon=True)
    th.start()

    k, cur_k, cur_ov, j, st = 0, -1, None, 0, {}
    try:
        while True:
            fr = q.get()
            if fr is None:
                break
            t = j / fps_out
            fr = cv2.flip(fr, 1) if flip_at(a, t) else fr.copy()
            if subs and font_path:
                while k < len(subs) and subs[k][1] < t:
                    k += 1
                if k < len(subs) and subs[k][0] <= t:
                    if cur_k != k:
                        cur_ov = build_overlay(subs[k][2], W, H, font_path, fs, a.sub_pos, 204, is_mm)
                        cur_k = k
                    if cur_ov:
                        blend_overlay(fr, cur_ov)
            proc.stdin.write(fr.data)
            j += 1
            if j % 30 == 0:
                tick("Render", min(j / nfr, 1.0), st)
    except BrokenPipeError:
        pass
    finally:
        stop.set()
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            pass
        th.join(timeout=5)
        cap.release()
        try:
            proc.stdin.close()
        except Exception:
            pass
        proc.wait()
        log_fh.close()
    if proc.returncode != 0:
        tail = open(log_path, "rb").read()[-800:].decode("utf-8", "ignore")
        raise RuntimeError("render: ffmpeg fail\n" + tail)
    os.remove(log_path)


def split_video(src, prefix, seg_min):
    """cut into <prefix>_part01.mp4 ...; leftover seconds go into the LAST part"""
    total, seg = probe_duration(src), seg_min * 60.0
    if seg <= 0 or total <= seg:
        return [src]
    n = max(1, int(total // seg))
    outs = []
    for i in range(n):
        st = i * seg
        dur = (total if i == n - 1 else (i + 1) * seg) - st
        out = f"{prefix}_part{i + 1:02d}.mp4"
        sh(["ffmpeg", "-y", "-ss", f"{st:.3f}", "-i", src, "-t", f"{dur:.3f}", "-c", "copy",
            "-avoid_negative_ts", "make_zero", "-movflags", "+faststart", out])
        outs.append(out)
    return outs


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(description="Recap video dubber (slim)")
    p.add_argument("source", help="YouTube / TikTok URL or local file")
    p.add_argument("-d", "--dir", default=os.path.join("work", "recap"), help="project folder")
    p.add_argument("--translated", help="Gemini ရလဒ် ဖိုင် (default <dir>/translated.txt)")
    p.add_argument("-o", "--output", help="final mp4 ကို ဒီနေရာ copy")
    p.add_argument("--lang", choices=list(LANG), default="my")
    p.add_argument("--voice", default="", help="Edge voice name")
    p.add_argument("--rate", type=int, default=None, help="Edge speed %% (default my 45 / de 10)")
    p.add_argument("--workers", type=int, default=8, help="parallel TTS jobs")
    p.add_argument("--model", choices=["tiny", "base", "small"], default="base")
    p.add_argument("--src-lang", default="auto", help="auto | en | zh | ko | ja ...")
    p.add_argument("--loud", type=float, default=-14.0, help="voice LUFS")
    p.add_argument("--voice-db", type=float, default=0.0)
    p.add_argument("--bgm", default="", help="background music file")
    p.add_argument("--bgm-vol", type=float, default=0.25)
    p.add_argument("--no-duck", dest="duck", action="store_false")
    p.add_argument("--height", type=int, choices=[360, 480, 720, 1080], default=720)
    p.add_argument("--fps", type=int, choices=[20, 24, 25, 30], default=24)
    p.add_argument("--preset", choices=["ultrafast", "superfast", "veryfast"], default="ultrafast")
    p.add_argument("--max-speed", type=float, default=1.25)
    p.add_argument("--no-crop", dest="crop", action="store_false")
    p.add_argument("--no-flip", dest="flip", action="store_false", help="mirror လုံးဝ မလုပ်ဘူး")
    p.add_argument("--flip-every", type=float, default=5.0,
                   help="mirror ကို N စက္ကန့်တိုင်း ဘယ်ညာလည့် (default 5 · 0 = အမြဲ mirror)")
    p.add_argument("--split", dest="split_min", type=float, default=2.0, help="မိနစ်အလိုက် အပိုင်းခွဲ")
    p.add_argument("--no-split", dest="split", action="store_false")
    p.add_argument("--no-sub", dest="sub", action="store_false")
    p.add_argument("--sub-size", type=float, default=7.0, help="%% of width")
    p.add_argument("--sub-pos", type=int, default=90, help="0=top 100=bottom")
    p.add_argument("--font", default="", help="font file path")
    return p


def main():
    a = build_parser().parse_args()
    c = LANG[a.lang]
    voice = a.voice if a.voice in c["voices"] else c["voice"]
    if a.voice and a.voice != voice:
        log(f"⚠️ voice '{a.voice}' မရှိဘူး — {voice} သုံးမယ်။ ရနိုင်တာ: {', '.join(c['voices'])}")
    rate = f"{(c['rate'] if a.rate is None else max(-50, min(100, a.rate))):+d}%"
    a.max_speed = max(1.0, min(2.0, a.max_speed))
    a.flip_every = max(0.0, a.flip_every)
    a.split_min = max(0.0, a.split_min)

    wd = os.path.abspath(a.dir)
    os.makedirs(wd, exist_ok=True)
    video, meta = os.path.join(wd, "src.mp4"), os.path.join(wd, "units.json")

    # 1) download
    if os.path.exists(video):
        log("• video ရှိပြီးသား")
    else:
        log("• video ယူနေတယ်")
        fetch_video(a.source, video, a.height)

    # 2) whisper
    if os.path.exists(meta):
        log("• units ရှိပြီးသား၊ whisper ကျော်မယ်")
        data = json.load(open(meta, encoding="utf-8"))
        units, vdur = data["units"], data["vdur"]
    else:
        log("• Whisper")
        units, vdur = transcribe(video, wd, a.model, a.src_lang)
        json.dump({"units": units, "vdur": vdur}, open(meta, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
    nparts = write_prompts(units, a.lang, wd)

    # 3) Gemini (manual)
    src = a.translated or os.path.join(wd, "translated.txt")
    if not os.path.exists(src):
        print(f"\n📋 {wd}/prompt_part01..{nparts:02d}.txt ကို Gemini chat တစ်ခုတည်းမှာ အစဉ်လိုက် paste")
        print(f"   ရလဒ်အကုန်ကို {src} မှာ သိမ်းပါ")
        if not sys.stdin.isatty():
            sys.exit(f"⏸️ ဖြည့်ပြီးရင် --translated {src} နဲ့ ပြန် run ပါ")
        input("   ပြီးရင် Enter နှိပ်ပါ ▶ ")
    got = parse_numbered(open(src, encoding="utf-8").read(), len(units))
    texts = [re.sub(c["clean"], "", got.get(i, "")).strip() for i in range(len(units))]
    missing = [i for i, t in enumerate(texts) if not t]
    if missing:
        sys.exit(f"❌ စာကြောင်း မပြည့်သေးပါ ({len(units) - len(missing)}/{len(units)}) · မပါတာ: {missing[:15]}")

    # 4) TTS + mix
    durs = synthesize(texts, wd, voice, rate, a.workers, os.environ.get("EDGE_TTS_PROXY") or None)
    plan, total = build_plan(units, durs, vdur, a.max_speed)
    mixed = mix_audio(durs, wd, a, total)

    # 5) render
    font = find_font(a.lang, a.font) if a.sub else None
    if a.sub and not font:
        log("⚠️ font မတွေ့ပါ — စာတန်းမပါဘဲ render လုပ်မယ် (--font သို့ fonts/ ထဲ ထည့်ပါ)")
    subs = build_subs(texts, durs, a.lang) if font else []
    final = os.path.join(wd, "final.mp4")
    render_final(video, mixed, final, plan, total, a, subs, font, c["is_mm"])
    out = a.output or final
    if a.output:
        shutil.copy(final, out)
    print(f"\n✅ {out}  ({os.path.getsize(out) / 1e6:.1f} MB · {total:.0f}s, မူရင်း {vdur:.0f}s)")

    # 6) split
    if a.split and a.split_min > 0:
        prefix = os.path.splitext(out)[0]
        for old in glob.glob(prefix + "_part*.mp4"):
            os.remove(old)
        parts = split_video(out, prefix, a.split_min)
        if len(parts) > 1:
            print(f"✂️ {len(parts)} ပိုင်း:")
            for p in parts:
                print(f"   {p}  ({probe_duration(p):.0f}s)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\n⛔ ရပ်လိုက်ပါပြီ (ပြန် run ရင် ပြီးသား အသံတွေ cache ကနေ ယူမယ်)")
    except Exception as ex:
        sys.exit(f"\n❌ {ex}")
