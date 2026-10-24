"""Web UI (Gradio) for recap.py -- no Streamlit.

Run:  python app.py        (needs recap.py in the same folder)
"""
import os
import shutil
import sys
import uuid

import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recap  # noqa: E402

BASE = os.path.join(os.getcwd(), "work_ui")
os.makedirs(BASE, exist_ok=True)
PORT = int(os.environ.get("PORT", "7860"))

LANGS = {"မြန်မာ": "my", "ဂျာမနီ": "de", "ဂျပန်": "ja"}


def voices_for(code):
    return [v for v, _ in recap.LANG_CONF[code]["edge_voices"]]


def _path(f):
    if not f:
        return ""
    return f if isinstance(f, str) else getattr(f, "name", "")


def _cb(progress):
    return lambda frac, label=None: progress(frac, desc=label or "")


# ---------------------------------------------------------------- step 1
def step1(url, upload, lang_name, model, src_lang, max_min, height, progress=gr.Progress()):
    source = _path(upload) or (url or "").strip()
    if not source:
        raise gr.Error("Video link ထည့်ပါ (သို့) ဖိုင်တင်ပါ")
    code = LANGS[lang_name]
    pdir = os.path.join(BASE, uuid.uuid4().hex[:8])
    cb = _cb(progress)
    try:
        cb(0.0, "Video ယူနေတယ်...")
        dl = recap.Progress(0.0, 0.3, parent=lambda f, l=None: cb(f, "Video ယူနေတယ်..."))
        proj = recap.init_project(pdir, source, int(height), os.environ.get("YT_COOKIES"), dl)
        tr = recap.Progress(0.3, 1.0, parent=lambda f, l=None: cb(f, l))
        info = recap.run_transcribe(proj, model, src_lang, True, int(max_min), code, tr)
    except Exception as ex:
        raise gr.Error(str(ex)[:600])
    prompt = open(proj.prompt_file, encoding="utf-8").read()
    return pdir, f"✅ {info}", prompt


# ---------------------------------------------------------------- step 2
def step2(pdir, translated, lang_name, engine, voice, rate, workers, vox_ref,
          height, fps, preset, max_speed, crop, flip, zoom,
          sub, sub_size, sub_pos, neon, progress=gr.Progress()):
    if not pdir or not os.path.isdir(pdir):
        raise gr.Error("အဆင့် 1 ကို အရင်လုပ်ပါ")
    code = LANGS[lang_name]
    try:
        proj = recap.Project.load(pdir)
        got, missing = recap.apply_translation(proj, translated or "", code)
        if missing:
            raise RuntimeError(f"ဘာသာပြန် မပြည့်သေးဘူး ({got}/{len(proj.units)}) · မပါတာ: {missing[:15]}")
        s = proj.settings
        s.lang, s.engine = code, ("vox" if engine == "VoxCPM" else "edge")
        s.voice = voice if voice in voices_for(code) else ""
        s.rate, s.tts_workers = int(rate), int(workers)
        s.vox_ref = _path(vox_ref)
        s.height, s.fps, s.preset = int(height), int(fps), preset
        s.max_speed, s.crop, s.flip, s.zoom = float(max_speed), bool(crop), bool(flip), float(zoom)
        s.sub, s.sub_size, s.sub_pos, s.neon = bool(sub), float(sub_size), int(sub_pos), bool(neon)
        prog = recap.Progress(0.0, 1.0, parent=_cb(progress))
        res = recap.run_render(proj, s, os.environ.get("EDGE_TTS_PROXY") or None,
                               os.environ.get("HF_TOKEN") or None, prog)
    except gr.Error:
        raise
    except Exception as ex:
        raise gr.Error(str(ex)[:600])
    out = os.path.join(pdir, f"final_{code}.mp4")
    shutil.copy(res["final"], out)
    mb = os.path.getsize(out) / 1e6
    return out, out, f"✅ ပြီးပြီ · {res['new_dur']:.1f}s (မူရင်း {res['old_dur']:.1f}s) · {mb:.1f} MB"


def reset(pdir):
    if pdir and os.path.isdir(pdir) and pdir.startswith(BASE):
        shutil.rmtree(pdir, ignore_errors=True)
    return "", "", "", "", None, None, "🗑️ ရှင်းပြီး"


# ---------------------------------------------------------------- UI
with gr.Blocks(title="Recap → Multi-Language") as demo:
    gr.Markdown("# 🎙️ Recap → Multi-Language\nWhisper → Gemini → Edge TTS / VoxCPM → Smart Sync → စာတန်း + Neon")
    pdir = gr.State("")

    with gr.Accordion("🔊 ဘာသာစကား + အသံ", open=True):
        lang = gr.Radio(list(LANGS), value="မြန်မာ", label="ဘာသာစကား")
        engine = gr.Radio(["Edge TTS", "VoxCPM"], value="Edge TTS", label="TTS engine")
        voice = gr.Dropdown(voices_for("my"), value=voices_for("my")[0], label="Edge voice")
        with gr.Row():
            rate = gr.Slider(-50, 100, value=45, step=5, label="Edge speed %")
            workers = gr.Slider(1, 10, value=8, step=1, label="TTS တစ်ပြိုင်နက်")
        vox_ref = gr.File(label="🎤 VoxCPM reference audio (optional)", type="filepath")
        lang.change(lambda l: gr.update(choices=voices_for(LANGS[l]), value=voices_for(LANGS[l])[0]),
                    lang, voice)

    with gr.Accordion("⚙️ Video / စာတန်း / Neon", open=False):
        with gr.Row():
            height = gr.Dropdown([360, 480, 720, 1080], value=720, label="Output height")
            fps = gr.Dropdown([20, 24, 25, 30], value=24, label="FPS")
            preset = gr.Dropdown(["ultrafast", "superfast", "veryfast"], value="ultrafast", label="Render speed")
        with gr.Row():
            max_speed = gr.Slider(1.0, 2.0, value=1.25, step=0.05, label="ရုပ် အမြန်ဆုံး ဆ")
            zoom = gr.Slider(1.0, 1.25, value=1.0, step=0.01, label="Zoom")
        with gr.Row():
            crop = gr.Checkbox(True, label="အနက်ဘောင် ဖြတ်")
            flip = gr.Checkbox(True, label="Mirror")
            sub = gr.Checkbox(True, label="စာတန်း")
            neon = gr.Checkbox(True, label="Neon")
        with gr.Row():
            sub_size = gr.Slider(3.0, 12.0, value=7.0, step=0.5, label="စာအရွယ် (% width)")
            sub_pos = gr.Slider(0, 100, value=90, step=1, label="နေရာ (0=အပေါ် 100=အောက်)")
        with gr.Row():
            model = gr.Dropdown(["tiny", "base", "small"], value="base", label="Whisper model")
            src_lang = gr.Dropdown(["auto", "en", "zh", "ko", "ja", "hi", "th", "vi", "id"],
                                   value="auto", label="မူရင်းဘာသာ")
            max_min = gr.Slider(1, 120, value=15, step=1, label="အရှည်ကန့်သတ် (မိနစ်)")

    gr.Markdown("### 1️⃣ Video")
    url = gr.Textbox(label="Video link (mp4 / Google Drive / Dropbox / YouTube / FB / TikTok ...)",
                     placeholder="https://...")
    upload = gr.File(label="(သို့) Video ဖိုင်တင်", type="filepath",
                     file_types=[".mp4", ".mov", ".mkv", ".webm", ".m4v"])
    b1 = gr.Button("1️⃣ စာသားထုတ်", variant="primary")
    status = gr.Markdown()
    prompt = gr.Textbox(label="📋 Gemini Prompt (Copy ယူပြီး Gemini မှာ paste)", lines=8,
                        interactive=False)

    gr.Markdown("### 2️⃣ ဘာသာပြန်ရလဒ်")
    translated = gr.Textbox(label="Gemini ရလဒ် paste ([0] ... [1] ...)", lines=8)
    b2 = gr.Button("2️⃣ အသံထုတ် + Video ပေါင်း", variant="primary")
    status2 = gr.Markdown()
    video = gr.Video(label="ရလဒ်")
    dl = gr.File(label="⬇️ Download MP4")
    b3 = gr.Button("🗑️ အသစ်ပြန်စ")

    b1.click(step1, [url, upload, lang, model, src_lang, max_min, height], [pdir, status, prompt])
    b2.click(step2, [pdir, translated, lang, engine, voice, rate, workers, vox_ref, height, fps,
                     preset, max_speed, crop, flip, zoom, sub, sub_size, sub_pos, neon],
             [video, dl, status2])
    b3.click(reset, pdir, [pdir, status, prompt, translated, video, dl, status2])

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=PORT, show_error=True)