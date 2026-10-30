"""Recap CLI (single file): Whisper -> Gemini (manual) -> TTS -> smart-sync render.

Usage: python recap.py run <video|url> -d work/ep1 --lang my
"""
__version__ = "1.6.0"

# ======================================================================
# config
# ======================================================================
"""Constants, language table and render/TTS settings."""
from dataclasses import dataclass, asdict, fields

# Whisper segment merge targets (seconds)
TARGET, MAXD = 4.2, 5.6

# Subtitle box always has this many text lines (fixed size, never grows/shrinks)
SUB_LINES = 2

# Preview scrub range (seconds)
PREVIEW_SECS = 5.0

# Trim leading/trailing silence from each TTS clip, then pad 0.2s
TRIM = (
    "silenceremove=start_periods=1:start_threshold=-50dB,areverse,"
    "silenceremove=start_periods=1:start_threshold=-50dB,areverse,apad=pad_dur=0.2"
)

VOX_SPACES = [
    {"space": "openbmb/VoxCPM-Demo", "type": "demo"},
    {"space": "hgghfhjfhjguyjf/Voxcpm-Burmese-Tts", "type": "burmese"},
]

VIDEO_EXT = (".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi", ".ts", ".flv")

# Human-like narration rules (added to every language prompt).
_HUMAN = (
    "VOICE STYLE (very important): the result will be read aloud by a speech engine, so it must "
    "sound like a real human storyteller talking to a friend, NOT a robot reading text. "
    "Write the way people actually speak: short natural phrases; put a comma-type mark exactly "
    "where a speaker would take a small breath; end thoughts with a full stop; vary the rhythm "
    "(mix short punchy lines with longer flowing ones); never write a long sentence without a "
    "breathing pause; use everyday spoken wording instead of stiff, literary or textbook wording; "
    "let emotion match the scene (tension, surprise, sadness, relief). {accent} "
    "Do NOT use ellipsis (...), emoji, brackets, asterisks or stage directions. "
)

# Third-person drama-recap style (dialogue is retold as reported speech).
_STORY = (
    "NARRATION STYLE (drama-recap, third person): the source lines are raw speech-to-text of "
    "characters TALKING. Do NOT translate them as direct dialogue and do NOT use quotation marks. "
    "Retell them as a narrator who watched the scene: turn what a character says into reported "
    "speech, for example 'Xiao Li said that ...', 'Xiao Li asked Mother ...', "
    "'Mother firmly replied that ...', 'Boss Wang told Xiao Li ...'. "
    "Work out who is speaking and who is being spoken to from context "
    "(mother, father, boss, neighbour and so on) and use the character names or relationship words. "
    "Turn first-person and second-person speech into third person. "
    "Keep the narrator calm and clear, like someone summarising a TV drama episode to a friend. "
    "Each output line is ONE complete narrated sentence, medium to long length, that flows "
    "naturally like a real story. Link actions with natural connectors (then, so, but, after that). "

    # --- Natural Translation Rule ---
    "NATURAL TRANSLATION (very important): The source language can be English, Chinese, or any "
    "other language. DO NOT translate word-by-word or literally. Translate the MEANING, "
    "INTENTION, and EMOTION, not just the words. Idioms, slang, proverbs, and cultural references "
    "must be converted into their closest natural Burmese equivalent. If there is no direct "
    "equivalent, rewrite the sentence so a native Burmese speaker immediately understands the "
    "situation. The final Burmese must sound like everyday spoken Burmese — smooth, clear, and "
    "meaningful — not like a machine or a textbook. If a translated line sounds confusing, "
    "unnatural, or meaningless, REWRITE it until it sounds natural in Burmese. "

    # --- (5) Inner Thoughts & Motives ---
    "INNER THOUGHTS & MOTIVES: Do not just state what happened. Describe HOW it happened and WHY. "
    "Add the emotion and the tone of the character (e.g., 'angrily shouted', 'whispered with tears', "
    "'smiled coldly'). Make the listener feel the scene like they are watching a movie. "

    # --- (6) Smooth Transitions ---
    "SMOOTH TRANSITIONS: Use varied transition words to connect scenes (e.g., 'at that moment', "
    "'meanwhile', 'suddenly', 'but then'). Avoid repeating the same connector (like 'then' or 'and') "
    "over and over. Make the story flow naturally from one event to the next. "

    # --- (7) Highlight Plot Twists ---
    "HIGHLIGHT PLOT TWISTS: When there is a surprising turn or a twist in the story, emphasize it. "
    "Use dramatic phrasing to make the listener feel the shock (e.g., 'but the truth was...', "
    "'at that moment, a shocking event happened...'). "

    # --- (9) Voice Tone & Pitch ---
    "VOICE TONE & PITCH: Write sentences that naturally convey the character's emotion through "
    "the words. For questions, use phrasing that sounds like a real question. For shock or surprise, "
    "use phrasing that sounds astonished (e.g., 'What? How did you know?'). "

    # --- (11) Sentence Segmentation for TTS ---
    "SENTENCE SEGMENTATION: Keep sentences short and punchy. Do NOT write long, complex sentences. "
    "Break long thoughts into 2 or 3 short sentences so the TTS engine can read them with natural "
    "pauses and intonation. (e.g., 'Mike came home. It was raining. But the door was open.') "

    # --- (14) Numbers & Symbols to Words ---
    "NUMBERS & SYMBOLS: Always write numbers and symbols as spoken words in the target language "
    "(e.g., '1 billion' -> 'တစ်ဘီလီယံ', '50%' -> 'ငါးဆယ်ရာခိုင်နှုန်း', '$100' -> 'ဒေါ်လာတစ်ရာ'). "
    "This helps the TTS engine read them naturally. "

    # --- (16) Relationship Dynamics ---
    "RELATIONSHIP DYNAMICS: Clearly state the relationship between characters when it matters "
    "(e.g., 'Mike's boss, Jake...', 'Mike's ex-girlfriend, Mary...'). This helps the listener "
    "keep track of who is who and how they are connected. "

    # --- (17) Emphasis on Key Points ---
    "EMPHASIS ON KEY POINTS: Use strong, clear language for the most important moments of the story "
    "(e.g., the lottery win, the big fight). Make those moments stand out so the listener remembers them. "

    # --- (18) Context & Background ---
    "CONTEXT & BACKGROUND: If the listener needs background information to understand the scene, "
    "add it naturally (e.g., 'This story started 5 years ago...', 'Their relationship had changed "
    "since that day...'). Do NOT leave the listener confused. "

    "Keep insults, jokes, arguments and emotions of the original, but report them rather than act "
    "them out. Do NOT add hooks, questions to the viewer, commentary or facts that are not in the "
    "source lines. Do NOT address the audience. "
    "REPORTING VERB VARIETY (very important): NEVER repeat the same reporting verb in two "
    "consecutive lines. Rotate through the whole vocabulary of reporting and reaction verbs that "
    "fit the scene — for Burmese prefer ဆိုတယ်, မေးတယ်, ဖြေတယ်, ရှင်းပြတယ်, ပြန်ပြောတယ်, အော်တယ်, "
    "ညည်းတယ်, သတိပေးတယ်, တိုက်တွန်းတယ်, ကတိပေးတယ်, ငြင်းတယ်, ဝန်ခံတယ်, ရယ်တယ်, ခြိမ်းခြောက်တယ်. "
    "The generic Burmese verb 'ပြောတယ်' may appear at most ONCE every four lines. "

    # --- Natural Pronoun Rule ---
    "PRONOUNS: You may use natural Burmese pronouns like သူ (he/she), သူ့ (his/her), သူမ (she), "
    "သူတို့ (they), ကျွန်တော် (I, male), ကျွန်မ (I, female) when it sounds natural in the context "
    "and helps the story flow smoothly. Do NOT overuse them to the point of confusion. Always make "
    "sure the reader clearly understands WHO is being referred to. If there is a risk of confusion "
    "between two characters of the same gender, use their NAME instead of a pronoun. "

    "CHARACTER NAMES: if a character's name is clearly spoken in the source lines, use ONLY that "
    "name for the character and spell it the same way every time. If the name is NOT known, "
    "never write 'the person', 'a person' or 'someone'. "
    "BURMESE UNNAMED CHARACTER (very important): NEVER write လူတစ်ယောက် (a person), "
    "လူတစ်ဦး or တစ်ယောက် for a character. Whenever the character's name is not known, use the "
    "relationship word if the dialogue makes it clear (အမေ, အဖေ, အမျိုးသမီး, သူငယ်ချင်း, သူဌေး, ဧည့်သည်) "
    "and attach the character name before it when the name is known (example pattern: the name + ရဲ့ + "
    "relationship word); if nothing is clear, simply leave the subject out of the sentence "
    "(Burmese does not need a subject word) and write only the action or the speech. "
    "If a second unnamed character appears in the same scene, use a role word for that second "
    "character (ကောင်မလေး young woman or girl, ဦးကြီး older man, အဒေါ်ကြီး older woman, "
    "ကလေးလေး small child) so the two characters are never mixed up. "
    "For other languages, when the name is unknown, call the character by a simple role or "
    "look-based word chosen from age and gender and keep the same word for the same person "
    "(German: der junge Mann, die junge Frau, der alte Mann, die ältere Frau; "
    "Japanese: 青年, 少女, 老人, おばさん). "
    "Speech-to-text often misspells names: if two similar-sounding names appear, treat them as "
    "the same person and use one spelling throughout. "
)

_PROMPT_TAIL = (
    "Rules: do NOT summarize, shorten or skip; do NOT move content between lines; "
    "keep character names consistent across all lines; {script} " + _HUMAN + _STORY +
    "Output exactly one line per input line in the format [number] {name} text "
    "-- same numbers, no extra comments.\n\n"
)

LANG_CONF = {
    "my": {
        "name": "Burmese",
        "label": "မြန်မာ",
        "sample": "စာတန်းထိုး နမူနာ စာသား ဖြစ်ပါတယ်။",
        "edge_voices": [
            ("my-MM-ThihaNeural", "Thiha (male)"),
            ("my-MM-NilarNeural", "Nilar (female)"),
        ],
        "sentence_end": r"(?<=။)",
        "prompt": (
            "You are a Burmese (Myanmar) recap-video narrator. Translate every numbered line "
            "below into natural spoken Burmese narration (storyteller style). "
            + _PROMPT_TAIL.format(
                script="use ONLY Burmese script (Myanmar Unicode), no English/Chinese letters.",
                accent=("ENDINGS (very important): do NOT end every line with ပါတယ်။. Use the polite "
                        "ပါတယ် ONLY when it is really needed (a respectful request, a formal "
                        "statement, or a very polite speaker), at most about one line in five, and "
                        "never in two lines in a row. Normally end lines with plain spoken endings "
                        "such as တယ်။ (ဆိုတယ်၊ မေးတယ်၊ ဖြေတယ်၊ ပြန်ပြောတယ်), ခဲ့တယ်။, လိုက်တယ်။, "
                        "နေတယ်။, တော့တယ်။, ပြီ။, လေ။, ပေါ့။, and vary the ending from line to line. "
                        "Report speech with a VARIETY of verbs and structures, never the same "
                        "verb twice in a row and never 'ပြောတယ်' more than once every four lines. "

                        # --- Natural Translation Rule for Burmese ---
                        "ဘာသာပြန်တဲ့အခါ အင်္ဂလိပ်စာ၊ တရုတ်စာ ဘာပဲဖြစ်ဖြစ် စကားလုံးတစ်လုံးချင်း "
                        "တိုက်ရိုက်ဘာသာမပြန်ပါနဲ့။ မူရင်းစာသားရဲ့ အဓိပ္ပာယ်၊ ခံစားချက်နဲ့ "
                        "ဆိုလိုရင်းကို မြန်မာလို သဘာဝကျကျ ပြန်ပေးပါ။ ဆိုစကား၊ စကားအသုံးအနှုန်းတွေကို "
                        "မြန်မာလူမျိုး နားလည်လွယ်တဲ့ စကားပြောပုံစံအတိုင်း ပြန်ပေးပါ။ "
                        "နားမလည်နိုင်တဲ့ စာသားတွေ၊ ဘာသာပြန်ဆိုးနေတဲ့ စာသားတွေ မထွက်ရဘူး။ "
                        "မြန်မာစကားပြောသလို ချောချောမွေ့မွေ့ ဖတ်ရလွယ်အောင်၊ အဓိပ္ပာယ်ရှိအောင် ရေးပေးပါ။ "
                        # ------------------------------------------

                        "Use spoken Burmese (တယ်၊ တော့၊ ပေါ့၊ လေ) instead of literary endings "
                        "(၏၊ သည်၊ ၍). "
                        "Use '၊' only at natural breath points inside a long sentence and '။' at the end. "
                        "Transliterate spoken names into Burmese script the same way every time; "
                        "when a name is unknown NEVER write လူတစ်ယောက် — "
                        "use the relationship / role word, or leave the subject out."),
                name="Burmese")
        ),
        "font_keys": ("padauk", "myanmar", "pyidaung", "notosansmyanmar", "myanmarpadaung"),
        "clean": r"[^\u1000-\u109f\uaa60-\uaa7f\ua9e0-\ua9ff\s0-9.,!?\"'%\-]",
        "is_mm": True,
    },
    "de": {
        "name": "German",
        "label": "ဂျာမနီ",
        "sample": "Dies ist ein Beispiel-Untertitel.",
        "edge_voices": [
            ("de-DE-KatjaNeural", "Katja (female)"),
            ("de-DE-ConradNeural", "Conrad (male)"),
            ("de-DE-AmalaNeural", "Amala (female)"),
            ("de-DE-BerndNeural", "Bernd (male)"),
        ],
        "sentence_end": r"(?<=[.!?])\s+",
        "prompt": (
            "You are a German (Deutsch) recap-video narrator. Translate every numbered line "
            "below into natural spoken German narration (storyteller style). "
            + _PROMPT_TAIL.format(
                script="use ONLY German (Latin script), no Chinese letters.",
                accent=("Use relaxed spoken German (contractions like 'ist's', 'gibt's' are fine, "
                        "connectors like 'und dann', 'plötzlich', 'also', 'aber') and put a comma "
                        "at every natural breath point. "
                        "Vary the reporting verb: never the same verb twice in a row, and never "
                        "'sagte' more than once every four lines — rotate sagte, fragte, antwortete, "
                        "erklärte, erwiderte, rief, flüsterte, warnte, versprach, leugnete, gab zu, "
                        "lachte, drohte, bat. When the action already shows it, you may drop the "
                        "reporting verb and state the action."),
                name="German")
        ),
        "font_keys": ("dejavusans", "dejavu", "liberation", "notosans", "arial", "freesans"),
        "clean": r"[^A-Za-zÄÖÜäöüß0-9\s.,!?\"'%\-]",
        "is_mm": False,
    },
    "ja": {
        "name": "Japanese",
        "label": "ဂျပန်",
        "sample": "これはサンプルの字幕です。",
        "edge_voices": [
            ("ja-JP-NanamiNeural", "Nanami (female)"),
            ("ja-JP-KeitaNeural", "Keita (male)"),
            ("ja-JP-AoiNeural", "Aoi (female)"),
            ("ja-JP-DaichiNeural", "Daichi (male)"),
            ("ja-JP-MayuNeural", "Mayu (female)"),
            ("ja-JP-NaokiNeural", "Naoki (male)"),
            ("ja-JP-ShioriNeural", "Shiori (female)"),
        ],
        "sentence_end": r"(?<=[。！？!?])",
        "prompt": (
            "You are a Japanese (日本語) recap-video narrator. Translate every numbered line "
            "below into natural spoken Japanese narration (storyteller style). "
            + _PROMPT_TAIL.format(
                script="use ONLY Japanese (Kanji, Hiragana, Katakana), no English/Chinese "
                       "letters except kanji.",
                accent=("Use a warm, natural spoken storyteller tone (mix だ/だった and です/ました "
                        "naturally, with softeners like でも、そして、すると). Use '、' at every "
                        "breath point and '。' to end a sentence. "
                        "Vary the reporting verb: never the same verb twice in a row, and never "
                        "'言った' more than once every four lines — rotate 言った, 尋ねた, 答えた, "
                        "説明した, 返した, 叫んだ, 囁いた, 警告した, 約束した, 否定した, 認めた, "
                        "笑った, 脅した, 頼んだ. When the action already shows it, you may drop "
                        "the reporting verb and state the action."),
                name="Japanese")
        ),
        "font_keys": ("notosanscjk", "notosansjp", "notosansjapanese", "takao", "ipag",
                      "vl-gothic", "msgothic", "cjk"),
        "clean": r"[^\u3000-\u303f\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff\uff00-\uffef0-9\s.,!?\"'%\-]",
        "is_mm": False,
    },
}

# Burmese name aliases are accepted on the command line too
LANG_ALIASES = {c["label"]: k for k, c in LANG_CONF.items()}
LANG_ALIASES.update({c["name"].lower(): k for k, c in LANG_CONF.items()})


def resolve_lang(value):
    v = (value or "my").strip()
    if v in LANG_CONF:
        return v
    if v in LANG_ALIASES:
        return LANG_ALIASES[v]
    if v.lower() in LANG_ALIASES:
        return LANG_ALIASES[v.lower()]
    raise ValueError(f"မသိတဲ့ ဘာသာစကား: {value} (my / de / ja)")


@dataclass
class Settings:
    # TTS
    lang: str = "my"
    engine: str = "edge"          # edge | vox
    voice: str = ""               # empty -> first voice of language
    rate: int = 45                # Edge speed %
    tts_workers: int = 8
    vox_retry: int = 4
    vox_ref: str = ""             # reference audio path for VoxCPM
    # Video
    height: int = 720
    fps: int = 24
    preset: str = "ultrafast"
    max_speed: float = 1.25
    crop: bool = True             # auto-remove black bars
    flip: bool = True             # mirror horizontally
    zoom: float = 1.0
    # Subtitles
    sub: bool = True
    sub_size: float = 7.0         # % of width
    sub_pos: int = 90             # 0 = top, 100 = bottom
    sub_alpha: int = 204
    sub_sweep: bool = True        # music-style light sweep across the subtitle
    font: str = ""                # explicit font path
    # Neon
    neon: bool = True
    neon_th: int = 3
    neon_speed: float = 0.15      # laps per second (0.15 -> ~6.7s per lap)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in names})

    def resolved_voice(self):
        if self.voice:
            return self.voice
        return LANG_CONF[self.lang]["edge_voices"][0][0]

    def rate_str(self):
        return f"{int(self.rate):+d}%"

# ======================================================================
# utils
# ======================================================================
"""Small shared helpers."""
import re
import subprocess
import sys
import time
import wave


class Cancelled(Exception):
    pass


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


def fmt_t(sec):
    sec = float(sec)
    return f"{sec:.1f}s" if sec < 60 else f"{int(sec // 60)}m {sec % 60:.0f}s"


def clean_text(text, lang="my"):
    # keep only characters of the target language (no pronoun substitution any more)
    return re.sub(LANG_CONF[lang]["clean"], "", text).strip()


def log(msg):
    print(msg, file=sys.stderr, flush=True)


class Progress:
    """Single-line terminal progress bar. Call p(frac, label)."""

    def __init__(self, lo=0.0, hi=1.0, parent=None, quiet=False):
        self.lo, self.hi, self.parent, self.quiet = lo, hi, parent, quiet
        self.t0 = time.time()
        self._label = ""
        self._last = 0.0

    def sub(self, lo, hi):
        return Progress(lo, hi, parent=self, quiet=self.quiet)

    def __call__(self, frac, label=None):
        frac = max(0.0, min(float(frac), 1.0))
        if label is not None:
            self._label = label
        if self.parent is not None:
            self.parent(self.lo + (self.hi - self.lo) * frac, self._label)
            return
        now = time.time()
        if self.quiet or (now - self._last < 0.2 and frac < 1.0):
            return
        self._last = now
        width = 24
        fill = int(width * frac)
        bar = "█" * fill + "░" * (width - fill)
        line = f"\r[{bar}] {frac * 100:5.1f}%  {self._label}  ({fmt_t(now - self.t0)})"
        sys.stderr.write(line[:160].ljust(100))
        sys.stderr.flush()

    def done(self):
        if self.parent is None and not self.quiet:
            sys.stderr.write("\n")
            sys.stderr.flush()


def timing_table(d):
    w = max(len(k) for k in d) if d else 10
    return "\n".join(f"  {k.ljust(w)}  {fmt_t(v)}" for k, v in d.items())

# ======================================================================
# fonts
# ======================================================================
"""Font discovery, text wrapping and subtitle splitting (Burmese / German / Japanese)."""
import glob
import os
import re

from PIL import ImageFont, features


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def _prefer_regular(paths):
    def score(p):
        b = os.path.basename(p).lower()
        s = 0
        if "regular" in b:
            s -= 2
        if any(x in b for x in ("bold", "italic", "light", "thin", "black", "medium", "mono", "serif")):
            s += 2
        return (s, len(b))
    return sorted(paths, key=score)


def _globs(base):
    out = []
    for ext in ("ttf", "otf", "ttc"):
        out += glob.glob(os.path.join(base, f"*.{ext}"))
        out += glob.glob(os.path.join(base, "fonts", f"*.{ext}"))
    return out


def find_font(lang="my", explicit=""):
    """Order: --font path -> RECAP_FONT env -> repo fonts/ -> system fonts."""
    conf = LANG_CONF[lang]

    if explicit and os.path.exists(explicit):
        return explicit
    env_font = os.environ.get("RECAP_FONT", "")
    if conf["is_mm"] and env_font and os.path.exists(env_font):
        return env_font
    repo = _globs(REPO_ROOT) + _globs(os.getcwd())

    if conf["is_mm"]:
        for name in ("MyanmarPadaung.ttf", "fonts/MyanmarPadaung.ttf"):
            p = os.path.join(REPO_ROOT, name)
            if os.path.exists(p):
                return p
        for f in repo:
            if any(k in os.path.basename(f).lower() for k in conf["font_keys"]):
                return f
        if repo:
            return sorted(repo)[0]
    else:
        for key in conf["font_keys"]:
            for f in _prefer_regular(repo):
                if key in os.path.basename(f).lower():
                    return f

    sysf = []
    for ext in ("ttf", "otf", "ttc"):
        sysf += glob.glob(f"/usr/share/fonts/**/*.{ext}", recursive=True)
        sysf += glob.glob(f"/Library/Fonts/**/*.{ext}", recursive=True)
        sysf += glob.glob(f"C:/Windows/Fonts/*.{ext}")
    sysf = _prefer_regular(sysf)
    for key in conf["font_keys"]:
        for f in sysf:
            if key in os.path.basename(f).lower():
                return f
    if conf["is_mm"] and repo:
        return repo[0]
    return None


def has_raqm():
    try:
        return bool(features.check("raqm"))
    except Exception:
        return False


_FONT_CACHE = {}


def get_font(path, size, layout="raqm"):
    key = (path, size, layout)
    if key not in _FONT_CACHE:
        try:
            use_raqm = layout == "raqm" and has_raqm()
            eng = ImageFont.Layout.RAQM if use_raqm else ImageFont.Layout.BASIC
            _FONT_CACHE[key] = ImageFont.truetype(path, size, layout_engine=eng)
        except Exception:
            try:
                _FONT_CACHE[key] = ImageFont.truetype(path, size)
            except Exception:
                _FONT_CACHE[key] = ImageFont.load_default()
    return _FONT_CACHE[key]


# ---- Burmese syllable splitting -------------------------------------------
def _is_start(c):
    return (0x1000 <= c <= 0x1021) or (0x1023 <= c <= 0x1027) or c in (0x1029, 0x102A, 0x103F)


def split_syll(word):
    idx = [0]
    n = len(word)
    for i in range(1, n):
        c = ord(word[i])
        if not _is_start(c):
            continue
        prev = ord(word[i - 1])
        nxt = ord(word[i + 1]) if i + 1 < n else 0
        if prev == 0x1039 or nxt in (0x103A, 0x1039):
            continue
        idx.append(i)
    idx.append(n)
    return [word[idx[k]:idx[k + 1]] for k in range(len(idx) - 1)]


def wrap_text(text, font, maxw, is_mm=False):
    lines, cur = [], ""
    if not is_mm and " " not in text and any(0x3040 <= ord(c) <= 0x9FFF for c in text):
        for ch in text:
            if not cur or font.getlength(cur + ch) <= maxw:
                cur += ch
            else:
                lines.append(cur)
                cur = ch
        if cur:
            lines.append(cur)
        return lines

    for word in text.split():
        pieces = [word] if font.getlength(word) <= maxw else (split_syll(word) if is_mm else list(word))
        for pi, p in enumerate(pieces):
            sep = " " if (pi == 0 and cur) else ""
            cand = cur + sep + p
            if not cur or font.getlength(cand) <= maxw:
                cur = cand
            else:
                lines.append(cur)
                cur = p
    if cur:
        lines.append(cur)
    return lines


def split_for_sub(text, mc=36, lang="my"):
    conf = LANG_CONF[lang]
    is_mm = conf["is_mm"]
    sents = [s.strip() for s in re.split(conf["sentence_end"], text) if s.strip()]
    parts = []
    for s in sents:
        if len(s) <= mc:
            parts.append(s)
            continue
        if lang == "ja":
            cur = ""
            for ch in s:
                if not cur or len(cur + ch) <= mc:
                    cur += ch
                else:
                    parts.append(cur)
                    cur = ch
            if cur:
                parts.append(cur)
            continue
        cur = ""
        for word in s.split():
            pieces = [word] if len(word) <= mc else (split_syll(word) if is_mm else list(word))
            for pi, p in enumerate(pieces):
                sep = " " if (pi == 0 and cur) else ""
                if not cur or len(cur + sep + p) <= mc:
                    cur = cur + sep + p
                else:
                    parts.append(cur)
                    cur = p
        if cur:
            parts.append(cur)
    return parts or [text]

# ======================================================================
# overlay
# ======================================================================
"""Subtitle overlay (PNG-style, drawn with Pillow), music sweep, neon border and zoom/flip."""
import cv2
import numpy as np
from PIL import Image, ImageDraw


SWEEP_TRAIL = np.array([255, 235, 60], np.float32)
SWEEP_BAND = np.array([255, 255, 235], np.float32)


def build_overlay(text, W, H, font_path, fs, pos_y, alpha=204, color=(255, 255, 255),
                  layout="raqm", is_mm=True):
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    ink_img = Image.new("L", (W, H), 0)
    di = ImageDraw.Draw(ink_img)

    pad_x, pad_y = int(fs * 0.6), int(fs * 0.5)
    lh_box = int(fs * 1.4)
    box_h = SUB_LINES * lh_box + 2 * pad_y
    box_h = min(box_h, H)
    # FIX: box_y was used before it was defined (UnboundLocalError -> empty preview)
    max_y = max(0, H - box_h)
    box_y = max(0, min(int(pos_y / 100 * max_y), max_y))

    cur_fs = fs
    font = get_font(font_path, cur_fs, layout)
    lines = wrap_text(text, font, W - 2 * pad_x, is_mm)
    while len(lines) > SUB_LINES and cur_fs > 12:
        cur_fs = max(12, int(cur_fs * 0.92))
        font = get_font(font_path, cur_fs, layout)
        lines = wrap_text(text, font, W - 2 * pad_x, is_mm)
    if not lines:
        return None

    d.rounded_rectangle([0, box_y, W, box_y + box_h], radius=int(fs * 0.66),
                        fill=(0, 0, 0, alpha))
    ink = (int(color[0]), int(color[1]), int(color[2]), 255)
    sw = max(1, round(cur_fs / 15))
    lh = int(cur_fs * 1.4)
    block_h = len(lines) * lh
    ty = box_y + (box_h - block_h) // 2
    for ln in lines:
        lx = (W - font.getlength(ln)) / 2
        d.text((lx, ty), ln, font=font, fill=(0, 0, 0, 255),
               stroke_width=sw, stroke_fill=(0, 0, 0, 255))
        d.text((lx, ty), ln, font=font, fill=ink)
        di.text((lx, ty), ln, font=font, fill=255)
        ty += lh

    arr = np.array(img)
    rows = np.where(arr[:, :, 3].any(axis=1))[0]
    if len(rows) == 0:
        return None
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    crop = arr[y0:y1]
    a = crop[:, :, 3:4].astype(np.float32) / 255.0
    bgr = crop[:, :, [2, 1, 0]].astype(np.float32)
    tm = (np.array(ink_img)[y0:y1].astype(np.float32) / 255.0)[:, :, None]
    return {"y0": y0, "y1": y1, "inv": 1.0 - a, "pre": bgr * a,
            "tm": tm, "xs": np.arange(W, dtype=np.float32)[None, :, None]}


def blend_overlay(fr, ov, p=None):
    reg = fr[ov["y0"]:ov["y1"]].astype(np.float32) * ov["inv"] + ov["pre"]
    if p is not None:
        W = reg.shape[1]
        p = max(0.0, min(float(p), 1.0))
        c = (-0.12 + 1.24 * p) * W
        bw = 0.07 * W
        x = ov["xs"]
        band = np.exp(-(((x - c) / bw) ** 2))
        trail = (x < c).astype(np.float32) * 0.55
        tm = ov["tm"]
        k_trail = np.clip(trail, 0.0, 1.0) * tm
        reg = reg * (1.0 - k_trail) + SWEEP_TRAIL * k_trail
        k_band = np.clip(band * 0.95, 0.0, 1.0) * tm
        reg = reg * (1.0 - k_band) + SWEEP_BAND * k_band
        reg += band * (1.0 - ov["inv"]) * 22.0
    fr[ov["y0"]:ov["y1"]] = np.clip(reg, 0, 255).astype(np.uint8)


class Neon:
    BASE = np.array([12, 8, 10], np.float32)
    CYAN = np.array([255, 255, 0], np.float32)
    MAGENTA = np.array([110, 30, 255], np.float32)
    IDLE = 0.14
    NBINS = 2048

    def __init__(self, W, H, th=3, speed=0.15, tail=0.5):
        self.W, self.H, self.speed, self.tail = W, H, speed, tail
        th = max(2, min(th, min(W, H) // 4))
        self.P = 2 * (W + H)
        yy, xx = np.mgrid[0:H, 0:W]
        ring = (xx < th) | (xx >= W - th) | (yy < th) | (yy >= H - th)
        ys, xs = np.nonzero(ring)
        dt, db, dl, dr = ys, H - 1 - ys, xs, W - 1 - xs
        m = np.minimum.reduce([dt, db, dl, dr])
        S = np.where(m == dt, xs,
            np.where(m == dr, W + ys,
            np.where(m == db, 2 * W + H + (W - xs),
                     2 * W + 2 * H - ys))).astype(np.float64)
        self.ys, self.xs = ys, xs
        Sf = ((S % self.P) / float(self.P)).astype(np.float32)
        self.L = np.arange(self.NBINS, dtype=np.float32) / float(self.NBINS)
        self.idx = np.clip((Sf * self.NBINS).astype(np.int32), 0, self.NBINS - 1)

    def _layer(self, h, color):
        dist = (h - self.L) % 1.0
        a = np.clip(1.0 - dist / self.tail, 0, 1)
        glow = self.IDLE + (1.0 - self.IDLE) * (a ** 0.55)
        core = np.clip(1.0 - dist / 0.06, 0, 1) ** 2
        return color * glow[:, None] + 255.0 * 0.9 * core[:, None]

    def apply(self, fr, t):
        head = (t * self.speed) % 1.0
        c = self.BASE + self._layer(head, self.CYAN) + self._layer(head + 0.5, self.MAGENTA)
        strip = np.clip(c, 0, 255).astype(np.uint8)
        fr[self.ys, self.xs] = strip[self.idx]


def zoom_flip(fr, zoom, flip):
    zoomed = zoom > 1.001
    if zoomed:
        H, W = fr.shape[:2]
        cw, ch = int(W / zoom), int(H / zoom)
        x0, y0 = (W - cw) // 2, (H - ch) // 2
        fr = cv2.resize(fr[y0:y0 + ch, x0:x0 + cw], (W, H), interpolation=cv2.INTER_LINEAR)
    if flip:
        return cv2.flip(fr, 1)
    return fr if zoomed else fr.copy()

# ======================================================================
# download
# ======================================================================
import glob
import hashlib
import os
import re
import shutil
import tempfile
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse


def _fix_dropbox(url):
    u = urlparse(url)
    q = parse_qs(u.query)
    q["dl"] = ["1"]
    return urlunparse(u._replace(query=urlencode(q, doseq=True)))


def _dl_direct(url, out, progress=None):
    import requests
    with requests.get(url, stream=True, timeout=30, allow_redirects=True,
                      headers={"User-Agent": "Mozilla/5.0"}) as r:
        r.raise_for_status()
        if "text/html" in r.headers.get("content-type", "").lower():
            raise RuntimeError("Link က video ဖိုင် မဟုတ်ဘဲ web page ဖြစ်နေတယ်")
        total = int(r.headers.get("content-length") or 0)
        got = 0
        with open(out, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                got += len(chunk)
                if progress and total:
                    progress(got / total)
    if not os.path.exists(out) or os.path.getsize(out) < 1024:
        raise RuntimeError("ဖိုင် မရပါ")
    return out


def _dl_gdrive(url, out):
    import gdown
    gdown.download(url, out, quiet=True, fuzzy=True)
    if not os.path.exists(out) or os.path.getsize(out) < 1024:
        raise RuntimeError("Google Drive ဖိုင် မရပါ (Share ကို 'Anyone with the link' ထားပါ)")
    return out


def _dl_ytdlp(url, out_noext, height, progress=None, cookies_file=None):
    import yt_dlp

    def hook(d):
        if progress and d.get("status") == "downloading":
            tot = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            if tot:
                progress(d.get("downloaded_bytes", 0) / tot)

    opts = {
        "format": (f"bv*[height<={height}][vcodec^=avc1]+ba[ext=m4a]/"
                   f"b[height<={height}][vcodec^=avc1]/"
                   f"bv*[height<={height}]+ba/b[height<={height}]/bv*+ba/b"),
        "outtmpl": out_noext + ".%(ext)s",
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "retries": 5,
        "fragment_retries": 5,
        "concurrent_fragment_downloads": 4,
        "progress_hooks": [hook],
    }
    if cookies_file and os.path.exists(cookies_file):
        opts["cookiefile"] = cookies_file
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    files = [p for p in glob.glob(out_noext + ".*")
             if not p.endswith((".part", ".ytdl", "_cookies.txt"))]
    if not files:
        raise RuntimeError("yt-dlp ဖိုင် မထွက်ပါ")
    return max(files, key=os.path.getsize)


def ensure_mp4(raw, final, height):
    try:
        codec = sh(["ffprobe", "-v", "error", "-select_streams", "v:0",
                    "-show_entries", "stream=codec_name", "-of", "csv=p=0", raw]).strip()
    except Exception:
        codec = ""
    if not codec:
        raise RuntimeError("Video stream မတွေ့ပါ (ဖိုင်က video မဟုတ်နိုင်)")
    if codec == "h264":
        try:
            sh(["ffmpeg", "-y", "-i", raw, "-c", "copy", "-movflags", "+faststart", final])
            return final
        except Exception:
            pass
    sh(["ffmpeg", "-y", "-i", raw, "-vf", f"scale=-2:'min(ih,{int(height)})'",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "26", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", final])
    return final


def fetch_video(source, final, height=720, progress=None, cookies_file=None):
    source = source.strip()

    if not re.match(r"^https?://", source, re.I):
        if not os.path.isfile(source):
            raise FileNotFoundError(f"ဖိုင် မတွေ့ပါ: {source}")
        ensure_mp4(source, final, height)
        return final

    work = tempfile.mkdtemp(prefix="dl_")
    try:
        u = urlparse(source)
        host, path_l = u.netloc.lower(), u.path.lower()
        raw, errs = None, []
        if "drive.google.com" in host or "docs.google.com" in host:
            raw = _dl_gdrive(source, os.path.join(work, "gd.bin"))
        else:
            url = source
            if "dropbox.com" in host:
                url = _fix_dropbox(url)
                methods = ["direct", "ytdlp"]
            elif path_l.endswith(VIDEO_EXT):
                methods = ["direct", "ytdlp"]
            else:
                methods = ["ytdlp", "direct"]
            for m in methods:
                try:
                    if m == "direct":
                        raw = _dl_direct(url, os.path.join(work, "direct.bin"), progress)
                    else:
                        raw = _dl_ytdlp(url, os.path.join(work, "yt"), height, progress,
                                        cookies_file)
                    break
                except Exception as ex:
                    errs.append(f"{m}: {str(ex)[:200]}")
                    raw = None
            if not raw:
                raise RuntimeError("Link ကနေ မဆွဲနိုင်ပါ\n" + "\n".join(errs))
        if progress:
            progress(1.0)
        ensure_mp4(raw, final, height)
        return final
    finally:
        shutil.rmtree(work, ignore_errors=True)


def source_key(source):
    return hashlib.md5(source.encode()).hexdigest()[:8]

# ======================================================================
# transcribe
# ======================================================================
import os
import wave

import numpy as np


_MODELS = {}


def get_model(size):
    if size not in _MODELS:
        from faster_whisper import WhisperModel
        _MODELS[size] = WhisperModel(size, device="cpu", compute_type="int8",
                                     cpu_threads=os.cpu_count() or 4)
    return _MODELS[size]


def _run(wm, audio, lang, fast, vdur, progress):
    raw = []
    if fast:
        try:
            from faster_whisper import BatchedInferencePipeline
            pipe = BatchedInferencePipeline(model=wm)
            segs, info = pipe.transcribe(audio, language=lang, beam_size=1,
                                         batch_size=8, without_timestamps=False)
            for s in segs:
                if s.text.strip():
                    raw.append((s.start, s.end, s.text.strip()))
                progress(min(s.end / vdur, 1.0))
            return raw, info
        except Exception as ex:
            log(f"\n(batch mode မရလို့ ပုံမှန်နည်း သုံးမယ်: {str(ex)[:80]})")
            raw = []
    segs, info = wm.transcribe(audio, language=lang, vad_filter=True, beam_size=1,
                               temperature=0.0, condition_on_previous_text=False)
    for s in segs:
        if s.text.strip():
            raw.append((s.start, s.end, s.text.strip()))
        progress(min(s.end / vdur, 1.0))
    return raw, info


def merge_segments(raw):
    units = []
    for s, e, t in raw:
        if units and (units[-1][1] - units[-1][0]) < TARGET and (e - units[-1][0]) <= MAXD:
            units[-1] = (units[-1][0], e, units[-1][2] + " " + t)
        else:
            units.append((s, e, t))
    return units


def transcribe_video(video, workdir, model_size="base", src_lang="auto", fast=True,
                     max_min=0, progress=None):
    progress = progress or (lambda f, l=None: None)
    vdur = probe_duration(video)
    if max_min and vdur > max_min * 60:
        raise RuntimeError(f"Video က {vdur / 60:.1f} မိနစ်ရှိတယ်။ --max-min {max_min} ထက် ကျော်နေတယ် "
                           "(--max-min တိုးပါ၊ 0 = ကန့်သတ်မရှိ)")

    progress(0.0, "Audio ထုတ်နေတယ်")
    wav = os.path.join(workdir, "audio16k.wav")
    # Error ဖြစ်နေတဲ့ အသံကို ဖျက်ပြီး (Discard) အသစ်ပြန် Encode လုပ်တဲ့ အတွက်
    # -err_detect ignore_err နဲ့ -fflags +discardcorrupt ကို ထည့်ထားပါတယ်
    sh(["ffmpeg", "-y", "-err_detect", "ignore_err", "-fflags", "+discardcorrupt",
        "-i", video, "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav])
    with wave.open(wav) as wf:
        audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0

    progress(0.08, "Whisper model ဖွင့်နေတယ် (ပထမအကြိမ် download ကြာမယ်)")
    wm = get_model(model_size)

    lang = None if src_lang == "auto" else src_lang
    progress(0.15, "Whisper စာသားပြောင်းနေတယ်")
    raw, info = _run(wm, audio, lang, fast, vdur,
                     lambda f: progress(0.15 + 0.85 * f, "Whisper စာသားပြောင်းနေတယ်"))
    units = merge_segments(raw)
    if not units:
        raise RuntimeError("စကားပြောသံ မတွေ့ပါ")
    progress(1.0, "ပြီးပြီ")
    text = (f"ဘာသာစကား: {info.language} ({info.language_probability:.0%}) · "
            f"အပိုင်း {len(units)} ခု")
    return ([{"start": u[0], "end": u[1], "src": u[2], "my": ""} for u in units], vdur, text)

# ======================================================================
# translate
# ======================================================================
import re


CHUNK = 50   # lines per Gemini prompt part


def build_prompt(units, lang="my"):
    lines = "\n".join(f"[{i}] {u['src']}" for i, u in enumerate(units))
    return LANG_CONF[lang]["prompt"] + lines


def build_prompt_chunks(units, lang="my", size=CHUNK):
    head = LANG_CONF[lang]["prompt"]
    total = (len(units) + size - 1) // size
    out = []
    for k in range(total):
        lo, hi = k * size, min((k + 1) * size, len(units))
        lines = "\n".join(f"[{i}] {units[i]['src']}" for i in range(lo, hi))
        note = (f"(PART {k + 1} of {total} of the same episode. Only lines [{lo}] to [{hi - 1}]. "
                f"Keep the SAME numbers. Keep character names and role words consistent "
                f"with the other parts. Never use pronouns, always repeat the character name. "
                f"For Burmese, when a name is unknown never write လူတစ်ယောက် — "
                f"use a role word or leave the subject out. "
                f"For Burmese, do not end every line with ပါတယ်; use it only when needed. "
                f"For Burmese, vary the reporting verb — 'ပြောတယ်' at most once every four lines; "
                f"prefer ဆိုတယ်, မေးတယ်, ဖြေတယ်, ရှင်းပြတယ်, ပြန်ပြောတယ်, အော်တယ် and rotate.)\n\n")
        out.append({"lo": lo, "hi": hi, "text": head + note + lines})
    return out


def parse_numbered(text, n):
    got = {}
    for line in text.splitlines():
        line = line.strip().strip("*").strip()
        m = re.match(r"^[\[\(#]?\s*(\d+)\s*[\]\)\.:\-–]?\s*(.+)$", line)
        if m and int(m.group(1)) < n:
            got[int(m.group(1))] = m.group(2).strip().strip("*").strip()
    if not got:
        plain = [l.strip() for l in text.splitlines() if l.strip()]
        if len(plain) == n:
            got = {i: plain[i] for i in range(n)}
    return got


def check_part(text, lo, hi):
    got = {i: t for i, t in parse_numbered(text or "", hi).items() if i >= lo}
    missing = [i for i in range(lo, hi) if i not in got]
    return got, missing

# ======================================================================
# tts
# ======================================================================
import asyncio
import shutil
import threading


def tts_edge(text, voice, rate, mp3, proxy=None):
    import edge_tts

    async def run():
        kw = {"proxy": proxy} if proxy else {}
        await edge_tts.Communicate(text, voice, rate=rate, **kw).save(mp3)

    asyncio.run(run())


def _gclient(space, token):
    from gradio_client import Client
    if token:
        try:
            return Client(space, token=token)
        except TypeError:
            return Client(space, hf_token=token)
    return Client(space)


class VoxEngine:
    def __init__(self, ref_path=None, token=None):
        self.ref = ref_path or None
        self.token = token
        self.clients = {}
        self.fails = {i: 0 for i in range(len(VOX_SPACES))}
        self.dead = set()
        self.active = None
        self.last_err = ""
        self.lock = threading.Lock()

    def reset(self):
        with self.lock:
            self.dead.clear()
            self.fails = {i: 0 for i in range(len(VOX_SPACES))}
            self.clients = {}

    def synth(self, text, out_wav):
        from gradio_client import handle_file
        order = [i for i in range(len(VOX_SPACES)) if i not in self.dead]
        if self.active in order:
            order.remove(self.active)
            order.insert(0, self.active)
        for i in order:
            sp = VOX_SPACES[i]
            if sp["type"] == "burmese" and not self.ref:
                self.dead.add(i)
                self.last_err = f"{sp['space']}: reference audio လိုအပ်တယ် (--vox-ref)"
                continue
            try:
                cl = self.clients.get(i)
                if cl is None:
                    cl = _gclient(sp["space"], self.token)
                    self.clients[i] = cl
                rf = handle_file(self.ref) if self.ref else None
                if sp["type"] == "demo":
                    res = cl.predict(
                        text_input=text,
                        control_instruction="A warm young woman, calm and expressive",
                        reference_wav_path_input=rf,
                        use_prompt_text=False, prompt_text_input="",
                        cfg_value_input=2.0, do_normalize=True, denoise=False,
                        api_name="/generate")
                else:
                    res = cl.predict(
                        target_text=text, ref_audio=rf,
                        ref_text="မြန်မာ အသံနမူနာ", cfg_value=2.0,
                        inference_timesteps=10, api_name="/tts")
                p = res[0] if isinstance(res, (tuple, list)) else res
                shutil.copy(p, out_wav)
                self.active = i
                return sp["space"]
            except Exception as ex:
                self.last_err = f"{sp['space']}: {str(ex)[:120]}"
                self.fails[i] += 1
                if self.fails[i] >= 2:
                    self.dead.add(i)
        return None

# ======================================================================
# render
# ======================================================================
import os
import queue
import re
import subprocess
import threading

import cv2


def detect_crop(src):
    r = subprocess.run(
        ["ffmpeg", "-ss", "1", "-i", src, "-t", "8", "-vf", "cropdetect=24:2:0",
         "-f", "null", "-"], capture_output=True, text=True)
    m = re.findall(r"crop=(\d+):(\d+):(\d+):(\d+)", r.stderr)
    if not m:
        return None
    return tuple(int(x) for x in m[-1])


def prepare_video(src, dst, target_h, crop=True):
    target_h = int(target_h) // 2 * 2
    vf = None
    if crop:
        c = detect_crop(src)
        cap = cv2.VideoCapture(src)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()
        if c and c[1] < h * 0.95 and c[0] > 0 and c[1] > 0:
            vf = f"crop={c[0]}:{c[1]}:{c[2]}:{c[3]},scale=-2:{target_h}"
    if vf is None:
        vf = f"scale=-2:{target_h}"
    r = subprocess.run(["ffmpeg", "-y", "-i", src, "-vf", vf, "-an", "-c:v", "libx264",
                        "-preset", "ultrafast", "-crf", "24", "-pix_fmt", "yuv420p", dst],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("video prepare မအောင်မြင်ပါ\n" + r.stderr[-800:])
    cap = cv2.VideoCapture(dst)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return W, H


def render_final(src, audio, out, plan, total, s, subs, font_path, is_mm=True,
                 progress=None):
    tmp_vid = out + "_prep.mp4"
    W, H = prepare_video(src, tmp_vid, s.height, s.crop)

    cap = cv2.VideoCapture(tmp_vid)
    sfps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stotal = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 10 ** 9
    fps_out = int(s.fps)
    nfr = max(1, int(round(total * fps_out)))
    fs = max(16, int(W * s.sub_size / 100))
    neon = Neon(W, H, s.neon_th, s.neon_speed) if s.neon else None
    style = {"alpha": int(s.sub_alpha), "color": (255, 255, 255), "layout": "raqm"}

    q = queue.Queue(maxsize=32)
    stop = threading.Event()

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
                        if f.shape[1] != W or f.shape[0] != H:
                            f = cv2.resize(f, (W, H))
                        last = f
                if last is None:
                    return
                put(last)
        finally:
            put(None)

    log_path = out + ".log"
    log_fh = open(log_path, "wb")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{W}x{H}", "-r", str(fps_out), "-i", "-", "-i", audio,
           "-map", "0:v", "-map", "1:a:0",
           "-c:v", "libx264", "-preset", s.preset, "-crf", "27", "-pix_fmt", "yuv420p",
           "-threads", "0", "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart",
           "-t", f"{total:.3f}", out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log_fh)
    th = threading.Thread(target=producer, daemon=True)
    th.start()

    k, cur_k, cur_ov, j = 0, -1, None, 0
    try:
        while True:
            fr = q.get()
            if fr is None:
                break
            t = j / fps_out
            fr = zoom_flip(fr, s.zoom, s.flip)
            if neon:
                neon.apply(fr, t)
            if subs and font_path:
                while k < len(subs) and subs[k][1] < t:
                    k += 1
                if k < len(subs) and subs[k][0] <= t:
                    if cur_k != k:
                        cur_ov = build_overlay(subs[k][2], W, H, font_path, fs, s.sub_pos,
                                               is_mm=is_mm, **style)
                        cur_k = k
                    if cur_ov:
                        pr = None
                        if s.sub_sweep:
                            pr = (t - subs[k][0]) / max(subs[k][1] - subs[k][0], 1e-3)
                        blend_overlay(fr, cur_ov, pr)
            proc.stdin.write(fr.data)
            j += 1
            if progress and j % 30 == 0:
                progress(min(j / nfr, 1.0), "Video render")
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
        try:
            proc.wait()
        except Exception:
            pass
        log_fh.close()
        if os.path.exists(tmp_vid):
            os.remove(tmp_vid)

    if proc.returncode != 0:
        with open(log_path, "rb") as f:
            tail = f.read()[-800:].decode("utf-8", "ignore")
        raise RuntimeError("render: ffmpeg fail\n" + tail)
    try:
        os.remove(log_path)
    except OSError:
        pass
    if progress:
        progress(1.0, "Video render")
    return out


# ---- preview (5-second scrubbable clip) ----------------------------------
_NEON_CACHE = {}


def _neon_for(W, H, th, speed):
    key = (W, H, th, speed)
    if key not in _NEON_CACHE:
        if len(_NEON_CACHE) > 4:
            _NEON_CACHE.clear()
        _NEON_CACHE[key] = Neon(W, H, th, speed)
    return _NEON_CACHE[key]


def _preview_clip(src, clip_path, height, crop):
    """Cut a short (~5.5s) prepared (cropped+scaled) clip once and cache it on disk."""
    if os.path.exists(clip_path) and os.path.getsize(clip_path) > 1000:
        return clip_path
    raw = clip_path + "_raw.mp4"
    tmp = clip_path + "_tmp.mp4"
    try:
        for ss in (["-ss", "1"], []):
            subprocess.run(["ffmpeg", "-y", *ss, "-i", src, "-t", f"{PREVIEW_SECS + 0.5:.1f}",
                            "-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "26",
                            "-pix_fmt", "yuv420p", raw],
                           capture_output=True, text=True)
            if os.path.exists(raw) and os.path.getsize(raw) > 1000:
                break
        if not os.path.exists(raw) or os.path.getsize(raw) <= 1000:
            raise RuntimeError("preview clip မရပါ")
        prepare_video(raw, tmp, height, crop)
        os.replace(tmp, clip_path)
    finally:
        for f in (raw, tmp):
            if os.path.exists(f):
                try:
                    os.remove(f)
                except OSError:
                    pass
    return clip_path


def _preview_frame_at(src, cache_dir, height, crop, t):
    key = f"pv_clip_{os.path.getsize(src)}_{int(height)}_{int(bool(crop))}.mp4"
    clip = _preview_clip(src, os.path.join(cache_dir, key), height, crop)
    cap = cv2.VideoCapture(clip)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    idx = max(0, min(int(round(float(t) * fps)), n - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, fr = cap.read()
    if not ok:
        # fallback: sequential read up to idx (or last readable frame)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        fr = None
        for _ in range(idx + 1):
            ok2, f2 = cap.read()
            if not ok2:
                break
            fr = f2
    cap.release()
    if fr is None:
        raise RuntimeError("preview frame မရပါ")
    return fr


def make_preview_frame(src, out_png, s, text, font_path, is_mm=True, t=0.0):
    cache_dir = os.path.dirname(os.path.abspath(out_png))
    t = max(0.0, min(float(t), PREVIEW_SECS))
    base = _preview_frame_at(src, cache_dir, s.height, s.crop, t)
    H, W = base.shape[:2]
    fr = zoom_flip(base.copy(), s.zoom, s.flip)
    if s.neon:
        _neon_for(W, H, s.neon_th, s.neon_speed).apply(fr, t)
    if font_path and s.sub:
        ov = build_overlay(text, W, H, font_path, max(16, int(W * s.sub_size / 100)),
                           s.sub_pos, alpha=int(s.sub_alpha), is_mm=is_mm)
        if ov:
            blend_overlay(fr, ov, (t / PREVIEW_SECS) if s.sub_sweep else None)
    cv2.imwrite(out_png, fr)
    return out_png

# ======================================================================
# project
# ======================================================================
import json
import os


class Project:
    FILE = "project.json"

    def __init__(self, path):
        self.path = os.path.abspath(path)
        self.data = {"source": "", "vdur": 0.0, "units": [], "timings": {}, "settings": {}}

    # -- paths --
    @property
    def video(self):
        return os.path.join(self.path, "src.mp4")

    @property
    def prompt_file(self):
        return os.path.join(self.path, "prompt.txt")

    @property
    def translated_file(self):
        return os.path.join(self.path, "translated.txt")

    @property
    def final(self):
        return os.path.join(self.path, "final.mp4")

    @property
    def units(self):
        return self.data["units"]

    @property
    def settings(self):
        return Settings.from_dict(self.data.get("settings"))

    # -- io --
    @classmethod
    def create(cls, path):
        os.makedirs(os.path.abspath(path), exist_ok=True)
        if os.path.exists(os.path.join(os.path.abspath(path), cls.FILE)):
            return cls.load(path)
        p = cls(path)
        p.save()
        return p

    @classmethod
    def load(cls, path):
        p = cls(path)
        f = os.path.join(p.path, cls.FILE)
        if not os.path.exists(f):
            raise FileNotFoundError(f"Project မတွေ့ပါ: {p.path} (recap init နဲ့ အရင်ဆောက်ပါ)")
        with open(f, encoding="utf-8") as fh:
            p.data.update(json.load(fh))
        return p

    def save(self):
        tmp = os.path.join(self.path, self.FILE + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.data, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, os.path.join(self.path, self.FILE))

    def update_settings(self, s):
        self.data["settings"] = s.to_dict()

# ======================================================================
# pipeline
# ======================================================================
import hashlib
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


def init_project(path, source, height=720, cookies_file=None, progress=None):
    proj = Project.create(path)
    if os.path.exists(proj.video) and proj.data.get("source") == source:
        log("• video ရှိပြီးသား၊ ပြန်မဆွဲတော့ဘူး")
    else:
        fetch_video(source, proj.video, height, progress, cookies_file)
    proj.data["source"] = source
    proj.save()
    return proj


def run_transcribe(proj, model="base", src_lang="auto", fast=True, max_min=0, lang="my",
                   progress=None):
    t0 = time.time()
    units, vdur, info = transcribe_video(proj.video, proj.path, model, src_lang, fast,
                                         max_min, progress)
    proj.data["units"] = units
    proj.data["vdur"] = vdur
    proj.data["timings"]["transcribe"] = time.time() - t0
    proj.data["info"] = info
    s = proj.settings
    s.lang = lang
    proj.update_settings(s)
    proj.save()
    write_prompt(proj, lang)
    return info


def write_prompt(proj, lang):
    with open(proj.prompt_file, "w", encoding="utf-8") as f:
        f.write(build_prompt(proj.units, lang))
    for k, c in enumerate(build_prompt_chunks(proj.units, lang)):
        with open(os.path.join(proj.path, f"prompt_part{k + 1:02d}.txt"), "w", encoding="utf-8") as f:
            f.write(c["text"])
    return proj.prompt_file


def apply_translation(proj, text, lang):
    units = proj.units
    got = parse_numbered(text, len(units))
    for i, t in got.items():
        units[i]["my"] = clean_text(t, lang)
    missing = [i for i in range(len(units)) if not units[i]["my"]]
    proj.save()
    return len(got), missing


def _synthesize(proj, my_list, s, progress, proxy, hf_token):
    wd = proj.path
    n = len(my_list)
    is_vox = s.engine == "vox"
    vox = VoxEngine(s.vox_ref or None, hf_token) if is_vox else None
    ref_sig = "noref"
    if is_vox and s.vox_ref and os.path.exists(s.vox_ref):
        ref_sig = f"{os.path.basename(s.vox_ref)}-{os.path.getsize(s.vox_ref)}"
    voice = s.resolved_voice()
    sig_base = f"{s.engine}|{voice}|{s.rate_str()}|{ref_sig}"

    def do_line(i):
        wav = os.path.join(wd, f"a{i:04d}.wav")
        keyf = os.path.join(wd, f"a{i:04d}.key")
        key = hashlib.md5(f"{sig_base}|{my_list[i]}".encode()).hexdigest()
        if os.path.exists(wav) and os.path.exists(keyf):
            try:
                with open(keyf) as kf:
                    if kf.read().strip() == key:
                        return i, wav_dur(wav)
            except Exception:
                pass
        if vox is not None:
            raw_audio = os.path.join(wd, f"a{i:04d}_raw.wav")
            used = None
            for _ in range(s.vox_retry):
                used = vox.synth(my_list[i], raw_audio)
                if used:
                    break
                vox.reset()
                time.sleep(3)
            if not used:
                raise RuntimeError(f"VoxCPM မအောင်မြင်ပါ (စာကြောင်း #{i}): {vox.last_err}\n"
                                   "Edge TTS မသုံးထားပါ။ ခဏနေပြီး ပြန်စမ်းပါ။")
        else:
            raw_audio = os.path.join(wd, f"a{i:04d}_raw.mp3")
            for attempt in range(4):
                try:
                    tts_edge(my_list[i], voice, s.rate_str(), raw_audio, proxy)
                    break
                except Exception as ex:
                    if attempt == 3:
                        raise RuntimeError(f"Edge TTS မအောင်မြင်ပါ (စာကြောင်း #{i}): {ex}\n{my_list[i]}")
                    time.sleep(1.5)
        sh(["ffmpeg", "-y", "-i", raw_audio, "-af", TRIM, "-ar", "44100", "-ac", "1", wav])
        with open(keyf, "w") as kf:
            kf.write(key)
        return i, wav_dur(wav)

    durs = [0.0] * n
    workers = 1 if is_vox else max(1, s.tts_workers)
    done = 0
    progress(0.0, f"အသံ ထုတ်နေတယ် 0/{n}")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(do_line, i) for i in range(n)]
        try:
            for fu in as_completed(futs):
                i_, d_ = fu.result()
                durs[i_] = d_
                done += 1
                progress(done / n, f"အသံ ထုတ်နေတယ် {done}/{n}")
        except BaseException:
            for f_ in futs:
                f_.cancel()
            raise
    return durs


def _concat_audio(wd, n):
    lst = os.path.join(wd, "alist.txt")
    with open(lst, "w") as f:
        for i in range(n):
            f.write(f"file '{os.path.join(wd, f'a{i:04d}.wav')}'\n")
    allwav = os.path.join(wd, "all.wav")
    sh(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", allwav])
    return allwav


def _build_plan(units, durs, vdur, max_speed):
    n = len(units)
    bounds = [0.0] + [(units[i]["end"] + units[i + 1]["start"]) / 2
                      for i in range(n - 1)] + [vdur]
    plan, cum = [], 0.0
    for i in range(n):
        s_, e_ = bounds[i], bounds[i + 1]
        w = max(e_ - s_, 0.2)
        d = max(durs[i], 0.05)
        w_use = d * max_speed if w / d > max_speed else w
        plan.append((cum, d, s_, w_use / d))
        cum += d
    return plan, cum


def _build_subs(my_list, durs, lang):
    subs, t0 = [], 0.0
    for i, text in enumerate(my_list):
        parts = split_for_sub(text, lang=lang)
        tot = sum(len(p) for p in parts) or 1
        cur = t0
        for p in parts:
            dd = durs[i] * len(p) / tot
            subs.append((cur, cur + dd, p))
            cur += dd
        t0 += durs[i]
    return subs


def run_render(proj, s, proxy=None, hf_token=None, progress=None):
    progress = progress or Progress()
    lang = s.lang
    conf = LANG_CONF[lang]
    units = proj.units
    if not units:
        raise RuntimeError("units မရှိသေးပါ — recap transcribe အရင်လုပ်ပါ")
    my_list = [clean_text(u["my"], lang) for u in units]
    empty = [i for i, t in enumerate(my_list) if not t.strip()]
    if empty:
        raise RuntimeError(f"စာသား မဖြည့်ရသေးတဲ့ စာကြောင်း: {empty[:20]}"
                           " — recap apply နဲ့ translated.txt ကို အရင်ဖြည့်ပါ")
    if s.engine == "vox" and lang != "my":
        log("⚠️ VoxCPM က မြန်မာအတွက်သာ သင့်တယ်။ ဂျာမနီ/ဂျပန်အတွက် --engine edge သုံးပါ။")

    font = ""
    if s.sub:
        font = find_font(lang, s.font)
        if not font:
            log(f"⚠️ {conf['label']} font မတွေ့ပါ — စာတန်းမပါဘဲ render လုပ်မယ် (--font သို့ fonts/ ထဲ ထည့်ပါ)")
        elif conf["is_mm"] and not has_raqm():
            log("⚠️ libraqm မရှိလို့ မြန်မာစာ ပုံစံ မမှန်နိုင်ပါ (apt install libraqm0)")

    T = {}
    t_all = time.time()

    t_a = time.time()
    durs = _synthesize(proj, my_list, s, progress.sub(0.0, 0.3), proxy, hf_token)
    T["TTS"] = time.time() - t_a

    t_a = time.time()
    allwav = _concat_audio(proj.path, len(units))
    T["audio concat"] = time.time() - t_a

    vdur = proj.data["vdur"]
    plan, total = _build_plan(units, durs, vdur, s.max_speed)
    subs = _build_subs(my_list, durs, lang) if (s.sub and font) else []

    t_a = time.time()
    render_final(proj.video, allwav, proj.final, plan, total, s, subs, font,
                 is_mm=conf["is_mm"], progress=progress.sub(0.3, 1.0))
    T["render"] = time.time() - t_a
    T["total (stage 2)"] = time.time() - t_all

    proj.data["timings"].update({f"stage2 {k}": v for k, v in T.items()})
    proj.update_settings(s)
    proj.save()
    return {"final": proj.final, "new_dur": total, "old_dur": vdur, "timings": T}

# ======================================================================
# cli
# ======================================================================
import argparse
import os
import shutil
import sys


SUP = argparse.SUPPRESS


def add_style_args(p):
    g = p.add_argument_group("voice / language")
    g.add_argument("--lang", type=resolve_lang, default=SUP, help="my | de | ja")
    g.add_argument("--engine", choices=["edge", "vox"], default=SUP)
    g.add_argument("--voice", default=SUP, help="Edge voice name (e.g. my-MM-NilarNeural)")
    g.add_argument("--rate", type=int, default=SUP, help="Edge speed %% (-50..100)")
    g.add_argument("--workers", dest="tts_workers", type=int, default=SUP, help="parallel TTS jobs")
    g.add_argument("--vox-retry", type=int, default=SUP)
    g.add_argument("--vox-ref", default=SUP, help="reference audio for VoxCPM")
    g = p.add_argument_group("video")
    g.add_argument("--height", type=int, choices=[360, 480, 720, 1080], default=SUP)
    g.add_argument("--fps", type=int, choices=[20, 24, 25, 30], default=SUP)
    g.add_argument("--preset", choices=["ultrafast", "superfast", "veryfast"], default=SUP)
    g.add_argument("--max-speed", type=float, default=SUP, help="max video speed-up (1.0-2.0)")
    g.add_argument("--zoom", type=float, default=SUP, help="1.0-1.25")
    g.add_argument("--no-crop", dest="crop", action="store_false", default=SUP,
                   help="black bar auto-crop မလုပ်ဘူး")
    g.add_argument("--no-flip", dest="flip", action="store_false", default=SUP,
                   help="mirror မလုပ်ဘူး")
    g = p.add_argument_group("subtitles / neon")
    g.add_argument("--no-sub", dest="sub", action="store_false", default=SUP)
    g.add_argument("--no-sweep", dest="sub_sweep", action="store_false", default=SUP,
                   help="စာတန်း music sweep မထည့်ဘူး")
    g.add_argument("--sub-size", type=float, default=SUP, help="%% of width (3-12)")
    g.add_argument("--sub-pos", type=int, default=SUP, help="0=top 100=bottom")
    g.add_argument("--font", default=SUP, help="font file path")
    g.add_argument("--no-neon", dest="neon", action="store_false", default=SUP)
    g.add_argument("--neon-th", type=int, default=SUP)
    g.add_argument("--neon-speed", type=float, default=SUP,
                   help="neon laps per second (default 0.15, lower = slower)")


def add_whisper_args(p):
    g = p.add_argument_group("whisper")
    g.add_argument("--model", choices=["tiny", "base", "small"], default="base")
    g.add_argument("--src-lang", default="auto", help="auto | en | zh | ko | ja | hi | th | vi | id")
    g.add_argument("--no-batch", action="store_true", help="whisper batch mode ပိတ်")
    g.add_argument("--max-min", type=int, default=0,
                   help="video အရှည် ကန့်သတ် (မိနစ်) · 0 = ကန့်သတ်မရှိ")


def settings_for(proj, args):
    s = proj.settings
    for k, v in vars(args).items():
        if hasattr(s, k):
            setattr(s, k, v)
    s.rate = max(-50, min(100, s.rate))
    s.max_speed = max(1.0, min(2.0, s.max_speed))
    s.zoom = max(1.0, min(1.25, s.zoom))
    s.sub_size = max(3.0, min(12.0, s.sub_size))
    s.sub_pos = max(0, min(100, s.sub_pos))
    if s.voice and s.engine == "edge":
        valid = [v for v, _ in LANG_CONF[s.lang]["edge_voices"]]
        if s.voice not in valid:
            log(f"⚠️ voice '{s.voice}' က {s.lang} စာရင်းထဲမှာ မပါဘူး — default သုံးမယ်။ ရနိုင်တာ: {', '.join(valid)}")
            s.voice = ""
    return s


def _cookies(args):
    return getattr(args, "cookies", None) or os.environ.get("YT_COOKIES") or None


def cmd_init(args):
    prog = Progress()
    proj = init_project(args.dir, args.source, args.height, _cookies(args), prog)
    prog.done()
    print(f"✅ Project: {proj.path}\n   video: {proj.video}")
    return proj


def cmd_transcribe(args, proj=None):
    proj = proj or Project.load(args.dir)
    lang = getattr(args, "lang", None) or proj.settings.lang
    prog = Progress()
    info = run_transcribe(proj, args.model, args.src_lang, not args.no_batch, args.max_min,
                          lang, prog)
    prog.done()
    nparts = len(build_prompt_chunks(proj.units, lang))
    print(f"✅ {info}\n   Prompt: {proj.prompt_file}  (+ prompt_part01..{nparts:02d}.txt · {CHUNK} ကြောင်းစီ)")
    print("   → prompt_partNN.txt ကို တစ်ခုချင်း Gemini ထဲ paste၊ ရလဒ်အကုန်ကို translated.txt ထဲ အစဉ်လိုက်ထည့်ပြီး `recap apply` လုပ်ပါ")
    return proj


def cmd_prompt(args):
    proj = Project.load(args.dir)
    lang = getattr(args, "lang", None) or proj.settings.lang
    path = write_prompt(proj, lang)
    s = proj.settings
    s.lang = lang
    proj.update_settings(s)
    proj.save()
    if args.print:
        print(open(path, encoding="utf-8").read())
    else:
        print(f"✅ {path}")


def cmd_apply(args, proj=None, file=None):
    proj = proj or Project.load(args.dir)
    file = file or getattr(args, "file", None) or proj.translated_file
    if file == "-":
        text = sys.stdin.read()
    else:
        if not os.path.exists(file):
            sys.exit(f"❌ ဖိုင် မတွေ့ပါ: {file}")
        with open(file, encoding="utf-8") as f:
            text = f.read()
    lang = getattr(args, "lang", None) or proj.settings.lang
    got, missing = apply_translation(proj, text, lang)
    print(f"✅ ဖြည့်ပြီး {got}/{len(proj.units)} ကြောင်း"
          + (f" · မပါသေးတာ: {missing[:15]}" if missing else ""))
    return not missing


def cmd_render(args, proj=None):
    proj = proj or Project.load(args.dir)
    s = settings_for(proj, args)
    prog = Progress()
    res = run_render(proj, s, os.environ.get("EDGE_TTS_PROXY") or None,
                     os.environ.get("HF_TOKEN") or None, prog)
    prog.done()
    out = args.output
    if out:
        shutil.copy(res["final"], out)
    else:
        out = res["final"]
    mb = os.path.getsize(out) / 1e6
    print(f"\n✅ ပြီးပြီ: {out}  ({mb:.1f} MB)")
    print(f"   {res['new_dur']:.1f}s (မူရင်း {res['old_dur']:.1f}s)")
    t = {**{"transcribe": proj.data["timings"].get("transcribe", 0)}, **res["timings"]}
    print("⏱️\n" + timing_table({k: v for k, v in t.items() if v}))


def cmd_preview(args):
    proj = Project.load(args.dir)
    s = settings_for(proj, args)
    font = find_font(s.lang, s.font)
    out = args.output or os.path.join(proj.path, "preview.png")
    make_preview_frame(proj.video, out, s, LANG_CONF[s.lang]["sample"], font,
                       LANG_CONF[s.lang]["is_mm"], getattr(args, "at", 0.0) or 0.0)
    print(f"✅ {out}")


def cmd_run(args):
    key_dir = args.dir or os.path.join("work", "recap")
    args.dir = key_dir
    proj = cmd_init(args)
    s = settings_for(proj, args)
    proj.update_settings(s)
    proj.save()
    args.lang = s.lang
    if not proj.units:
        cmd_transcribe(args, proj)
    else:
        log("• units ရှိပြီးသား၊ whisper ကျော်မယ်")
        write_prompt(proj, s.lang)

    src = args.translated or proj.translated_file
    if not os.path.exists(src):
        print(f"\n📋 Prompt: {proj.prompt_file}  (အပိုင်းခွဲ: prompt_partNN.txt)")
        print(f"   ① prompt_partNN.txt ကို Gemini မှာ တစ်ခုချင်း paste  ② ရလဒ်အကုန်ကို {src} မှာ အစဉ်လိုက် သိမ်း")
        if not sys.stdin.isatty():
            sys.exit(f"\n⏸️ translated ဖိုင်မရှိလို့ ရပ်လိုက်တယ်။ ဖြည့်ပြီးရင် `recap run ... --translated {src}` "
                     "(သို့) `recap apply` + `recap render` လုပ်ပါ")
        input("   ③ ပြီးရင် Enter နှိပ်ပါ ▶ ")
    if not cmd_apply(args, proj, src):
        sys.exit("❌ စာကြောင်း မပြည့်သေးပါ — translated ဖိုင်ကို စစ်ပြီး ပြန် run ပါ (cache ရှိလို့ မြန်မယ်)")
    cmd_render(args, proj)


# -------------------------------------------------------------- parser
def build_parser():
    p = argparse.ArgumentParser(
        prog="recap.py",
        description="Recap video dubber: Whisper -> Gemini (manual) -> TTS -> smart-sync render")
    p.add_argument("--version", action="version", version=f"recap-cli {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("init", help="video link/ဖိုင်ကို project ထဲ ထည့်")
    a.add_argument("source", help="URL (mp4/GDrive/Dropbox/YouTube/FB/TikTok...) or local file")
    a.add_argument("-d", "--dir", required=True, help="project folder")
    a.add_argument("--height", type=int, default=720)
    a.add_argument("--cookies", help="yt-dlp cookies.txt (သို့) env YT_COOKIES")
    a.set_defaults(fn=cmd_init)

    a = sub.add_parser("transcribe", help="Whisper စာသားထုတ် + prompt.txt")
    a.add_argument("dir")
    a.add_argument("--lang", type=resolve_lang, default=None, help="target: my | de | ja")
    add_whisper_args(a)
    a.set_defaults(fn=cmd_transcribe)

    a = sub.add_parser("prompt", help="Gemini prompt ပြန်ထုတ် (ဘာသာစကားပြောင်းချင်ရင်)")
    a.add_argument("dir")
    a.add_argument("--lang", type=resolve_lang, default=None)
    a.add_argument("--print", action="store_true")
    a.set_defaults(fn=cmd_prompt)

    a = sub.add_parser("apply", help="Gemini ရလဒ် ([0] ... [1] ...) ကို ဖြည့်")
    a.add_argument("dir")
    a.add_argument("file", nargs="?", help="default: <dir>/translated.txt  ( - = stdin )")
    a.add_argument("--lang", type=resolve_lang, default=None)
    a.set_defaults(fn=cmd_apply)

    a = sub.add_parser("render", help="အသံထုတ် + Video ပေါင်း")
    a.add_argument("dir")
    a.add_argument("-o", "--output", help="final mp4 ကို ဒီနေရာ copy")
    add_style_args(a)
    a.set_defaults(fn=cmd_render)

    a = sub.add_parser("preview", help="စာတန်း/neon style စစ်ဖို့ ပုံတစ်ပုံ")
    a.add_argument("dir")
    a.add_argument("-o", "--output")
    a.add_argument("--at", type=float, default=0.0, help="preview time (0-5 seconds)")
    add_style_args(a)
    a.set_defaults(fn=cmd_preview)

    a = sub.add_parser("run", help="အစအဆုံး (translation မှာ တစ်ခါ ရပ်မယ်)")
    a.add_argument("source")
    a.add_argument("-d", "--dir", help="project folder (default work/recap)")
    a.add_argument("--translated", help="Gemini ရလဒ် ဖိုင် (ရှိပြီးသားဆိုရင် မရပ်ဘူး)")
    a.add_argument("-o", "--output")
    a.add_argument("--cookies")
    add_whisper_args(a)
    add_style_args(a)
    a.set_defaults(fn=cmd_run)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.cmd == "run" and not hasattr(args, "height"):
        args.height = 720
    try:
        args.fn(args)
    except KeyboardInterrupt:
        sys.exit("\n⛔ ရပ်လိုက်ပါပြီ (ပြန် run ရင် ပြီးသား အသံတွေ cache ကနေ ယူမယ်)")
    except Exception as ex:
        sys.exit(f"\n❌ {ex}")


# ======================================================================
# Web UI (Gradio)
# ======================================================================
import hashlib
import uuid

import gradio as gr

recap = sys.modules[__name__]   # UI code below calls recap.X; engine lives in this same file

BASE = os.path.join(os.getcwd(), "work_ui")
os.makedirs(BASE, exist_ok=True)
PORT = int(os.environ.get("PORT", "7860"))

LANGS = {"မြန်မာ": "my", "ဂျာမနီ": "de", "ဂျပန်": "ja"}


def voices_for(code):
    return [v for v, _ in LANG_CONF[code]["edge_voices"]]


def _path(f):
    if not f:
        return ""
    return f if isinstance(f, str) else getattr(f, "name", "")


def _cb(progress):
    return lambda frac, label=None: progress(frac, desc=label or "")


def _source(url, upload):
    up = _path(upload)
    return up or (url or "").strip()


def _pbar(frac, width=20):
    frac = max(0.0, min(float(frac), 1.0))
    fill = int(width * frac)
    return "█" * fill + "░" * (width - fill)


def _ensure_video(url, upload, height, cb):
    source = _source(url, upload)
    if not source:
        raise gr.Error("Video link ထည့်ပါ (သို့) ဖိုင်တင်ပါ")
    key = source
    if os.path.isfile(source):
        key += f"|{os.path.getsize(source)}"
    pdir = os.path.join(BASE, hashlib.md5(key.encode()).hexdigest()[:10])
    dl_prog = Progress(0.0, 1.0, parent=lambda f, l=None: cb(f, "Video ယူနေတယ်..."))
    proj = init_project(pdir, source, int(height), os.environ.get("YT_COOKIES"), dl_prog)
    return pdir, proj


# ---------------------------------------------------------------- fetch + preview
FETCH = {}   # job id -> state (background download jobs)


def start_fetch(url, upload, height):
    """ခလုတ်နှိပ်တာနဲ့ ချက်ချင်း return လုပ်ပြီး download ကို နောက်ကွယ်မှာ စတင်မယ်။"""
    if not _source(url, upload):
        raise gr.Error("Video link ထည့်ပါ (သို့) ဖိုင်တင်ပါ")
    jid = uuid.uuid4().hex[:8]
    st = {"f": 0.0, "label": "Video ယူနေတယ်...", "res": None, "err": None,
          "done": False, "t0": time.time()}
    FETCH[jid] = st

    def cb(f, label=None):
        st["f"] = max(0.0, min(float(f), 1.0))
        if label:
            st["label"] = label

    def work():
        try:
            st["res"] = _ensure_video(url, upload, height, cb)
        except BaseException as ex:
            st["err"] = ex
        finally:
            st["done"] = True

    threading.Thread(target=work, daemon=True).start()
    return jid, "⏳ စတင်နေတယ်...", gr.Timer(active=True)


def poll_fetch(jid, lang_name, height, zoom, crop, flip, sub, sub_size, sub_pos, neon, sweep,
               pv_t):
    """Timer က ၁ စက္ကန့်တစ်ခါ ခေါ်မယ် — % ပြ၊ ပြီးရင် Preview ထုတ်ပြီး timer ရပ်။"""
    st = FETCH.get(jid) if jid else None
    if st is None:
        return gr.skip(), gr.skip(), gr.skip(), gr.Timer(active=False)
    elapsed = time.time() - st["t0"]
    if not st["done"]:
        if st["f"] >= 1.0:
            msg = "🔄 MP4 ပြင်ဆင်နေတယ် (ဖိုင်ကြီးရင် ခဏကြာမယ်)"
        else:
            msg = f"⬇️ {st['label']} {int(st['f'] * 100)}%"
        return gr.skip(), f"{msg} · {elapsed:.0f}s", gr.skip(), gr.skip()
    FETCH.pop(jid, None)
    if st["err"] is not None:
        return "", f"❌ {str(st['err'])[:500]}", gr.skip(), gr.Timer(active=False)
    pdir, proj = st["res"]
    mb = os.path.getsize(proj.video) / 1e6
    img = _preview(pdir, lang_name, height, zoom, crop, flip, sub, sub_size, sub_pos, neon,
                   sweep, pv_t, False)
    return (pdir, f"✅ Video ရပြီ ({mb:.1f} MB) · အောက်က Preview မှာ စာတန်း/Neon ချိန်ပြီး 1️⃣ နှိပ်ပါ",
            img, gr.Timer(active=False))


def _preview(pdir, lang_name, height, zoom, crop, flip, sub, sub_size, sub_pos, neon, sweep,
             pv_t, strict):
    if not pdir or not os.path.exists(os.path.join(pdir, "src.mp4")):
        if strict:
            raise gr.Error("Video link ထည့်ပြီး ⬇️ အရင်နှိပ်ပါ")
        return gr.skip()
    code = LANGS[lang_name]
    try:
        proj = Project.load(pdir)
        s = proj.settings
        s.lang, s.height, s.zoom = code, int(height), float(zoom)
        s.crop, s.flip, s.sub, s.neon = bool(crop), bool(flip), bool(sub), bool(neon)
        s.sub_size, s.sub_pos, s.sub_sweep = float(sub_size), int(sub_pos), bool(sweep)
        s.neon_speed = Settings().neon_speed   # always use the current default speed in the UI
        font = find_font(code, s.font) if s.sub else None

        # ဖိုင်နာမည် အသစ်ပေးမှ browser/gradio cache မကပ်ဘဲ preview ပြောင်းမယ်
        for old in glob.glob(os.path.join(pdir, "preview_*.png")):
            try:
                os.remove(old)
            except OSError:
                pass
        out = os.path.join(pdir, f"preview_{int(time.time() * 1000)}.png")
        make_preview_frame(proj.video, out, s, LANG_CONF[code]["sample"], font,
                           LANG_CONF[code]["is_mm"], float(pv_t or 0.0))
        return out
    except Exception as ex:
        log(f"preview error: {ex}")
        if strict:
            raise gr.Error(str(ex)[:400])
        return gr.skip()


def preview_btn(*a):
    return _preview(*a, True)


def preview_auto(*a):
    return _preview(*a, False)


# ---------------------------------------------------------------- step 1 (whisper) + live %
TRANS = {}   # job id -> state (background Whisper jobs)


def start_step1(url, upload, lang_name, model, src_lang, max_min, height):
    """1️⃣ ခလုတ်နှိပ်တာနဲ့ ချက်ချင်း return လုပ်ပြီး Whisper ကို နောက်ကွယ်မှာ run ပြီး၊
    Timer က ၁ စက္ကန့်တစ်ခါ % ကို ပြမယ်။"""
    if not _source(url, upload):
        raise gr.Error("Video link ထည့်ပါ (သို့) ဖိုင်တင်ပါ")
    code = LANGS[lang_name]
    jid = uuid.uuid4().hex[:8]
    st = {"f": 0.0, "label": "စတင်နေတယ်...", "res": None, "err": None,
          "done": False, "t0": time.time()}
    TRANS[jid] = st

    def cb(f, label=None):
        st["f"] = max(0.0, min(float(f), 1.0))
        if label:
            st["label"] = label

    def work():
        try:
            cb(0.0, "Video ယူနေတယ်...")
            pdir, proj = _ensure_video(url, upload, height, lambda f, l=None: cb(f * 0.2, l))
            tr = Progress(0.2, 1.0, parent=lambda f, l=None: cb(f, l))
            info = run_transcribe(proj, model, src_lang, True, int(max_min), code, tr)
            chunks = build_prompt_chunks(proj.units, code)
            st["res"] = (pdir, info, chunks)
        except BaseException as ex:
            st["err"] = ex
        finally:
            st["done"] = True

    threading.Thread(target=work, daemon=True).start()
    return jid, "⏳ စတင်နေတယ်... 0%", gr.Timer(active=True), gr.update(interactive=False)


def poll_step1(jid):
    """Timer tick — % + progress bar ပြ၊ ပြီးရင် Prompt အပိုင်းတွေ ထုတ်ပြီး timer ရပ်။"""
    st = TRANS.get(jid) if jid else None
    if st is None:
        return (gr.skip(), gr.skip(), gr.skip(), gr.skip(),
                gr.Timer(active=False), gr.update(interactive=True))
    elapsed = time.time() - st["t0"]
    if not st["done"]:
        pct = int(st["f"] * 100)
        msg = f"⏳ {st['label']}\n\n`[{_pbar(st['f'])}] {pct}%` · {elapsed:.0f}s"
        return gr.skip(), msg, gr.skip(), gr.skip(), gr.skip(), gr.skip()
    TRANS.pop(jid, None)
    if st["err"] is not None:
        return (gr.skip(), f"❌ {str(st['err'])[:500]}", gr.skip(), gr.skip(),
                gr.Timer(active=False), gr.update(interactive=True))
    pdir, info, chunks = st["res"]
    return (pdir,
            f"✅ 100% · {info} · Prompt {len(chunks)} ပိုင်း · {elapsed:.0f}s",
            chunks, {}, gr.Timer(active=False), gr.update(interactive=True))


# ---------------------------------------------------------------- step 2 (tts + render)
def _make_thumb(video_path, thumb_path):
    """Extract a still frame so the result shows a picture without having to press play."""
    for ss in ("1", "0"):
        try:
            sh(["ffmpeg", "-y", "-ss", ss, "-i", video_path, "-frames:v", "1",
                "-q:v", "3", thumb_path])
            if os.path.exists(thumb_path) and os.path.getsize(thumb_path) > 100:
                return thumb_path
        except Exception:
            continue
    return None


def step2(pdir, translated, parts, lang_name, engine, voice, rate, workers, vox_ref,
          height, fps, preset, max_speed, crop, flip, zoom,
          sub, sub_size, sub_pos, neon, sweep, progress=gr.Progress()):
    if not pdir or not os.path.isdir(pdir):
        raise gr.Error("အဆင့် 1 ကို အရင်လုပ်ပါ")
    code = LANGS[lang_name]
    try:
        proj = Project.load(pdir)
        if not proj.units:
            raise RuntimeError("1️⃣ စာသားထုတ် ကို အရင်လုပ်ပါ")
        text = "\n".join((parts or {})[k] for k in sorted(parts or {})) + "\n" + (translated or "")
        got, missing = apply_translation(proj, text, code)
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
        s.sub_sweep = bool(sweep)
        s.neon_speed = Settings().neon_speed   # always use the current default speed in the UI
        prog = Progress(0.0, 1.0, parent=_cb(progress))
        res = run_render(proj, s, os.environ.get("EDGE_TTS_PROXY") or None,
                         os.environ.get("HF_TOKEN") or None, prog)
    except gr.Error:
        raise
    except Exception as ex:
        raise gr.Error(str(ex)[:600])

    # old results out; unique file name so the browser never shows a stale/blank cached video
    for old in glob.glob(os.path.join(pdir, "final_*.mp4")) + glob.glob(os.path.join(pdir, "final_*.jpg")):
        try:
            os.remove(old)
        except OSError:
            pass
    stamp = int(time.time())
    out = os.path.join(pdir, f"final_{code}_{stamp}.mp4")
    shutil.copy(res["final"], out)
    thumb = _make_thumb(out, os.path.join(pdir, f"final_{code}_{stamp}.jpg"))
    mb = os.path.getsize(out) / 1e6
    return (out, out, thumb,
            f"✅ ပြီးပြီ · {res['new_dur']:.1f}s (မူရင်း {res['old_dur']:.1f}s) · {mb:.1f} MB")


def reset_all(pdir):
    if pdir and os.path.isdir(pdir) and os.path.abspath(pdir).startswith(BASE):
        shutil.rmtree(pdir, ignore_errors=True)
    # pdir_state, status_fetch, status1, chunks_state, parts_state, translated, pv_img, video, dl, thumb, status2
    return "", "", "", [], {}, "", None, None, None, None, "🗑️ ရှင်းပြီး"


def make_setter(lo, hi):
    """Returns a handler for one part's paste box: stores the text + shows ✅/⚠️ status."""
    def f(text, parts):
        parts = dict(parts or {})
        text = text or ""
        parts[lo] = text          # key = part's first line number
        if not text.strip():
            return parts, "⏳ မဖြည့်သေး"
        got, missing = check_part(text, lo, hi)
        if missing:
            return parts, f"⚠️ {len(got)}/{hi - lo} · မပါတာ: {missing[:10]}"
        return parts, f"✅ {hi - lo}/{hi - lo}"
    return f


# ---------------------------------------------------------------- UI layout
with gr.Blocks(title="Recap → Multi-Language") as demo:
    gr.Markdown("# 🎙️ Recap → Multi-Language  `build 15`\nWhisper → Gemini (၅၀ ကြောင်းစီ) → Edge TTS / VoxCPM → Smart Sync → စာတန်း (music sweep) + Neon")
    pdir_state = gr.State("")

    # hidden settings (UI မှာ မပြဘူး) — တန်ဖိုးတွေက နောက်ကွယ်မှာ ပုံသေအတိုင်း အလုပ်လုပ်တယ်
    height = gr.State(720)
    fps = gr.State(24)
    preset = gr.State("ultrafast")
    max_speed = gr.State(1.25)
    crop = gr.State(True)
    model = gr.State("base")
    src_lang = gr.State("auto")
    max_min = gr.State(0)       # 0 = unlimited
    translated = gr.State("")   # "ရလဒ်အကုန် တစ်ခါတည်း paste" box ကို ဖျောက်ထားတယ် (step2 အတွက် အလွတ်ပဲ ပို့မယ်)

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

    gr.Markdown("### 📥 Video")
    url = gr.Textbox(label="Video link (mp4 / Google Drive / Dropbox / YouTube / FB / TikTok ...) — paste ပြီး Enter",
                     placeholder="https://...")
    upload = gr.File(label="(သို့) Video ဖိုင်တင်", type="filepath",
                     file_types=[".mp4", ".mov", ".mkv", ".webm", ".m4v"])
    b_fetch = gr.Button("⬇️ Video ဆွဲ + Preview")
    status_fetch = gr.Markdown()

    gr.Markdown("### 🎨 စာတန်း / Neon (Preview ကြည့်ပြီး ချိန်ပါ)")
    pv_img = gr.Image(label="Preview", type="filepath", interactive=False)
    pv_t = gr.Slider(0.0, PREVIEW_SECS, value=0.0, step=0.1,
                     label=f"⏱️ Preview အချိန် (0 – {PREVIEW_SECS:.0f} စက္ကန့် · ဆွဲပြီး ရွှေ့ကြည့်ပါ)")
    with gr.Accordion("🎛️ စာတန်း / Neon / Mirror", open=False):
        with gr.Row():
            sub = gr.Checkbox(True, label="စာတန်း")
            sweep = gr.Checkbox(True, label="🎵 Music sweep")
            neon = gr.Checkbox(True, label="Neon")
            flip = gr.Checkbox(True, label="Mirror")
        sub_size = gr.Slider(3.0, 12.0, value=7.0, step=0.5, label="စာအရွယ် (% width)")
        sub_pos = gr.Slider(0, 100, value=90, step=1, label="စာတန်း နေရာ (0=အပေါ် 100=အောက်)")
        zoom = gr.Slider(1.0, 1.25, value=1.0, step=0.01, label="Zoom")

    pv_in = [pdir_state, lang, height, zoom, crop, flip, sub, sub_size, sub_pos, neon, sweep, pv_t]
    fetch_job = gr.State("")
    fetch_timer = gr.Timer(1.0, active=False)
    for trig in (url.submit, b_fetch.click, upload.upload):
        trig(start_fetch, [url, upload, height], [fetch_job, status_fetch, fetch_timer])
    fetch_timer.tick(poll_fetch,
                     [fetch_job, lang, height, zoom, crop, flip, sub, sub_size, sub_pos, neon,
                      sweep, pv_t],
                     [pdir_state, status_fetch, pv_img, fetch_timer],
                     show_progress="hidden")

    # Slider: .release က phone (touch) မှာ မခံစားရလို့ .change သုံးတယ်၊
    # always_last = ဆွဲနေတုန်း တန်းစီမနေဘဲ နောက်ဆုံးတန်ဖိုးပဲ ယူမယ်
    for c in (sub_size, sub_pos, zoom, sub, sweep, neon, flip, lang, pv_t):
        c.change(preview_auto, pv_in, pv_img,
                 trigger_mode="always_last", show_progress="hidden", queue=True)

    gr.Markdown("### 1️⃣ စာသားထုတ်")
    b1 = gr.Button("1️⃣ စာသားထုတ် (Whisper)", variant="primary")
    status1 = gr.Markdown()
    trans_job = gr.State("")
    trans_timer = gr.Timer(1.0, active=False)

    chunks_state = gr.State([])
    parts_state = gr.State({})

    # ">" မြှားလေးနဲ့ ခေါက်/ဖွင့် လုပ်လို့ရတဲ့ Prompt အပိုင်းများ
    with gr.Accordion("📋 Prompt အပိုင်းများ (၅၀ ကြောင်းစီ) — Copy → Gemini → ရလဒ်ကို အောက်က box မှာ paste",
                      open=False):

        @gr.render(inputs=chunks_state)
        def show_chunks(chunks):
            if not chunks:
                gr.Markdown("_1️⃣ နှိပ်ပြီးရင် Prompt အပိုင်းတွေ ဒီမှာ ပေါ်မယ်_")
                return
            for k, c in enumerate(chunks):
                with gr.Accordion(f"Part {k + 1}/{len(chunks)} · [{c['lo']} – {c['hi'] - 1}]",
                                  open=False):
                    gr.Textbox(value=c["text"], lines=3, max_lines=5, interactive=False,
                               show_copy_button=True, label="① Prompt (Copy ခလုတ်နှိပ်)")
                    box = gr.Textbox(label=f"② Gemini ရလဒ် Part {k + 1} ကို ဒီမှာ paste",
                                     lines=3, max_lines=5)
                    st = gr.Markdown("⏳ မဖြည့်သေး")
                    box.change(make_setter(c["lo"], c["hi"]), [box, parts_state], [parts_state, st])

    gr.Markdown("### 2️⃣ အသံထုတ် + Video")
    b2 = gr.Button("2️⃣ အသံထုတ် + Video ပေါင်း", variant="primary")
    status2 = gr.Markdown()
    # autoplay=True -> result shows/plays right away (no need to tap to make it appear)
    video = gr.Video(label="ရလဒ်", autoplay=True)
    thumb_img = gr.Image(label="ရလဒ် Thumbnail", type="filepath", interactive=False)
    dl = gr.File(label="⬇️ Download MP4")
    b3 = gr.Button("🗑️ အသစ်ပြန်စ")

    # 1️⃣ Whisper: ခလုတ်နှိပ်ရင် job စပြီး Timer က status1 မှာ % ကို ၁ စက္ကန့်တစ်ခါ update လုပ်မယ်
    b1.click(start_step1, [url, upload, lang, model, src_lang, max_min, height],
             [trans_job, status1, trans_timer, b1])
    trans_timer.tick(poll_step1, trans_job,
                     [pdir_state, status1, chunks_state, parts_state, trans_timer, b1],
                     show_progress="hidden")

    b2.click(step2, [pdir_state, translated, parts_state, lang, engine, voice, rate, workers,
                     vox_ref, height, fps, preset, max_speed, crop, flip, zoom,
                     sub, sub_size, sub_pos, neon, sweep],
             [video, dl, thumb_img, status2])
    b3.click(reset_all, pdir_state,
             [pdir_state, status_fetch, status1, chunks_state, parts_state, translated,
              pv_img, video, dl, thumb_img, status2])

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=PORT, show_error=True)
