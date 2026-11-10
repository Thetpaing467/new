"""Recap CLI (slim): Whisper -> Gemini (manual) -> Edge TTS -> smart-sync render -> auto split.

Languages: Burmese (my) / German (de) / English (en) / Spanish (es)
Sources: YouTube / TikTok (yt-dlp), Rednote, local file

Usage: python recap.py run <video|url> -d work/ep1 --lang my
       python recap.py run <video|url> -d work/ep1 --lang de
       python recap.py run <video|url> -d work/ep1 --lang en
       python recap.py run <video|url> -d work/ep1 --lang es
       python recap.py ui            (Gradio web UI, dark mode)
"""
__version__ = "1.27.4"

import argparse
import asyncio
import glob
import hashlib
import inspect
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict, fields
from typing import Optional
from urllib.parse import urlparse

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, features

# ======================================================================
# config
# ======================================================================
TARGET, MAXD = 4.2, 5.6
SUB_LINES = 2
PREVIEW_SECS = 5.0
CHUNK = 100

ASPECTS = {"free": None, "9:16": 9 / 16, "1:1": 1.0, "4:5": 4 / 5,
           "3:4": 3 / 4, "16:9": 16 / 9, "4:3": 4 / 3}

# Burmese — soft trim + 0.2s pad
TRIM = (
    "silenceremove=start_periods=1:start_threshold=-50dB,areverse,"
    "silenceremove=start_periods=1:start_threshold=-50dB,areverse,apad=pad_dur=0.2"
)

# German / English / Spanish — GAPLESS (hard silence trim, 40ms pad only)
TRIM_GAPLESS = (
    "silenceremove=start_periods=1:start_threshold=-38dB:start_silence=0.02:"
    "stop_periods=-1:stop_threshold=-38dB:stop_silence=0.02,"
    "areverse,"
    "silenceremove=start_periods=1:start_threshold=-38dB:start_silence=0.02:"
    "stop_periods=-1:stop_threshold=-38dB:stop_silence=0.02,"
    "areverse,"
    "apad=pad_dur=0.04"
)
TRIM_DE = TRIM_GAPLESS
TRIM_EN = TRIM_GAPLESS
TRIM_ES = TRIM_GAPLESS

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

_RECAP_MODE = (
    "RECAP MODE (very important): first read ALL numbered lines below and understand the whole "
    "story: who the characters are, what happens, and why. Then retell each line as recap "
    "narration, not as a translation. You MAY condense: drop filler, repeated words, small talk "
    "and unimportant details, and keep only what moves the story (who did what, why, the result, "
    "the emotion). Each output line should be about 50-70% as long as a literal translation. "
    "Never leave a line empty: if a line is unimportant, write one very short bridge sentence. "
    "Do NOT move content between lines and do NOT invent events that are not in the source. "
)

_RECAP_NOTE = (
    "Continue in the same RECAP MODE as the earlier parts: condense, keep only story-moving "
    "content, about 50-70% of a literal translation, never leave a line empty. "
)

_STORY_CORE = (
    "NARRATION STYLE (drama-recap, third person): the source lines are raw speech-to-text of "
    "characters TALKING. Do NOT translate them as direct dialogue and do NOT use quotation marks. "
    "Retell them as a narrator who watched the scene: turn what a character says into reported "
    "speech (who said what to whom). Work out who is speaking and who is being spoken to from "
    "context (mother, father, boss, neighbour and so on) and use the character names or "
    "relationship words. Turn first-person and second-person speech into third person. "
    "Keep the narrator calm, warm and clear, like someone summarising a TV drama episode to a "
    "friend. Each output line is ONE narrated passage that flows naturally like a real story; "
    "link actions with natural connectors. "
    "NATURAL TRANSLATION (very important): the source language can be English, Chinese, or any "
    "other language. DO NOT translate word-by-word or literally. Translate the MEANING, "
    "INTENTION and EMOTION, not just the words. Idioms, slang, proverbs and cultural references "
    "must be converted into their closest natural equivalent in the target language. If there is "
    "no direct equivalent, rewrite the sentence so a native speaker immediately understands the "
    "situation. The result must sound like everyday spoken language, not like a machine, a "
    "dictionary or a textbook. If a translated line sounds confusing, unnatural or meaningless, "
    "REWRITE it until it sounds natural. "
    "INNER THOUGHTS & MOTIVES: do not just state what happened, show HOW it happened. Add the "
    "emotion and tone of the character when the scene makes it clear (angrily shouted, whispered "
    "with tears, smiled coldly). Make the listener feel the scene like a movie. "
    "SMOOTH TRANSITIONS: use varied transition words to connect scenes (at that moment, "
    "meanwhile, suddenly, but then). Never repeat the same connector or the same sentence "
    "opening in consecutive lines. "
    "HIGHLIGHT PLOT TWISTS: when there is a surprising turn in the story, emphasize it with "
    "dramatic but natural phrasing so the listener feels the shock. "
    "VOICE TONE & PITCH: let the words themselves carry the character's emotion. A question "
    "should sound like a real question, shock should sound astonished, anger should sound sharp. "
    "SENTENCE SEGMENTATION: keep sentences short and punchy. Break long thoughts into two or "
    "three short sentences so the speech engine can read them with natural pauses and "
    "intonation. "
    "NUMBERS & SYMBOLS: always write numbers, percentages, currencies and symbols as spoken "
    "words in the target language, so the speech engine reads them naturally. "
    "RELATIONSHIP DYNAMICS: state the relationship between characters when it matters "
    "(Mike's boss Jake, Mike's ex-girlfriend Mary) so the listener keeps track of who is who. "
    "EMPHASIS ON KEY POINTS: use strong, clear language for the most important moments "
    "(the big win, the big fight) so the listener remembers them. "
    "CONTEXT & BACKGROUND: if the listener needs background to understand a scene, add it "
    "briefly and naturally, but only from what the source lines make clear. "
    "Keep insults, jokes, arguments and emotions of the original, but report them rather than "
    "act them out. Do NOT add hooks, questions to the viewer, commentary or facts that are not "
    "in the source lines. Do NOT address the audience. "
    "CHARACTER NAMES: if a character's name is clearly spoken in the source lines, use ONLY that "
    "name for the character and spell it the same way every time. "
    "Speech-to-text often misspells names: if two similar-sounding names appear, treat them as "
    "the same person and use one spelling throughout. "
)

_STORY_MY = (
    "RECAP NARRATOR STYLE (very important): you are a calm Burmese movie/drama recap narrator "
    "retelling the story to viewers in third person, present tense, using the character names. "
    "Do NOT use ကျွန်တော်, do NOT use ဗျ, do NOT add your own opinions, jokes or comparisons. "
    "Just tell what happens, clearly and smoothly, in the order it happens. "
    "ENDINGS: most lines end with ပါတယ်။ (about three lines out of four). Use plain တယ်။ or "
    "ပါဘူး။ only now and then for variety. Never end with the literary ၏ / သည်။. "
    "CONNECTORS: link the events with natural spoken connectors and vary them: "
    "အဲ့ဒါနဲ့၊ ဒါပေမဲ့၊ အဲဒီအချိန်မှာပဲ၊ ပြီးတော့၊ အဲ့တော့၊ ဒါကြောင့်၊ နောက်ပိုင်းမှာတော့၊ "
    "တစ်ဖက်မှာတော့၊ ရုတ်တရက်၊ ဒီအချိန်မှာ. Never use the same connector in two lines in a row. "
    "SPEECH: report what characters say in the narrator's voice, with a variety of verbs: "
    "လို့ ပြောပါတယ်၊ လို့ မေးပါတယ်၊ လို့ ဖြေပါတယ်၊ လို့ ရှင်းပြပါတယ်၊ လို့ သတိပေးပါတယ်၊ "
    "လို့ အော်ပါတယ်၊ လို့ ငြင်းပါတယ်. No quotation marks. "
    "NAMES: write every character name in Burmese script (transliterate it), the same way "
    "every time, and use the name as the subject (ဝါလ်တာက ...၊ လန်စ်က ...). Never leave "
    "English or Chinese letters in the output. When a name is unknown use a role word "
    "(အမေ၊ သူငယ်ချင်း၊ သူဌေး၊ ကောင်မလေး၊ ဦးကြီး) or leave the subject out. Never write "
    "လူတစ်ယောက်. Use သူ only when it is perfectly clear who is meant. "
    "TRANSLATE MEANING: do not translate word by word. Convert idioms and slang into natural "
    "spoken Burmese. Mention the emotion when the scene shows it (စိတ်တိုတိုနဲ့၊ ဝမ်းသာသွားပြီး၊ "
    "အံ့ဩသွားပြီး). "
    "SENTENCES: keep them short to medium, one idea each, with a comma-type ၊ at breath points "
    "and ။ at the end. Write clean Burmese with normal spacing between phrases. "
    "STYLE EXAMPLES (copy the TONE only, never the content): "
    "'ဝိုင်းလ်ဒါဆိုတဲ့ ကောင်လေးက ငယ်ငယ်ထဲကတည်းက ဉာဏ်ကောင်းတဲ့ ကလေးတစ်ယောက် ဖြစ်ပါတယ်။' "
    "'ဒါပေမဲ့ တစ်ယောက်တည်းပဲ လှုပ်ရှားတတ်တဲ့ လန်စ်က အဖွဲ့တွေကို မခေါ်ဘဲ တစ်ယောက်တည်း ဝင်သွားပါတယ်။' "
    "'စိတ်တိုသွားတဲ့ လန်စ်က ဖြေဆေးပေးဖို့ တောင်းပေမယ့် ဖြေဆေးက မရှိသေးပါဘူး။' "
    "'အဲဒီအချိန်မှာပဲ မာစီနဲ့ သူ့အဖွဲ့တွေ ရောက်လာတာကြောင့် သူတို့နှစ်ယောက် ထွက်ပြေးကြပါတယ်။' "
    "'ကမ္ဘာပေါ်မှာ လူဆိုး လူကောင်းဆိုတာ မရှိဘူး၊ လူက လူပဲလို့ ဝါလ်တာက ပြောပါတယ်။' "
    "NUMBERS: write numbers as Burmese words (၇၀ → ခုနစ်ဆယ်). "
)

_STORY_DE = (
    "GERMAN STORYTELLER VOICE (very important): write like a real German person who tells a "
    "friend over coffee what happened in a series. It must NOT sound like a translation, a "
    "newspaper article, a school essay or a subtitle file. "
    "TENSE: tell the whole recap in the present tense (Präsens), the way people retell a film: "
    "'Xiao Li kommt nach Hause und sieht, dass die Tür offen steht.' Use Perfekt or Plusquamperfekt "
    "only for things that happened earlier ('Das hatte sie ihm schon vor Jahren versprochen.'). "
    "Do not use the literary Präteritum for narration ('erwiderte er', 'entgegnete sie'). "
    "REPORTED SPEECH, THE SPOKEN WAY: use a normal 'dass' clause or a plain main clause: "
    "'Sie sagt, dass sie keine Lust mehr hat.' or 'Sie sagt, sie hat keine Lust mehr.' "
    "NEVER use the stiff Konjunktiv I ('sie habe', 'er sei', 'sie wolle') — nobody speaks like that. "
    "REPORTING VERBS (very important): never the same reporting verb in two consecutive lines, "
    "and 'sagt' at most once every four lines. Rotate: fragt, antwortet, erklärt, meint, "
    "erwidert, ruft, schreit, flüstert, warnt, verspricht, streitet ab, gibt zu, lacht, droht, "
    "bittet, murmelt, fährt sie an, hakt nach. When the action already shows it, drop the "
    "reporting verb and just state what the person does. "
    "SPOKEN GERMAN FEEL: use everyday words and, sparingly, modal particles where they really fit "
    "(doch, ja, halt, eben, mal, schon, eigentlich, denn, ausgerechnet) — not in every line. "
    "Contractions like 'ist's', 'gibt's', 'hat's' are fine. "
    "Prefer lively verbs to noun phrases: NOT 'Er führt ein Gespräch mit ihr' but 'Sie reden "
    "miteinander'; NOT 'Sie trifft die Entscheidung zu gehen' but 'Sie beschließt zu gehen'. "
    "No office German (Behördendeutsch), no Nominalstil, no 'bezüglich', 'diesbezüglich', "
    "'hinsichtlich', 'jedoch' in every line. "
    "SENTENCE VARIETY: mix short lines ('Die Tür ist offen.') with medium ones. Do not start "
    "every sentence with the character's name; sometimes start with a time or place "
    "('Am nächsten Morgen ...', 'Kurz darauf ...', 'Draußen im Flur ...'). Do not start more "
    "than two lines in a row with the same word. Keep the verb in second position and avoid long "
    "nested sentences (Schachtelsätze). "
    "CONNECTORS: vary them — und dann, aber, plötzlich, kurz darauf, währenddessen, trotzdem, "
    "dabei, ausgerechnet, kaum ist sie da, inzwischen, auf einmal, nur leider. "
    "IDIOMS: never translate Chinese or English idioms literally. Replace them with a German "
    "expression that people really say, or just say plainly what is meant. "
    "PRONOUNS: use er, sie, ihm, ihr, ihnen naturally so the text flows and does not repeat "
    "names all the time. If two characters of the same gender are in the scene, use the name or "
    "a role word (die Mutter, der Chef) instead, so nobody gets mixed up. "
    "NAMES: keep Chinese and other Asian names in their Latin / Pinyin spelling, exactly the same "
    "every time, never translate them. If a name is not known, never write 'die Person', "
    "'jemand' or 'eine Person'. Use a simple role word from the context (die Mutter, der Vater, "
    "der Chef, die Nachbarin, die Freundin) or a look-based word: der junge Mann, die junge "
    "Frau, der alte Mann, die ältere Frau, das kleine Mädchen — and keep the same word for the "
    "same person. "
    "NUMBERS & SYMBOLS (German): write everything as words, never digits or symbols: "
    "'50%' -> 'fünfzig Prozent', '100 Dollar' -> 'hundert Dollar', '3 Uhr' -> 'drei Uhr', "
    "'2019' -> 'zweitausendneunzehn', '1,5 Millionen' -> 'eineinhalb Millionen'. "
    "No abbreviations: 'zum Beispiel' instead of 'z. B.', 'und so weiter' instead of 'usw.'. "
    "EMOTION: show feelings through word choice and rhythm: 'Sie schreit ihn an', 'er flüstert "
    "mit Tränen in den Augen', 'sie lächelt kalt', 'er traut seinen Ohren nicht'. A question a "
    "character asks should come across as a real question: 'Sie fragt ihn, woher er das "
    "überhaupt weiß.' "
    "FINAL SELF-CHECK for every line: imagine reading it aloud to a friend. If it sounds like a "
    "translation, a textbook or a robot, rewrite it in the way a native German would really "
    "say it. "
)

_STORY_EN = (
    "ENGLISH DRAMATIC RECAP NARRATOR (very important): you are a cinematic drama-recap "
    "narrator for a YouTube/TikTok recap channel. Your voice is tense, gripping and emotional — "
    "like a movie trailer narrator telling the story to a viewer who cannot look away. It must "
    "NOT sound like a news article, a Wikipedia summary, a translated textbook, or a flat "
    "subtitle file. "
    "TENSE: tell the whole recap in the simple present tense, the way people retell a film: "
    "'Xiao Li comes home and sees that the door is open.' Use past perfect only for things that "
    "happened earlier ('She had promised him that years ago.'). Do not mix tenses. "
    "SENTENCE RHYTHM — the heart of the dramatic style: mix VERY short punchy lines with "
    "medium flowing ones. A short line of two or three words is allowed and powerful ('A scream.', "
    "'Then everything changed.', 'But he did not explode.'). Never write three long sentences "
    "in a row — break them up. Vary sentence openings: sometimes a subject, sometimes a time or "
    "place ('That night...', 'In the hallway...', 'Out of nowhere...'), sometimes a "
    "conjunction ('But...', 'And then...', 'Only then...'). Do not start more than two lines "
    "in a row with the same word. Avoid long nested sentences — the TTS engine must never run "
    "out of breath. "
    "REPORTED SPEECH, THE SPOKEN WAY: use a natural 'that' clause or plain reporting: "
    "'She says she's had enough.' or 'He tells her that he's leaving.' "
    "NEVER use stiff literary reported speech ('she stated that she would henceforth...'). "
    "REPORTING VERBS (very important): never the same reporting verb in two consecutive "
    "lines, and 'says' at most once every four lines. Rotate: asks, answers, snaps, shouts, "
    "whispers, warns, promises, denies, admits, laughs, threatens, begs, mutters, fires back, "
    "presses. When the action already shows it, drop the reporting verb and just state what the "
    "person does. "
    "EMOTIONAL LANGUAGE (the dramatic edge): name the emotion when the scene shows it "
    "(furious, shattered, relieved, cold, heartbroken, stunned). Show how a person moves or "
    "speaks (trembles, storms out, goes pale, slams the door, holds back tears, freezes, "
    "screams). This is what makes the recap feel alive. "
    "PLOT TWISTS — the hook: when the story turns, make it hit hard with a short, sharp "
    "sentence ('Only then does he realize the truth.', 'In that moment, everything clicks.', "
    "'But the truth is far worse.'). Do not overuse this — one strong hit per scene is enough. "
    "SPOKEN ENGLISH FEEL: use everyday words and contractions (it's, he's, they're, "
    "gonna, wanna, kinda — but not in every line). Prefer lively verbs to noun phrases: "
    "NOT 'He engages in a conversation with her' but 'They talk'; NOT 'She makes the "
    "decision to leave' but 'She decides to leave.' No business English, no "
    "'utilize', 'commence', 'prior to', 'in order to' in every line. "
    "CONNECTORS: vary them — but, and then, suddenly, a moment later, meanwhile, "
    "out of nowhere, only then, still, yet, of all people, hardly is she inside when, "
    "in the meantime, all at once. "
    "IDIOMS: never translate Chinese or other idioms literally. Replace them with an "
    "English expression that people really say, or just say plainly what is meant. "
    "PRONOUNS: use he, she, him, her, they, them naturally so the text flows and does "
    "not repeat names all the time. If two characters of the same gender are in the scene, "
    "use the name or a role word (his mother, the boss) instead, so nobody gets mixed up. "
    "NAMES: keep Chinese and other Asian names in their Latin / Pinyin spelling, exactly "
    "the same every time, never translate them. If a name is not known, never write 'the "
    "person', 'someone' or 'a person'. Use a simple role word from the context (the mother, "
    "the father, the boss, the neighbour, the girlfriend) or a look-based word: the young "
    "man, the young woman, the old man, the older woman, the little girl — and keep the "
    "same word for the same person. "
    "NUMBERS & SYMBOLS (English): write everything as words, never digits or symbols: "
    "'50%' -> 'fifty percent', '100 dollars' -> 'a hundred dollars', '3 o'clock' -> "
    "'three o'clock', '2019' -> 'twenty nineteen', '1.5 million' -> 'one and a half "
    "million'. No abbreviations: 'for example' instead of 'e.g.', 'and so on' instead "
    "of 'etc.'. "
    "NO QUOTES, NO EMOJI, NO BRACKETS: never use quotation marks, emoji, brackets, "
    "asterisks or stage directions. The narration is spoken, not acted. Report the dialogue "
    "in third person and let the narration carry the emotion through word choice. "
    "FINAL SELF-CHECK for every line: imagine reading it aloud to a friend in a low, "
    "tense voice. If it sounds like a translation, a textbook or a robot, rewrite it the way "
    "a native English narrator of a hit drama recap channel would really say it. "
)

_STORY_ES = (
    "NARRADOR DE RECAPITULACIÓN EN ESPAÑOL (muy importante): escribe como una persona "
    "española real que le cuenta a un amigo en el café lo que pasó en una serie. NO debe sonar "
    "como una traducción, un artículo de periódico, un ensayo escolar ni un archivo de subtítulos. "
    "TIEMPO: cuenta todo el recap en presente (como cuando alguien te cuenta una película): "
    "'Xiao Li llega a casa y ve que la puerta está abierta.' Usa pretérito perfecto o "
    "pluscuamperfecto solo para cosas que pasaron antes ('Ella se lo había prometido años atrás.'). "
    "No uses el pretérito literario para narrar ('respondió él', 'replicó ella'). "
    "ESTILO INDIRECTO, COMO SE HABLA: usa una frase normal con 'que' o una coordinada: "
    "'Ella dice que ya no tiene ganas.' o 'Ella dice que ya no quiere.' "
    "NUNCA uses el subjuntivo literario rígido ('ella haya dicho', 'él sea', 'ella quiera'). "
    "VERBOS DE HABLA (muy importante): nunca repitas el mismo verbo en dos líneas seguidas, y "
    "'dice' como máximo una vez cada cuatro líneas. Rota: pregunta, responde, explica, opina, "
    "replica, grita, chilla, susurra, advierte, promete, lo niega, lo admite, se ríe, amenaza, "
    "suplica, murmura, le contesta, insiste. Cuando la acción ya lo muestra, quita el verbo de "
    "habla y solo cuenta lo que hace la persona. "
    "ESPAÑOL HABLADO Y NATURAL: usa palabras del día a día y, con moderación, partículas "
    "modales donde de verdad encajen (pues, bueno, hombre, mira, oye, vale) — no en todas las "
    "líneas. Están bien contracciones como 'pa'l', 'pal', 'dél' cuando hablas rápido, pero no "
    "en todas partes. "
    "Prefiere verbos vivos a sustantivos: NO 'Él mantiene una conversación con ella' sino "
    "'Ellos hablan'; NO 'Ella toma la decisión de irse' sino 'Ella decide irse'. "
    "Nada de español administrativo, ni 'con respecto a', 'en relación con', 'asimismo', "
    "'no obstante' en todas las líneas. "
    "VARIEDAD DE FRASES: mezcla frases cortas ('La puerta está abierta.') con medianas. No "
    "empieces todas las frases con el nombre del personaje; a veces empieza con el tiempo o "
    "el lugar ('A la mañana siguiente...', 'Poco después...', 'Afuera, en el pasillo...'). No "
    "empieces más de dos líneas seguidas con la misma palabra. "
    "CONECTORES: varíalos — y entonces, pero, de repente, poco después, mientras tanto, aun así, "
    "mientras, justo, apenas entra cuando, entretanto, de pronto, solo que, por desgracia. "
    "EXPRESIONES: nunca traduzcas refranes chinos o ingleses literalmente. Cámbialos por una "
    "expresión que un español realmente diría, o simplemente di lo que quiere decir. "
    "PRONOMBRES: usa él, ella, le, la, lo, les naturalmente para que el texto fluya y no repita "
    "los nombres todo el tiempo. Si hay dos personajes del mismo género en la escena, usa el "
    "nombre o una palabra de rol (la madre, el jefe) para que nadie se confunda. "
    "NOMBRES: mantén los nombres chinos y asiáticos en su forma latina o pinyin, exactamente "
    "igual siempre, nunca los traduzcas. Si no sabes un nombre, nunca escribas 'la persona', "
    "'alguien' o 'una persona'. Usa una palabra de rol simple según el contexto (la madre, el "
    "padre, el jefe, la vecina, la novia) o una palabra por aspecto: el joven, la joven, el "
    "anciano, la señora mayor, la niña pequeña — y mantén la misma palabra para la misma persona. "
    "NÚMEROS Y SÍMBOLOS (español): escribe todo con palabras, nunca con dígitos ni símbolos: "
    "'50%' -> 'cincuenta por ciento', '100 dólares' -> 'cien dólares', '3 de la tarde' -> "
    "'tres de la tarde', '2019' -> 'dos mil diecinueve', '1,5 millones' -> 'un millón "
    "y medio'. Sin abreviaturas: 'por ejemplo' en vez de 'p. ej.', 'etcétera' en vez de 'etc.'. "
    "EMOCIÓN: muestra los sentimientos con las palabras y el ritmo: 'Le grita', 'susurra con "
    "lágrimas en los ojos', 'sonríe con frialdad', 'no da crédito a sus oídos'. Una pregunta "
    "de un personaje debe sonar a pregunta de verdad: 'Ella le pregunta de dónde saca eso.' "
    "AUTOCORRECCIÓN FINAL para cada línea: imagina que la lees en voz alta a un amigo. Si suena "
    "a traducción, a libro de texto o a robot, reescríbela como la diría un español de verdad. "
)

def _tail(script, accent, story):
    return (
        _RECAP_MODE
        + "Keep character names consistent across all lines; " + script + " "
        + _HUMAN.replace("{accent}", accent) + _STORY_CORE + story +
        "Output exactly one line per input line in the format [number] text "
        "-- same numbers, same count, no extra comments, no headings, no blank lines, "
        "no language label.\n\n"
    )

LANG_CONF = {
    "my": {
        "name": "Burmese", "label": "မြန်မာ",
        "sample": "စာတန်းထိုး နမူနာ စာသား ဖြစ်ပါတယ်။",
        "rate": 45, "trim": TRIM,
        "edge_voices": [("my-MM-ThihaNeural", "Thiha (male)"),
                        ("my-MM-NilarNeural", "Nilar (female)")],
        "sentence_end": r"(?<=။)",
        "prompt": (
            "You are a Burmese (Myanmar) movie/drama recap narrator. Turn every numbered "
            "line below into natural spoken Burmese recap narration (calm third-person storyteller "
            "style). "
            + _tail(script="use ONLY Burmese script (Myanmar Unicode), no English/Chinese letters.",
                    accent=("Speak calm, clear, everyday Burmese like a recap narrator. Mostly end lines "
                            "with ပါတယ်။ Do not translate word by word: translate the meaning and "
                            "feeling, and turn idioms into natural Burmese. Use '၊' at breath points "
                            "and '။' at the end."),
                    story=_STORY_MY)
        ),
        "chunk_note": (
            _RECAP_NOTE +
            "Keep the same calm third-person recap voice, the same ending style (mostly ပါတယ်။), "
            "and the same name spellings as the other parts. Never use ကျွန်တော် or ဗျ. "
            "When a name is unknown never write လူတစ်ယောက် — use a role word or leave the subject out."
        ),
        "font_keys": ("padauk", "myanmar", "pyidaung", "notosansmyanmar", "myanmarpadaung"),
        "clean": r"[^\u1000-\u109f\uaa60-\uaa7f\ua9e0-\ua9ff\s0-9.,!?\"'%\-]",
        "is_mm": True,
    },
    "de": {
        "name": "German", "label": "ဂျာမနီ",
        "sample": "Dies ist ein Beispiel-Untertitel.",
        "rate": 10, "trim": TRIM_DE,
        "edge_voices": [("de-DE-KatjaNeural", "Katja (female)"),
                        ("de-DE-ConradNeural", "Conrad (male)"),
                        ("de-DE-AmalaNeural", "Amala (female)"),
                        ("de-DE-BerndNeural", "Bernd (male)")],
        "sentence_end": r"(?<=[.!?])\s+",
        "prompt": (
            "You are a native German (Deutsch) storyteller who retells TV dramas to friends in a "
            "recap video. Rewrite every numbered line below as natural, relaxed, spoken German — "
            "the way a real person would tell it, not a translation. "
            + _tail(script="use ONLY German (Latin script), no Chinese letters.",
                    accent=("Speak relaxed, everyday German as in a chat with a friend: contractions "
                            "like 'ist's', 'gibt's', 'hat's' are fine, and put a comma at every "
                            "natural breath point."),
                    story=_STORY_DE)
        ),
        "chunk_note": (
            _RECAP_NOTE +
            "Keep character names, their spelling and role words exactly the same as in the other "
            "parts. Keep the same tense (Präsens) and the same relaxed spoken German tone as in "
            "all other parts. Never use Konjunktiv I. Vary reporting verbs and sentence openings; "
            "'sagt' at most once every four lines. Use er/sie naturally, but use the name when two "
            "characters of the same gender are in the scene."
        ),
        "font_keys": ("dejavusans", "dejavu", "liberation", "notosans", "arial", "freesans"),
        "clean": r"[^A-Za-zÄÖÜäöüß0-9\s.,!?\"'%\-]",
        "is_mm": False,
    },
    "en": {
        "name": "English", "label": "အင်္ဂလိပ်",
        "sample": "This is a sample subtitle line.",
        "rate": 5, "trim": TRIM_EN,
        "edge_voices": [("en-US-AriaNeural", "Aria (female, US)"),
                        ("en-US-GuyNeural", "Guy (male, US)"),
                        ("en-US-JennyNeural", "Jenny (female, US)"),
                        ("en-GB-SoniaNeural", "Sonia (female, UK)"),
                        ("en-GB-RyanNeural", "Ryan (male, UK)"),
                        ("en-AU-NatashaNeural", "Natasha (female, AU)")],
        "sentence_end": r"(?<=[.!?])\s+",
        "prompt": (
            "You are a dramatic English drama-recap narrator for a YouTube/TikTok recap "
            "channel. Your voice is tense, gripping and cinematic — you tell the story to a "
            "viewer who cannot look away. Rewrite every numbered line below as natural, "
            "spoken, dramatic English — not a translation, not a news article. "
            + _tail(script="use ONLY English (Latin script), no Chinese letters.",
                    accent=("Speak tense, gripping, everyday English — like a movie trailer "
                            "narrator telling the story to a friend. Use contractions and put a "
                            "comma at every natural breath point."),
                    story=_STORY_EN)
        ),
        "chunk_note": (
            _RECAP_NOTE +
            "Keep character names, their spelling and role words exactly the same as in the "
            "other parts. Keep the same tense (simple present), the same DRAMATIC cinematic "
            "tone, and the same mix of short punchy lines with flowing ones. Vary reporting "
            "verbs and sentence openings; 'says' at most once every four lines. Never use "
            "quotation marks, emoji, brackets or stage directions."
        ),
        "font_keys": ("dejavusans", "dejavu", "liberation", "notosans", "arial", "freesans"),
        "clean": r"[^A-Za-z0-9\s.,!?\"'%\-]",
        "is_mm": False,
    },
    "es": {
        "name": "Spanish", "label": "စပိန်",
        "sample": "Esta es una línea de subtítulo de ejemplo.",
        "rate": 5, "trim": TRIM_ES,
        "edge_voices": [("es-ES-ElviraNeural", "Elvira (female, ES)"),
                        ("es-ES-AlvaroNeural", "Álvaro (male, ES)"),
                        ("es-MX-DaliaNeural", "Dalia (female, MX)"),
                        ("es-MX-JorgeNeural", "Jorge (male, MX)"),
                        ("es-AR-ElenaNeural", "Elena (female, AR)"),
                        ("es-AR-TomasNeural", "Tomás (male, AR)")],
        "sentence_end": r"(?<=[.!?])\s+",
        "prompt": (
            "You are a native Spanish (Español) storyteller who retells TV dramas to friends in "
            "a recap video. Rewrite every numbered line below as natural, relaxed, spoken "
            "Spanish — the way a real person would tell it, not a translation. "
            + _tail(script="use ONLY Spanish (Latin script with accents á é í ó ú ñ ü), no Chinese letters.",
                    accent=("Habla un español relajado y cotidiano, como en una charla con un amigo: "
                            "las contracciones y el habla coloquial están bien, y pon una coma en "
                            "cada punto natural de respiración."),
                    story=_STORY_ES)
        ),
        "chunk_note": (
            _RECAP_NOTE +
            "Keep character names, their spelling and role words exactly the same as in the other "
            "parts. Keep the same tense (presente) and the same relaxed spoken Spanish tone as in "
            "all other parts. Vary reporting verbs and sentence openings; 'dice' at most once "
            "every four lines. Use él/ella naturally, but use the name when two characters of the "
            "same gender are in the scene."
        ),
        "font_keys": ("dejavusans", "dejavu", "liberation", "notosans", "arial", "freesans"),
        "clean": r"[^A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9\s.,!?\"'%\-]",
        "is_mm": False,
    },
}

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
    raise ValueError(f"မသိတဲ့ ဘာသာစကား: {value} (my / de / en / es)")

@dataclass
class Settings:
    lang: str = "my"
    voice: str = ""
    rate: Optional[int] = None
    tts_workers: int = 8
    loud_lufs: float = -14.0
    voice_db: float = 0.0
    bgm: str = ""
    bgm_vol: float = 0.25
    duck: bool = True
    height: int = 720
    fps: int = 24
    preset: str = "ultrafast"
    max_speed: float = 1.25
    crop: bool = True
    flip: bool = True
    mirror_every: float = 5.0      # default 5s (hidden from UI)
    zoom: float = 1.0
    render_workers: int = 4        # default 4  (hidden from UI)
    split: bool = True
    split_min: float = 2.0
    sub: bool = True
    sub_size: float = 7.0
    sub_pos: int = 90
    sub_alpha: int = 140
    sub_sweep: bool = True
    font: str = ""
    neon: bool = True
    neon_th: int = 3
    neon_speed: float = 0.15
    punch: bool = True             # ၃ စက္ကန့်တိုင်း ရပ်ပြီး zoom in
    punch_every: float = 3.0
    punch_hold: float = 0.5
    punch_zoom: float = 1.15

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in names})

    def resolved_voice(self):
        return self.voice or LANG_CONF[self.lang]["edge_voices"][0][0]

    def resolved_rate(self):
        if self.rate is None:
            return int(LANG_CONF[self.lang].get("rate", 0))
        return int(self.rate)

    def rate_str(self):
        return f"{self.resolved_rate():+d}%"

    def mirror_phase(self, t):
        if self.mirror_every < 0:
            return False
        if self.mirror_every == 0:
            return bool(self.flip)
        return (int(t // self.mirror_every) % 2 == 0)

# ======================================================================
# utils
# ======================================================================
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
    return re.sub(LANG_CONF[lang]["clean"], "", text).strip()

def log(msg):
    print(msg, file=sys.stderr, flush=True)

class Progress:
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
# overlay (DO NOT TOUCH — blur effect preserved exactly)
# ======================================================================
SWEEP_TRAIL = np.array([185, 232, 255], np.float32)
SWEEP_BAND = np.array([235, 255, 255], np.float32)
GLASS_TINT = np.array([86, 104, 124], np.float32)
GOLD = (255, 214, 40)

def build_overlay(text, W, H, font_path, fs, pos_y, alpha=140, color=GOLD,
                  layout="raqm", is_mm=True):
    mx = int(W * 0.04)
    bw = W - 2 * mx
    pad_x, pad_y = int(fs * 0.6), int(fs * 0.5)
    lh_box = int(fs * 1.4)
    bh = min(SUB_LINES * lh_box + 2 * pad_y, H)
    max_y = max(0, H - bh)
    box_y = max(0, min(int(pos_y / 100 * max_y), max_y))
    cur_fs = fs
    font = get_font(font_path, cur_fs, layout)
    lines = wrap_text(text, font, bw - 2 * pad_x, is_mm)
    while len(lines) > SUB_LINES and cur_fs > 12:
        cur_fs = max(12, int(cur_fs * 0.92))
        font = get_font(font_path, cur_fs, layout)
        lines = wrap_text(text, font, bw - 2 * pad_x, is_mm)
    if not lines:
        return None
    radius = max(4, int(fs * 0.30))
    m = Image.new("L", (bw, bh), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, bw - 1, bh - 1], radius=radius, fill=255)
    mask = cv2.GaussianBlur(np.array(m).astype(np.float32) / 255.0, (0, 0), 0.8)
    inner = Image.new("L", (bw, bh), 0)
    ImageDraw.Draw(inner).rounded_rectangle([2, 2, bw - 3, bh - 3],
                                            radius=max(2, radius - 2), fill=255)
    inner = cv2.GaussianBlur(np.array(inner).astype(np.float32) / 255.0, (0, 0), 0.8)
    grad = np.linspace(1.0, 0.25, bh, dtype=np.float32)[:, None]
    edge = np.clip(mask - inner, 0.0, 1.0) * grad * 0.40
    sheen = (np.linspace(16.0, 0.0, bh, dtype=np.float32))[:, None, None]
    img = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    ink_img = Image.new("L", (bw, bh), 0)
    di = ImageDraw.Draw(ink_img)
    ink = (int(color[0]), int(color[1]), int(color[2]), 255)
    sw = max(2, round(cur_fs / 11))
    lh = int(cur_fs * 1.4)
    ty = (bh - len(lines) * lh) // 2
    for ln in lines:
        lx = (bw - font.getlength(ln)) / 2
        d.text((lx + 1, ty + 2), ln, font=font, fill=(0, 0, 0, 170),
               stroke_width=sw, stroke_fill=(0, 0, 0, 170))
        d.text((lx, ty), ln, font=font, fill=(0, 0, 0, 255),
               stroke_width=sw, stroke_fill=(0, 0, 0, 255))
        d.text((lx, ty), ln, font=font, fill=ink)
        di.text((lx, ty), ln, font=font, fill=255)
        ty += lh
    arr = np.array(img)
    a = arr[:, :, 3:4].astype(np.float32) / 255.0
    bgr = arr[:, :, [2, 1, 0]].astype(np.float32)
    tm = (np.array(ink_img).astype(np.float32) / 255.0)[:, :, None]
    return {"y0": box_y, "y1": box_y + bh, "x0": mx, "x1": mx + bw,
            "mask": mask[:, :, None], "edge": edge[:, :, None], "sheen": sheen,
            "tint_a": max(0.0, min(float(alpha) / 255.0, 1.0)),
            "sigma": max(3.0, fs * 0.30),
            "inv": 1.0 - a, "pre": bgr * a, "tm": tm,
            "xs": np.arange(bw, dtype=np.float32)[None, :, None]}

_BLUR_DOWN = 4

def blend_overlay(fr, ov, p=None):
    y0, y1, x0, x1 = ov["y0"], ov["y1"], ov["x0"], ov["x1"]
    base = fr[y0:y1, x0:x1]
    h, w = base.shape[:2]
    sc = _BLUR_DOWN
    small = cv2.resize(base, (max(1, w // sc), max(1, h // sc)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), max(1.0, ov["sigma"] / sc))
    blur = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    ta = ov["tint_a"]
    glass = blur * (1.0 - ta) + GLASS_TINT * ta + ov["sheen"]
    reg = base.astype(np.float32) * (1.0 - ov["mask"]) + glass * ov["mask"]
    reg = reg * (1.0 - ov["edge"]) + 255.0 * ov["edge"]
    reg = reg * ov["inv"] + ov["pre"]
    if p is not None:
        W = reg.shape[1]
        p = max(0.0, min(float(p), 1.0))
        c = (-0.12 + 1.24 * p) * W
        bw_ = 0.07 * W
        x = ov["xs"]
        band = np.exp(-(((x - c) / bw_) ** 2))
        trail = (x < c).astype(np.float32) * 0.55
        tm = ov["tm"]
        k_trail = np.clip(trail, 0.0, 1.0) * tm
        reg = reg * (1.0 - k_trail) + SWEEP_TRAIL * k_trail
        k_band = np.clip(band * 0.95, 0.0, 1.0) * tm
        reg = reg * (1.0 - k_band) + SWEEP_BAND * k_band
        reg += band * (1.0 - ov["inv"]) * 22.0
    fr[y0:y1, x0:x1] = np.clip(reg, 0, 255).astype(np.uint8)

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

_NEON_CACHE = {}

def _neon_for(W, H, th, speed):
    key = (W, H, int(th), float(speed))
    if key not in _NEON_CACHE:
        if len(_NEON_CACHE) > 4:
            _NEON_CACHE.clear()
        _NEON_CACHE[key] = Neon(W, H, th, speed)
    return _NEON_CACHE[key]

def apply_view(fr, rect, W, H, mirror, interp=cv2.INTER_LINEAR):
    x, y, w, h = rect
    f = fr[y:y + h, x:x + w]
    if (w, h) != (W, H):
        f = cv2.resize(f, (W, H), interpolation=interp)
    else:
        f = f.copy()
    if mirror:
        f = cv2.flip(f, 1)
    return f

def out_size(rect, height):
    H = int(height) // 2 * 2
    W = int(round(rect[2] * H / rect[3] / 2)) * 2
    return W, H

# ======================================================================
# crop helpers
# ======================================================================
def resolve_crop(src_w, src_h, s, auto=None):
    if not getattr(s, "crop", True):
        return (0, 0, src_w, src_h)
    if auto and auto[1] < src_h * 0.95 and auto[0] > 0 and auto[1] > 0:
        cw, ch, cx, cy = auto
        cw, ch = min(cw, src_w - cx), min(ch, src_h - cy)
        return (cx, cy, cw, ch)
    return (0, 0, src_w, src_h)

def zoom_rect(rect, zoom):
    x, y, w, h = rect
    if zoom <= 1.001:
        return rect
    nw, nh = max(2, int(w / zoom) // 2 * 2), max(2, int(h / zoom) // 2 * 2)
    return (x + (w - nw) // 2, y + (h - nh) // 2, nw, nh)

def punch_state(T, every, hold, zmax):
    """၃ စက္ကန့်တိုင်း (every) ရဲ့ နောက်ဆုံး hold စက္ကန့်မှာ frame ကို ရပ်ပြီး zoom in ဆွဲမယ်။
    return (frame ရွေးမယ့် အချိန်, zoom)။ hold ပြီးရင် zoom ကို ချက်ချင်း ၁.၀ ပြန်ကျမယ်။"""
    if every <= 0 or hold <= 0:
        return T, 1.0
    ph = T % every
    start = every - hold
    if ph < start:
        return T, 1.0
    p = (ph - start) / hold
    e = p * p * (3.0 - 2.0 * p)          # ease in-out
    return T - (ph - start), 1.0 + (zmax - 1.0) * e

def center_zoom(fr, z):
    H, W = fr.shape[:2]
    cw, ch = max(2, int(W / z)), max(2, int(H / z))
    x0, y0 = (W - cw) // 2, (H - ch) // 2
    return cv2.resize(fr[y0:y0 + ch, x0:x0 + cw], (W, H), interpolation=cv2.INTER_LINEAR)

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
_RED_HOSTS = ("xhslink.com", "xiaohongshu.com", "rednote.com", "xhs.cn")
_RED_UA = ("Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 "
           "(KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1")

def extract_url(text):
    text = (text or "").strip()
    if os.path.isfile(text):
        return text
    m = re.search(r"https?://[^\s\"'<>，。！）)]+", text)
    return m.group(0).rstrip(".,;:!?") if m else text

def _is_rednote(host):
    host = host.lower()
    return any(host == h or host.endswith("." + h) for h in _RED_HOSTS)

def _walk(o, out):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == "masterUrl" and isinstance(v, str):
                out.insert(0, v)
            elif k in ("backupUrls", "backup_urls") and isinstance(v, list):
                out.extend(x for x in v if isinstance(x, str))
            elif k == "originVideoKey" and isinstance(v, str) and v:
                out.append("https://sns-video-bd.xhscdn.com/" + v.lstrip("/"))
            else:
                _walk(v, out)
    elif isinstance(o, list):
        for v in o:
            _walk(v, out)

def _red_video_urls(html):
    urls = []
    m = re.search(r"window\.__INITIAL_STATE__\s*=\s*(\{.*?\})\s*</script>", html, re.S)
    if m:
        raw = re.sub(r"\bundefined\b", "null", m.group(1))
        try:
            _walk(json.loads(raw), urls)
        except Exception:
            pass
    if not urls:
        for u in re.findall(r'"(?:masterUrl|backupUrls?)"\s*:\s*\[?\s*"([^"]+)"', html):
            urls.append(u.encode().decode("unicode_escape", "ignore"))
        for u in re.findall(r'https?:\\?/\\?/sns-video[^"\\\s]+(?:\\/[^"\s\\]+)*', html):
            urls.append(u.replace("\\u002F", "/").replace("\\/", "/"))
    clean, seen = [], set()
    for u in urls:
        u = u.replace("\\u002F", "/").replace("\\/", "/")
        if u.startswith("//"):
            u = "https:" + u
        u = u.replace("http://", "https://", 1)
        if u.startswith("https://") and u not in seen:
            seen.add(u)
            clean.append(u)
    return clean

def _dl_direct(url, out, progress=None, headers=None):
    import requests
    with requests.get(url, stream=True, timeout=30, allow_redirects=True,
                      headers=headers or {"User-Agent": "Mozilla/5.0"}) as r:
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

def _dl_rednote(url, out, progress=None):
    import requests
    hdr = {"User-Agent": _RED_UA, "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
           "Referer": "https://www.xiaohongshu.com/"}
    r = requests.get(url, headers=hdr, timeout=30, allow_redirects=True)
    r.raise_for_status()
    urls = _red_video_urls(r.text)
    if not urls:
        raise RuntimeError("Rednote page ထဲမှာ video URL မတွေ့ပါ")
    last = None
    for u in urls[:6]:
        try:
            return _dl_direct(u, out, progress, headers=hdr)
        except Exception as ex:
            last = ex
    raise RuntimeError(f"Rednote video ဖိုင် ဆွဲမရပါ: {str(last)[:200]}")

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
        "noplaylist": True, "quiet": True, "no_warnings": True,
        "retries": 5, "fragment_retries": 5, "concurrent_fragment_downloads": 4,
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
        raise RuntimeError("Video stream မတွေ့ပါ")
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
    source = extract_url(source)
    if not re.match(r"^https?://", source, re.I):
        if not os.path.isfile(source):
            raise FileNotFoundError(f"ဖိုင် မတွေ့ပါ: {source}")
        ensure_mp4(source, final, height)
        return final
    work = tempfile.mkdtemp(prefix="dl_")
    try:
        host = urlparse(source).netloc.lower()
        methods = ["rednote", "ytdlp"] if _is_rednote(host) else ["ytdlp"]
        raw, errs = None, []
        for m in methods:
            try:
                if m == "rednote":
                    raw = _dl_rednote(source, os.path.join(work, "red.bin"), progress)
                else:
                    raw = _dl_ytdlp(source, os.path.join(work, "yt"), height, progress,
                                    cookies_file)
                break
            except Exception as ex:
                errs.append(f"{m}: {str(ex)[:300]}")
                raw = None
        if not raw:
            raise RuntimeError("Link ကနေ မဆွဲနိုင်ပါ\n" + "\n".join(errs))
        if progress:
            progress(1.0)
        ensure_mp4(raw, final, height)
        return final
    finally:
        shutil.rmtree(work, ignore_errors=True)

# ======================================================================
# transcribe
# ======================================================================
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
        raise RuntimeError(f"Video က {vdur / 60:.1f} မိနစ်ရှိတယ်။ --max-min ကျော်နေတယ်")
    progress(0.0, "Audio ထုတ်နေတယ်")
    wav = os.path.join(workdir, "audio16k.wav")
    sh(["ffmpeg", "-y", "-err_detect", "ignore_err", "-fflags", "+discardcorrupt",
        "-i", video, "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav])
    with wave.open(wav) as wf:
        audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    progress(0.08, "Whisper model ဖွင့်နေတယ်")
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
def build_prompt(units, lang="my"):
    lines = "\n".join(f"[{i}] {u['src']}" for i, u in enumerate(units))
    return LANG_CONF[lang]["prompt"] + lines

def build_prompt_chunks(units, lang="my", size=CHUNK):
    conf = LANG_CONF[lang]
    head = conf["prompt"]
    total = (len(units) + size - 1) // size
    out = []
    for k in range(total):
        lo, hi = k * size, min((k + 1) * size, len(units))
        lines = "\n".join(f"[{i}] {units[i]['src']}" for i in range(lo, hi))
        note = (f"(PART {k + 1} of {total} of the same episode. Only lines [{lo}] to [{hi - 1}]. "
                f"Keep the SAME numbers. {conf['chunk_note']})\n\n")
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
def tts_edge(text, voice, rate, mp3, proxy=None):
    import edge_tts

    async def run():
        kw = {"proxy": proxy} if proxy else {}
        await edge_tts.Communicate(text, voice, rate=rate, **kw).save(mp3)

    asyncio.run(run())

# ======================================================================
# render
# ======================================================================
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
    cap = cv2.VideoCapture(src)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sfps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stotal = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 10 ** 9
    cap.release()

    auto = detect_crop(src) if s.crop else None
    base_rect = resolve_crop(src_w, src_h, s, auto)
    W, H = out_size(base_rect, s.height)
    rect = zoom_rect(base_rect, s.zoom)
    interp = cv2.INTER_AREA if H < rect[3] else cv2.INTER_LINEAR

    mirror_every = float(s.mirror_every)
    legacy_flip = bool(s.flip)
    seg_len = mirror_every if mirror_every > 0 else 0.0
    fps_out = int(s.fps)
    nfr = max(1, int(round(total * fps_out)))
    fs = max(16, int(W * s.sub_size / 100))
    style = {"alpha": int(s.sub_alpha), "color": GOLD, "layout": "raqm"}
    neon = _neon_for(W, H, s.neon_th, s.neon_speed) if s.neon else None

    punch_on = bool(getattr(s, "punch", False)) and float(s.punch_every) > 0
    p_every = float(s.punch_every)
    p_hold = float(s.punch_hold)
    p_zoom = float(s.punch_zoom)

    def mirror_at(t):
        if mirror_every < 0:
            return False
        if mirror_every == 0:
            return legacy_flip
        return (int(t // mirror_every) % 2 == 0)

    cpu = os.cpu_count() or 2
    target_workers = int(s.render_workers) if int(s.render_workers) > 0 else min(cpu, 6)
    target_workers = max(1, target_workers)
    raw_edges = [int(round(nfr * i / target_workers)) for i in range(target_workers + 1)]
    if seg_len > 0:
        boundaries_fr = {0, nfr}
        k = 1
        while k * seg_len * fps_out < nfr:
            boundaries_fr.add(int(round(k * seg_len * fps_out)))
            k += 1
        merged = set(raw_edges) | boundaries_fr
        edges = sorted(merged)
        min_frames = fps_out
        edges = [e for i, e in enumerate(edges)
                 if i == 0 or i == len(edges) - 1 or (e - edges[i - 1]) >= min_frames
                 or (edges[i + 1] - e) >= min_frames or (e - edges[i - 1]) < min_frames]
        edges = sorted(set(edges))
    else:
        edges = raw_edges

    nw = len(edges) - 1
    # FIX: chunk အရေအတွက် (nw) က mirror_every ကြောင့် ရာနဲ့ချီနိုင်တယ်၊
    # တစ်ပြိုင်နက် run မယ့် worker ကို render_workers နဲ့ ကန့်သတ်မယ်
    conc = max(1, min(nw, target_workers))
    counts = [0] * nw
    per_threads = max(1, cpu // conc)
    wd = os.path.dirname(os.path.abspath(out))
    tag = uuid.uuid4().hex[:6]
    parts = [os.path.join(wd, f"_part{tag}_{i:03d}.mp4") for i in range(nw)]
    abort = threading.Event()

    def worker(pi):
        if abort.is_set():
            return
        j0, j1 = edges[pi], edges[pi + 1]
        cap_ = cv2.VideoCapture(src)
        q = queue.Queue(maxsize=48)
        lstop = threading.Event()
        chunk_mirror = mirror_at(j0 / fps_out)

        def put(x):
            while not lstop.is_set() and not abort.is_set():
                try:
                    q.put(x, timeout=0.5)
                    return True
                except queue.Full:
                    continue
            return False

        def producer():
            cur, last = -1, None
            ui = 0
            T0 = j0 / fps_out
            while ui < len(plan) - 1 and T0 >= plan[ui + 1][0]:
                ui += 1
            try:
                for j in range(j0, j1):
                    if lstop.is_set() or abort.is_set():
                        return
                    T = j / fps_out
                    while ui < len(plan) - 1 and T >= plan[ui + 1][0]:
                        ui += 1
                    cs, _d, s0, spd = plan[ui]
                    if punch_on:
                        Te, pz = punch_state(T, p_every, p_hold, p_zoom)
                        Te = max(Te, cs)      # freeze ဇာတ်ခန်းနယ်မကျော်စေ
                    else:
                        Te, pz = T, 1.0
                    idx = min(max(int((s0 + (Te - cs) * spd) * sfps + 0.5), 0), stotal - 1)
                    if idx > cur + 90 or idx < cur:
                        cap_.set(cv2.CAP_PROP_POS_FRAMES, idx)
                        cur = idx - 1
                    new = False
                    while cur < idx:
                        if not cap_.grab():
                            break
                        cur += 1
                        new = True
                    if new:
                        ok, f = cap_.retrieve()
                        if ok:
                            last = apply_view(f, rect, W, H, chunk_mirror, interp)
                    if last is None:
                        return
                    out_fr = center_zoom(last, pz) if pz > 1.001 else last.copy()
                    if not put(out_fr):
                        return
            finally:
                while not lstop.is_set():
                    try:
                        q.put(None, timeout=0.5)
                        break
                    except queue.Full:
                        continue

        log_path = parts[pi] + ".log"
        log_fh = open(log_path, "wb")
        cmd = ["ffmpeg", "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{W}x{H}", "-r", str(fps_out), "-i", "-", "-an",
               "-c:v", "libx264", "-preset", s.preset, "-crf", "27",
               "-tune", "zerolatency",
               "-x264-params", "ref=1:bframes=0:scenecut=0:rc-lookahead=10:threads=" + str(per_threads),
               "-pix_fmt", "yuv420p", "-threads", str(per_threads), parts[pi]]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                stderr=log_fh)
        th = threading.Thread(target=producer, daemon=True)
        th.start()
        t_start = j0 / fps_out
        k, cur_k, cur_ov, j = 0, -1, None, j0
        try:
            while True:
                try:
                    fr = q.get(timeout=0.5)
                except queue.Empty:
                    if abort.is_set():
                        break
                    continue
                if fr is None:
                    break
                t = j / fps_out
                if neon:
                    neon.apply(fr, t)
                if subs and font_path:
                    if j == j0:
                        while k < len(subs) and subs[k][1] < t_start:
                            k += 1
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
                counts[pi] = j - j0
        except BrokenPipeError:
            pass
        finally:
            lstop.set()
            try:
                while True:
                    q.get_nowait()
            except queue.Empty:
                pass
            th.join(timeout=5)
            cap_.release()
            try:
                proc.stdin.close()
            except Exception:
                pass
            try:
                proc.wait()
            except Exception:
                pass
            log_fh.close()

        if abort.is_set():
            return
        if proc.returncode != 0 or j != j1:
            with open(log_path, "rb") as f:
                tail = f.read()[-800:].decode("utf-8", "ignore")
            abort.set()
            raise RuntimeError(f"render chunk {pi}: ffmpeg fail\n" + tail)
        try:
            os.remove(log_path)
        except OSError:
            pass

    lst = os.path.join(wd, f"_parts{tag}.txt")
    try:
        with ThreadPoolExecutor(max_workers=conc) as pool:
            futs = [pool.submit(worker, i) for i in range(nw)]
            while not all(f.done() for f in futs):
                time.sleep(0.25)
                if progress:
                    progress(min(sum(counts) / nfr, 0.97) * 0.95, "Video render")
                if any(f.done() and f.exception() for f in futs):
                    abort.set()
            for f in futs:
                f.result()
        if abort.is_set():
            raise RuntimeError("render ရပ်သွားတယ် (chunk error)")
        if progress:
            progress(0.96, "Video + အသံ ပေါင်းနေတယ်")
        with open(lst, "w") as fh:
            for p_ in parts:
                fh.write(f"file '{p_}'\n")
        r = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst,
             "-i", audio, "-map", "0:v", "-map", "1:a:0", "-c:v", "copy",
             "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
             "-t", f"{total:.3f}", out],
            capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("render: mux fail\n" + r.stderr[-800:])
    finally:
        for p_ in parts + [lst]:
            for f_ in (p_, p_ + ".log"):
                if os.path.exists(f_):
                    try:
                        os.remove(f_)
                    except OSError:
                        pass
    if progress:
        progress(1.0, "Video render")
    return out

# ---- splitter
def split_plan(total, seg_secs):
    total, seg = float(total), float(seg_secs)
    if seg <= 0 or total <= seg:
        return [(0.0, total)]
    n = max(1, int(total // seg))
    starts = [i * seg for i in range(n)]
    plan = []
    for i, st in enumerate(starts):
        end = total if i == n - 1 else starts[i + 1]
        plan.append((st, end - st))
    return plan

def split_video(src, out_prefix, seg_min=2.0, progress=None):
    total = probe_duration(src)
    plan = split_plan(total, float(seg_min) * 60.0)
    if len(plan) <= 1:
        return [src]
    outs, n = [], len(plan)
    for k, (st, dur) in enumerate(plan):
        out = f"{out_prefix}_part{k + 1:02d}.mp4"
        sh(["ffmpeg", "-y", "-ss", f"{st:.3f}", "-i", src, "-t", f"{dur:.3f}",
            "-c", "copy", "-avoid_negative_ts", "make_zero",
            "-movflags", "+faststart", out])
        outs.append(out)
        if progress:
            progress((k + 1) / n, f"အပိုင်းခွဲနေတယ် {k + 1}/{n}")
    return outs

# ---- preview
def _preview_clip(src, clip_path, height, crop):
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
    mirror = s.mirror_phase(t)
    fr = zoom_flip(base.copy(), s.zoom, mirror)
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
class Project:
    FILE = "project.json"

    def __init__(self, path):
        self.path = os.path.abspath(path)
        self.data = {"source": "", "vdur": 0.0, "units": [], "timings": {}, "settings": {}}

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
            raise FileNotFoundError(f"Project မတွေ့ပါ: {p.path}")
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

def _synthesize(proj, my_list, s, progress, proxy):
    wd = proj.path
    n = len(my_list)
    voice = s.resolved_voice()
    trim_filter = LANG_CONF[s.lang].get("trim", TRIM)
    sig_base = f"edge|{voice}|{s.rate_str()}|noref|{trim_filter}"

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
        raw_audio = os.path.join(wd, f"a{i:04d}_raw.mp3")
        for attempt in range(4):
            try:
                tts_edge(my_list[i], voice, s.rate_str(), raw_audio, proxy)
                break
            except Exception as ex:
                if attempt == 3:
                    raise RuntimeError(f"Edge TTS မအောင်မြင်ပါ (#{i}): {ex}\n{my_list[i]}")
                time.sleep(1.5)
        sh(["ffmpeg", "-y", "-i", raw_audio, "-af", trim_filter,
            "-ar", "44100", "-ac", "1", wav])
        with open(keyf, "w") as kf:
            kf.write(key)
        return i, wav_dur(wav)

    durs = [0.0] * n
    done = 0
    progress(0.0, f"အသံ ထုတ်နေတယ် 0/{n}")
    with ThreadPoolExecutor(max_workers=max(1, s.tts_workers)) as pool:
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

_AFMT = "aresample=44100,aformat=sample_fmts=fltp:channel_layouts=stereo"

def mix_audio(voice, wd, s, total):
    out = os.path.join(wd, "mix.wav")
    gain = f"loudnorm=I={s.loud_lufs:g}:TP=-1.5:LRA=11,volume={s.voice_db:g}dB"
    bgm = s.bgm if (s.bgm and os.path.isfile(s.bgm)) else ""
    if s.bgm and not bgm:
        log(f"⚠️ BGM ဖိုင် မတွေ့ပါ: {s.bgm} — BGM မပါဘဲ ဆက်သွားမယ်")
    if not bgm:
        fc = f"[0:a]{_AFMT},{gain},alimiter=limit=0.97[o]"
        sh(["ffmpeg", "-y", "-i", voice, "-filter_complex", fc, "-map", "[o]",
            "-t", f"{total:.3f}", "-ar", "44100", "-c:a", "pcm_s16le", out])
        return out
    bg = f"[1:a]{_AFMT},volume={s.bgm_vol:g},atrim=0:{total:.3f}[bg]"

    def graph(mix_opts, tail):
        if s.duck:
            return (f"[0:a]{_AFMT},{gain},asplit=2[v][sc];{bg};"
                    f"[bg][sc]sidechaincompress=threshold=0.02:ratio=10:attack=30:release=600[d];"
                    f"[v][d]amix=inputs=2:duration=first:dropout_transition=0{mix_opts}{tail}[o]")
        return (f"[0:a]{_AFMT},{gain}[v];{bg};"
                f"[v][bg]amix=inputs=2:duration=first:dropout_transition=0{mix_opts}{tail}[o]")

    def run(fc):
        sh(["ffmpeg", "-y", "-i", voice, "-stream_loop", "-1", "-i", bgm,
            "-filter_complex", fc, "-map", "[o]", "-t", f"{total:.3f}",
            "-ar", "44100", "-c:a", "pcm_s16le", out])

    try:
        run(graph(":normalize=0", ",alimiter=limit=0.97"))
    except RuntimeError:
        run(graph("", ",volume=2,alimiter=limit=0.97"))
    return out

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

def run_render(proj, s, proxy=None, progress=None):
    progress = progress or Progress()
    lang = s.lang
    conf = LANG_CONF[lang]
    units = proj.units
    if not units:
        raise RuntimeError("units မရှိသေးပါ")
    my_list = [clean_text(u["my"], lang) for u in units]
    empty = [i for i, t in enumerate(my_list) if not t.strip()]
    if empty:
        raise RuntimeError(f"စာသား မဖြည့်ရသေးတဲ့ စာကြောင်း: {empty[:20]}")
    font = ""
    if s.sub:
        font = find_font(lang, s.font)
        if not font:
            log(f"⚠️ {conf['label']} font မတွေ့ပါ — စာတန်းမပါဘဲ render လုပ်မယ်")
        elif conf["is_mm"] and not has_raqm():
            log("⚠️ libraqm မရှိလို့ မြန်မာစာ ပုံစံ မမှန်နိုင်ပါ")
    T = {}
    t_all = time.time()
    t_a = time.time()
    durs = _synthesize(proj, my_list, s, progress.sub(0.0, 0.3), proxy)
    T["TTS"] = time.time() - t_a
    t_a = time.time()
    allwav = _concat_audio(proj.path, len(units))
    T["audio concat"] = time.time() - t_a
    vdur = proj.data["vdur"]
    plan, total = _build_plan(units, durs, vdur, s.max_speed)
    subs = _build_subs(my_list, durs, lang) if (s.sub and font) else []
    t_a = time.time()
    mixed = mix_audio(allwav, proj.path, s, total)
    T["audio mix"] = time.time() - t_a
    t_a = time.time()
    render_final(proj.video, mixed, proj.final, plan, total, s, subs, font,
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
SUP = argparse.SUPPRESS

def add_style_args(p):
    g = p.add_argument_group("voice / language")
    g.add_argument("--lang", type=resolve_lang, default=SUP, help="my | de | en | es")
    g.add_argument("--voice", default=SUP)
    g.add_argument("--rate", type=int, default=SUP)
    g.add_argument("--workers", dest="tts_workers", type=int, default=SUP)
    g = p.add_argument_group("audio mix")
    g.add_argument("--loud", dest="loud_lufs", type=float, default=SUP)
    g.add_argument("--voice-db", type=float, default=SUP)
    g.add_argument("--bgm", default=SUP)
    g.add_argument("--bgm-vol", type=float, default=SUP)
    g.add_argument("--no-duck", dest="duck", action="store_false", default=SUP)
    g = p.add_argument_group("video")
    g.add_argument("--height", type=int, choices=[360, 480, 720, 1080], default=SUP)
    g.add_argument("--fps", type=int, choices=[20, 24, 25, 30], default=SUP)
    g.add_argument("--preset", choices=["ultrafast", "superfast", "veryfast"], default=SUP)
    g.add_argument("--max-speed", type=float, default=SUP)
    g.add_argument("--zoom", type=float, default=SUP)
    g.add_argument("--render-workers", type=int, default=SUP)
    g.add_argument("--mirror-every", type=float, default=SUP,
                   help="Mirror ကို ဘယ်နှစ်စက္ကန့်တစ်ခါ လှန်မလဲ · 0 = အမြဲ · -1 = မလုပ်ဘူး (default 5)")
    g.add_argument("--no-flip", dest="flip", action="store_false", default=SUP)
    g.add_argument("--no-crop", dest="crop", action="store_false", default=SUP)
    g.add_argument("--split", dest="split_min", type=float, default=SUP)
    g.add_argument("--no-split", dest="split", action="store_false", default=SUP)
    g = p.add_argument_group("subtitles / neon")
    g.add_argument("--no-sub", dest="sub", action="store_false", default=SUP)
    g.add_argument("--no-sweep", dest="sub_sweep", action="store_false", default=SUP)
    g.add_argument("--sub-size", type=float, default=SUP)
    g.add_argument("--sub-pos", type=int, default=SUP)
    g.add_argument("--font", default=SUP)
    g.add_argument("--no-neon", dest="neon", action="store_false", default=SUP)
    g.add_argument("--neon-th", type=int, default=SUP)
    g.add_argument("--neon-speed", type=float, default=SUP)
    g = p.add_argument_group("punch zoom")
    g.add_argument("--no-punch", dest="punch", action="store_false", default=SUP)
    g.add_argument("--punch-every", type=float, default=SUP)
    g.add_argument("--punch-hold", type=float, default=SUP)
    g.add_argument("--punch-zoom", type=float, default=SUP)

def add_whisper_args(p):
    g = p.add_argument_group("whisper")
    g.add_argument("--model", choices=["tiny", "base", "small"], default="base")
    g.add_argument("--src-lang", default="auto",
                   help="auto | en | zh | ko | ja | hi | th | vi | id | es | fr | de | ru | ...")
    g.add_argument("--no-batch", action="store_true")
    g.add_argument("--max-min", type=int, default=0)

def settings_for(proj, args):
    s = proj.settings
    for k, v in vars(args).items():
        if hasattr(s, k):
            setattr(s, k, v)
    if s.rate is not None:
        s.rate = max(-50, min(100, int(s.rate)))
    s.max_speed = max(1.0, min(2.0, s.max_speed))
    s.zoom = max(1.0, min(1.25, s.zoom))
    s.sub_size = max(3.0, min(12.0, s.sub_size))
    s.sub_pos = max(0, min(100, s.sub_pos))
    s.voice_db = max(-10.0, min(15.0, float(s.voice_db)))
    s.loud_lufs = max(-24.0, min(-8.0, float(s.loud_lufs)))
    s.render_workers = max(0, min(16, int(s.render_workers)))
    if s.mirror_every is None:
        s.mirror_every = 5.0
    s.mirror_every = float(s.mirror_every)
    s.punch_every = max(1.0, float(s.punch_every))
    s.punch_hold = max(0.1, min(float(s.punch_hold), s.punch_every - 0.5))
    s.punch_zoom = max(1.02, min(1.4, float(s.punch_zoom)))
    s.bgm_vol = max(0.0, min(1.0, float(s.bgm_vol)))
    s.split_min = max(0.0, float(s.split_min))
    if s.voice:
        valid = [v for v, _ in LANG_CONF[s.lang]["edge_voices"]]
        if s.voice not in valid:
            log(f"⚠️ voice '{s.voice}' က {s.lang} စာရင်းထဲမှာ မပါဘူး — default သုံးမယ်")
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
    print(f"✅ {info}\n   Prompt: {proj.prompt_file}  (+ prompt_part01..{nparts:02d}.txt)")
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
    res = run_render(proj, s, os.environ.get("EDGE_TTS_PROXY") or None, prog)
    prog.done()
    out = args.output
    if out:
        shutil.copy(res["final"], out)
    else:
        out = res["final"]
    mb = os.path.getsize(out) / 1e6
    print(f"\n✅ ပြီးပြီ: {out}  ({mb:.1f} MB)")
    print(f"   {res['new_dur']:.1f}s (မူရင်း {res['old_dur']:.1f}s)")
    if s.split and s.split_min and s.split_min > 0:
        prefix = os.path.splitext(out)[0]
        for old in glob.glob(prefix + "_part*.mp4"):
            try:
                os.remove(old)
            except OSError:
                pass
        parts = split_video(out, prefix, s.split_min)
        if len(parts) > 1:
            print(f"✂️ {len(parts)} ပိုင်း ခွဲပြီး:")
            for p_ in parts:
                print(f"   {p_}  ({fmt_t(probe_duration(p_))})")
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
    args.dir = args.dir or os.path.join("work", "recap")
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
        print(f"\n📋 Prompt: {proj.prompt_file}")
        if not sys.stdin.isatty():
            sys.exit(f"\n⏸️ translated ဖိုင်မရှိလို့ ရပ်လိုက်တယ်")
        input("   ③ ပြီးရင် Enter နှိပ်ပါ ▶ ")
    if not cmd_apply(args, proj, src):
        sys.exit("❌ စာကြောင်း မပြည့်သေးပါ")
    cmd_render(args, proj)

def cmd_ui(args):
    launch_ui()

def build_parser():
    p = argparse.ArgumentParser(prog="recap.py",
                                description="Recap video dubber")
    p.add_argument("--version", action="version", version=f"recap-cli {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("init"); a.add_argument("source")
    a.add_argument("-d", "--dir", required=True)
    a.add_argument("--height", type=int, default=720); a.add_argument("--cookies")
    a.set_defaults(fn=cmd_init)
    a = sub.add_parser("transcribe"); a.add_argument("dir")
    a.add_argument("--lang", type=resolve_lang, default=None); add_whisper_args(a)
    a.set_defaults(fn=cmd_transcribe)
    a = sub.add_parser("prompt"); a.add_argument("dir")
    a.add_argument("--lang", type=resolve_lang, default=None)
    a.add_argument("--print", action="store_true"); a.set_defaults(fn=cmd_prompt)
    a = sub.add_parser("apply"); a.add_argument("dir")
    a.add_argument("file", nargs="?"); a.add_argument("--lang", type=resolve_lang, default=None)
    a.set_defaults(fn=cmd_apply)
    a = sub.add_parser("render"); a.add_argument("dir"); a.add_argument("-o", "--output")
    add_style_args(a); a.set_defaults(fn=cmd_render)
    a = sub.add_parser("preview"); a.add_argument("dir"); a.add_argument("-o", "--output")
    a.add_argument("--at", type=float, default=0.0); add_style_args(a); a.set_defaults(fn=cmd_preview)
    a = sub.add_parser("run"); a.add_argument("source")
    a.add_argument("-d", "--dir"); a.add_argument("--translated")
    a.add_argument("-o", "--output"); a.add_argument("--cookies")
    add_whisper_args(a); add_style_args(a); a.set_defaults(fn=cmd_run)
    a = sub.add_parser("ui"); a.set_defaults(fn=cmd_ui)
    return p

# ======================================================================
# Web UI (Gradio) — DARK MODE ONLY
# ======================================================================
def launch_ui():
    import gradio as gr

    BASE = os.path.join(os.getcwd(), "work_ui")
    os.makedirs(BASE, exist_ok=True)
    PORT = int(os.environ.get("PORT", "7860"))
    LANGS = {"မြန်မာ": "my", "ဂျာမနီ": "de", "အင်္ဂလိပ်": "en", "စပိန်": "es"}

    def voices_for(code):
        return [v for v, _ in LANG_CONF[code]["edge_voices"]]

    def _cb(progress):
        return lambda frac, label=None: progress(frac, desc=label or "")

    def _path(f):
        if not f:
            return ""
        return f if isinstance(f, str) else getattr(f, "name", "")

    def _source(url):
        return extract_url(url or "")

    def _pbar(frac, width=20):
        frac = max(0.0, min(float(frac), 1.0))
        fill = int(width * frac)
        return "█" * fill + "░" * (width - fill)

    def _ensure_video(url, height, cb):
        source = _source(url)
        if not source:
            raise gr.Error("Video link ထည့်ပါ")
        key = source
        if os.path.isfile(source):
            key += f"|{os.path.getsize(source)}"
        pdir = os.path.join(BASE, hashlib.md5(key.encode()).hexdigest()[:10])
        dl_prog = Progress(0.0, 1.0, parent=lambda f, l=None: cb(f, "Video ယူနေတယ်..."))
        proj = init_project(pdir, source, int(height), os.environ.get("YT_COOKIES"), dl_prog)
        return pdir, proj

    dark_head = """
<script>
(function () {
  function forceDark() {
    var r = document.documentElement;
    if (!r.classList.contains('dark')) r.classList.add('dark');
    if (document.body && !document.body.classList.contains('dark')) document.body.classList.add('dark');
  }
  forceDark();
  new MutationObserver(forceDark).observe(document.documentElement, {attributes: true, attributeFilter: ['class']});
  document.addEventListener('DOMContentLoaded', function () {
    forceDark();
    new MutationObserver(forceDark).observe(document.body, {attributes: true, attributeFilter: ['class']});
  });
})();
</script>
"""
    ui_kw = {
        "title": "Recap → my / de / en / es",
        "theme": gr.themes.Base(primary_hue="indigo", secondary_hue="purple", neutral_hue="slate"),
        "css": ":root, html, body, gradio-app { color-scheme: dark; }"
               " body, gradio-app, .gradio-container { background: #0b0f19 !important; }"
               " footer {display:none !important;}",
        "js": "() => { document.documentElement.classList.add('dark'); document.body.classList.add('dark'); }",
        "head": dark_head,
    }
    try:
        bp = set(inspect.signature(gr.Blocks.__init__).parameters)
    except Exception:
        bp = set(ui_kw)
    blocks_kw = {k: v for k, v in ui_kw.items() if k in bp}
    launch_kw = {k: v for k, v in ui_kw.items() if k not in bp}

    FETCH = {}

    def start_fetch(url, height):
        if not _source(url):
            raise gr.Error("Video link ထည့်ပါ")
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
                st["res"] = _ensure_video(url, height, cb)
            except BaseException as ex:
                st["err"] = ex
            finally:
                st["done"] = True

        threading.Thread(target=work, daemon=True).start()
        return jid, "⏳ စတင်နေတယ်...", gr.Timer(active=True)

    def _preview(pdir, lang_name, height, zoom, crop, flip, sub, sub_size,
                 sub_pos, sweep, pv_t, strict):
     if not pdir or not os.path.exists(os.path.join(pdir, "src.mp4")):
            if strict:
                raise gr.Error("Video link ထည့်ပြီး ⬇️ အရင်နှိပ်ပါ")
            return gr.skip()
        code = LANGS[lang_name]
        try:
            proj = Project.load(pdir)
            s = proj.settings
            s.lang, s.height, s.zoom = code, int(height), float(zoom)
            s.crop, s.flip = bool(crop), bool(flip)
            s.mirror_every = 5.0   # hidden default
            s.sub = bool(sub)
            s.sub_size, s.sub_pos, s.sub_sweep = float(sub_size), int(sub_pos), bool(sweep)
            s.render_workers = 4   # hidden default
            font = find_font(code, s.font) if s.sub else None
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

    def preview_auto(*a):
        return _preview(*a, False)

    def poll_fetch(jid, *pv):
        st = FETCH.get(jid) if jid else None
        if st is None:
            return gr.skip(), gr.skip(), gr.skip(), gr.Timer(active=False)
        elapsed = time.time() - st["t0"]
        if not st["done"]:
            if st["f"] >= 1.0:
                msg = "🔄 MP4 ပြင်ဆင်နေတယ်"
            else:
                msg = f"⬇️ {st['label']} {int(st['f'] * 100)}%"
            return gr.skip(), f"{msg} · {elapsed:.0f}s", gr.skip(), gr.skip()
        FETCH.pop(jid, None)
        if st["err"] is not None:
            return "", f"❌ {str(st['err'])[:500]}", gr.skip(), gr.Timer(active=False)
        pdir, proj = st["res"]
        mb = os.path.getsize(proj.video) / 1e6
        img = _preview(pdir, *pv, False)
        return (pdir, f"✅ Video ရပြီ ({mb:.1f} MB) · အောက်က Preview မှာ ချိန်ပြီး 1️⃣ နှိပ်ပါ",
                img, gr.Timer(active=False))

    TRANS = {}

    def start_step1(url, lang_name, model, src_lang, max_min, height):
        if not _source(url):
            raise gr.Error("Video link ထည့်ပါ")
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
                pdir, proj = _ensure_video(url, height, lambda f, l=None: cb(f * 0.2, l))
                tr = Progress(0.2, 1.0, parent=lambda f, l=None: cb(f, l))
                info = run_transcribe(proj, model, src_lang, True, int(max_min), code, tr)
                st["res"] = (pdir, info, build_prompt_chunks(proj.units, code))
            except BaseException as ex:
                st["err"] = ex
            finally:
                st["done"] = True

        threading.Thread(target=work, daemon=True).start()
        return jid, "⏳ စတင်နေတယ်... 0%", gr.Timer(active=True), gr.update(interactive=False)

    def poll_step1(jid):
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
        return (pdir, f"✅ 100% · {info} · Prompt {len(chunks)} ပိုင်း · {elapsed:.0f}s",
                chunks, {}, gr.Timer(active=False), gr.update(interactive=True))

    def step2(pdir, translated, parts, lang_name, voice, rate, workers,
              bgm_file, bgm_vol, voice_db, duck,
              height, fps, preset, max_speed, crop, flip, zoom,
              sub, sub_size, sub_pos, sweep, punch,
              progress=gr.Progress()):
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
            s.lang = code
            s.voice = voice if voice in voices_for(code) else ""
            s.rate, s.tts_workers = int(rate), int(workers)
            s.bgm, s.bgm_vol = _path(bgm_file), max(0.0, min(1.0, float(bgm_vol)))
            s.voice_db, s.duck = float(voice_db), bool(duck)
            s.height, s.fps, s.preset = int(height), int(fps), preset
            s.max_speed, s.crop, s.flip, s.zoom = float(max_speed), bool(crop), bool(flip), float(zoom)

            # HIDDEN defaults
            s.mirror_every = 5.0        # 5s mirror
            s.render_workers = 4        # 4 parallel
            s.split = True
            s.split_min = 2.0           # 2 min per part

            s.sub, s.sub_size, s.sub_pos = bool(sub), float(sub_size), int(sub_pos)
            s.sub_sweep = bool(sweep)
            s.punch = bool(punch)
            s.punch_every, s.punch_hold, s.punch_zoom = 3.0, 0.5, 1.15
            # Neon: always ON with defaults
            s.neon = True
            s.neon_th = 3
            s.neon_speed = 0.15

            # FIX: အရင် run ရဲ့ final_*.mp4 (နဲ့ _partNN) ဖိုင်တွေကို render မလုပ်ခင် ရှင်းမယ်။
            # အရင်ကတော့ render ပြီးမှ ရှင်းလို့ အခုခွဲထားတဲ့ part ဖိုင်တွေပါ ပျက်သွားတာ။
            for old in glob.glob(os.path.join(pdir, "final_*.mp4")):
                try:
                    os.remove(old)
                except OSError:
                    pass

            prog = Progress(0.0, 1.0, parent=_cb(progress))
            res = run_render(proj, s, os.environ.get("EDGE_TTS_PROXY") or None, prog)

            # auto split
            out = os.path.join(pdir, f"final_{code}_{int(time.time())}.mp4")
            shutil.copy(res["final"], out)
            files = [out]
            split_note = ""
            if s.split and s.split_min > 0:
                parts_out = split_video(out, os.path.splitext(out)[0], s.split_min)
                if len(parts_out) > 1:
                    files = parts_out
                    durs = [probe_duration(p) for p in parts_out]
                    split_note = (f"\n\n✂️ {len(parts_out)} ပိုင်း: "
                                  + " · ".join(f"P{k + 1} {fmt_t(d)}" for k, d in enumerate(durs)))
                else:
                    split_note = (f"\n\n✂️ {2 * s.split_min:g} မိနစ်ထက် မရှည်လို့ မခွဲပါ "
                                  f"(ခွဲမယ့် အပိုင်းက {s.split_min:g} မိနစ်စီ)")
        except gr.Error:
            raise
        except Exception as ex:
            raise gr.Error(str(ex)[:600])

        mb = os.path.getsize(out) / 1e6
        return (out, files,
                f"✅ ပြီးပြီ · {res['new_dur']:.1f}s (မူရင်း {res['old_dur']:.1f}s) · {mb:.1f} MB"
                + split_note)

    def reset_all(pdir):
        if pdir and os.path.isdir(pdir) and os.path.abspath(pdir).startswith(BASE):
            shutil.rmtree(pdir, ignore_errors=True)
        return "", "", "", [], {}, "", None, None, None, "🗑️ ရှင်းပြီး"

    def make_setter(lo, hi):
        def f(text, parts):
            parts = dict(parts or {})
            text = text or ""
            parts[lo] = text
            if not text.strip():
                return parts, "⏳ မဖြည့်သေး"
            got, missing = check_part(text, lo, hi)
            if missing:
                return parts, f"⚠️ {len(got)}/{hi - lo} · မပါတာ: {missing[:10]}"
            return parts, f"✅ {hi - lo}/{hi - lo}"
        return f

    def on_lang_change(l):
        code = LANGS[l]
        v = voices_for(code)
        return gr.update(choices=v, value=v[0]), gr.update(value=LANG_CONF[code]["rate"])

    with gr.Blocks(**blocks_kw) as demo:
        gr.Markdown(f"# 🎙️ Recap → my / de / en / es  `v{__version__}`\n"
                    f"Whisper → Gemini → Edge TTS → Smart Sync → စာတန်း + Neon · Mirror=5s · Render=4 · Split=2min")
        pdir_state = gr.State("")
        height = gr.State(720)
        fps = gr.State(24)
        preset = gr.State("ultrafast")
        crop = gr.State(True)
        model = gr.State("base")
        src_lang = gr.State("auto")
        max_min = gr.State(0)
        translated = gr.State("")

        with gr.Accordion("🔊 ဘာသာစကား + အသံ", open=True):
            lang = gr.Radio(list(LANGS), value="မြန်မာ", label="ဘာသာစကား")
            voice = gr.Dropdown(voices_for("my"), value=voices_for("my")[0], label="Edge voice")
            with gr.Row():
                rate = gr.Slider(-50, 100, value=LANG_CONF["my"]["rate"], step=5,
                                 label="Edge speed %")
                workers = gr.Slider(1, 10, value=8, step=1, label="TTS တစ်ပြိုင်နက်")
            max_speed = gr.Radio([("1.25× (သဘာဝကျ၊ ဇာတ်ခန်း ပိုပျောက်)", 1.25),
                                  ("1.5× (ဇာတ်ခန်း ပိုပါ၊ ပိုမြန်)", 1.5)],
                                 value=1.5, label="Video အမြန်ဆုံး နှုန်း")
            lang.change(on_lang_change, lang, [voice, rate])

        with gr.Accordion("🎵 Background music + Voice အသံ", open=False):
            bgm_file = gr.File(label="🎵 Background music (optional)",
                               type="filepath",
                               file_types=[".mp3", ".wav", ".m4a", ".ogg", ".aac", ".flac"])
            with gr.Row():
                bgm_vol = gr.Slider(0.0, 1.0, value=0.25, step=0.05, label="BGM အသံ")
                voice_db = gr.Slider(-6, 12, value=0, step=1, label="Voice အပို gain (dB)")
                duck = gr.Checkbox(True, label="🦆 Ducking")

        gr.Markdown("### 📥 Video")
        url = gr.Textbox(label="Video link (YouTube / TikTok / Rednote) — paste ပြီး Enter",
                         placeholder="https://...")
        b_fetch = gr.Button("⬇️ Video ဆွဲ + Preview")
        status_fetch = gr.Markdown()

        gr.Markdown("### 🎨 Preview")
        pv_img = gr.Image(label="Preview (ရလဒ်)", type="filepath", interactive=False)
        pv_t = gr.Slider(0.0, PREVIEW_SECS, value=0.0, step=0.1,
                         label=f"⏱️ Preview အချိန် (0 – {PREVIEW_SECS:.0f} စက္ကန့်)")

        # Only subtitles in UI (mirror / render workers hidden)
        with gr.Accordion("🎛️ စာတန်း", open=False):
            with gr.Row():
                sub = gr.Checkbox(True, label="စာတန်း")
                sweep = gr.Checkbox(True, label="🎵 Music sweep")
                flip = gr.Checkbox(True, label="Mirror")
                punch = gr.Checkbox(True, label="🔍 ၃ စက္ကန့်တိုင်း ရပ်ပြီး Zoom")
            sub_size = gr.Slider(3.0, 12.0, value=7.0, step=0.5, label="စာအရွယ် (% width)")
            sub_pos = gr.Slider(0, 100, value=90, step=1, label="စာတန်း နေရာ (0=အပေါ် 100=အောက်)")
            zoom = gr.Slider(1.0, 1.25, value=1.0, step=0.01, label="Zoom")

        pv_in = [pdir_state, lang, height, zoom, crop, flip, sub, sub_size, sub_pos, sweep, pv_t]
        fetch_job = gr.State("")
        fetch_timer = gr.Timer(1.0, active=False)
        for trig in (url.submit, b_fetch.click):
            trig(start_fetch, [url, height], [fetch_job, status_fetch, fetch_timer])
        fetch_timer.tick(poll_fetch, [fetch_job] + pv_in[1:],
                         [pdir_state, status_fetch, pv_img, fetch_timer],
                         show_progress="hidden")
        for c in (sub_size, sub_pos, zoom, sub, sweep, flip, lang, pv_t):
            c.change(preview_auto, pv_in, pv_img,
                     trigger_mode="always_last", show_progress="hidden", queue=True)

        gr.Markdown("### 1️⃣ စာသားထုတ်")
        b1 = gr.Button("1️⃣ စာသားထုတ် (Whisper)", variant="primary")
        status1 = gr.Markdown()
        trans_job = gr.State("")
        trans_timer = gr.Timer(1.0, active=False)
        chunks_state = gr.State([])
        parts_state = gr.State({})

        with gr.Accordion(f"📋 Prompt အပိုင်းများ ({CHUNK} ကြောင်းစီ)", open=False):

            @gr.render(inputs=chunks_state)
            def show_chunks(chunks):
                if not chunks:
                    gr.Markdown("_1️⃣ နှိပ်ပြီးရင် Prompt အပိုင်းတွေ ဒီမှာ ပေါ်မယ်_")
                    return
                for k, c in enumerate(chunks):
                    with gr.Accordion(f"Part {k + 1}/{len(chunks)} · [{c['lo']} – {c['hi'] - 1}]",
                                      open=False):
                        gr.Textbox(value=c["text"], lines=3, max_lines=5, interactive=False,
                                   show_copy_button=True, label="① Prompt")
                        box = gr.Textbox(label=f"② Gemini ရလဒ် Part {k + 1}", lines=3, max_lines=5)
                        st = gr.Markdown("⏳ မဖြည့်သေး")
                        box.change(make_setter(c["lo"], c["hi"]), [box, parts_state], [parts_state, st])

        gr.Markdown("### 2️⃣ အသံထုတ် + Video")
        b2 = gr.Button("2️⃣ အသံထုတ် + Video ပေါင်း", variant="primary")
        status2 = gr.Markdown()
        video = gr.Video(label="ရလဒ်", autoplay=True)
        dl = gr.File(label="⬇️ Download MP4 (အပိုင်းခွဲ ဖိုင်များ)", file_count="multiple")
        b3 = gr.Button("🗑️ အသစ်ပြန်စ")

        b1.click(start_step1, [url, lang, model, src_lang, max_min, height],
                 [trans_job, status1, trans_timer, b1])
        trans_timer.tick(poll_step1, trans_job,
                         [pdir_state, status1, chunks_state, parts_state, trans_timer, b1],
                         show_progress="hidden")
        b2.click(step2, [pdir_state, translated, parts_state, lang, voice, rate, workers,
                         bgm_file, bgm_vol, voice_db, duck,
                         height, fps, preset, max_speed, crop, flip, zoom,
                         sub, sub_size, sub_pos, sweep, punch],
                 [video, dl, status2])
        b3.click(reset_all, pdir_state,
                 [pdir_state, status_fetch, status1, chunks_state, parts_state, translated,
                  pv_img, video, dl, status2])

    ok = set(inspect.signature(demo.launch).parameters)
    lk = {k: v for k, v in launch_kw.items() if k in ok}
    demo.queue().launch(server_name="0.0.0.0", server_port=PORT, show_error=True, **lk)

def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.cmd == "run" and not hasattr(args, "height"):
        args.height = 720
    try:
        args.fn(args)
    except KeyboardInterrupt:
        sys.exit("\n⛔ ရပ်လိုက်ပါပြီ")
    except Exception as ex:
        sys.exit(f"\n❌ {ex}")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        launch_ui()
    else:
        main()
