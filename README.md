#!/usr/bin/env python3
"""
SMART CULTURAL STORYTELLER – FULL FEATURED

Includes:
- interactive CLI and scripted commands
- exam generators: mcq, tf (true/false), short (short-answer)
- exam exporters: json, md, csv, tex, (pdf optional via reportlab)
- audio TTS and audio export (pyttsx3)
- voice search using VOSK (offline) or SpeechRecognition
- VOSK model download helper
- robust atomic save/load
"""
from __future__ import annotations
import json
import os
import sys
import tempfile
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any
import random
import argparse

DATA_FILE = Path("cultural_stories.json")


# ---------------------------
# Data model & defaults
# ---------------------------
@dataclass
class Story:
    id: int
    title: str
    culture: str
    era: str
    story: str
    moral: str
    background: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Story":
        return Story(
            id=int(d["id"]),
            title=str(d.get("title", "")).strip(),
            culture=str(d.get("culture", "")).strip(),
            era=str(d.get("era", "")).strip(),
            story=str(d.get("story", "")).strip(),
            moral=str(d.get("moral", "")).strip(),
            background=str(d.get("background", "")).strip(),
        )


DEFAULT_STORIES = [
    Story(
        id=1,
        title="Tenali Raman and the Thieves",
        culture="Indian",
        era="Vijayanagara Empire",
        story=("Tenali Raman overheard thieves planning to rob his house. "
               "Instead of panicking, he cleverly spread a rumor that treasure "
               "was buried inside his home. The thieves dug everywhere and "
               "were caught by the guards."),
        moral="Intelligence and presence of mind defeat greed.",
        background=("Tenali Raman was a poet and advisor in the court of King "
                    "Krishnadevaraya, known for wisdom and wit.")
    ),
    Story(
        id=2,
        title="The Lion and the Mouse",
        culture="African / Indian Folklore",
        era="Ancient",
        story=("A lion spared a tiny mouse. Later, the lion was trapped in a net. "
               "The mouse gnawed the ropes and freed him."),
        moral="No act of kindness is ever wasted.",
        background=("This story appears in folklore worldwide and teaches humility "
                    "and gratitude.")
    )
]


# ---------------------------
# Storage helpers
# ---------------------------
def atomic_save(data: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(prefix=path.name, dir=str(path.parent))
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, str(path))
    except Exception:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise


def load_data(path: Optional[Path] = None) -> List[Story]:
    if path is None:
        path = DATA_FILE
    if not path.exists():
        atomic_save([s.to_dict() for s in DEFAULT_STORIES], path)
        return list(DEFAULT_STORIES)
    try:
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        stories = [Story.from_dict(item) for item in raw]
        ids = {s.id for s in stories}
        if len(ids) != len(stories):
            for idx, s in enumerate(stories, start=1):
                s.id = idx
            atomic_save([s.to_dict() for s in stories], path)
        return stories
    except json.JSONDecodeError:
        backup = path.with_suffix(".corrupt.json")
        try:
            shutil.move(str(path), str(backup))
            print(f"⚠️ Data file corrupted, moved to {backup}. Recreating default dataset.")
        except Exception as exc:
            print("⚠️ Failed to back up corrupted data file:", exc)
        atomic_save([s.to_dict() for s in DEFAULT_STORIES], path)
        return list(DEFAULT_STORIES)
    except Exception as exc:
        print("⚠️ Unexpected error loading data:", exc)
        return list(DEFAULT_STORIES)


def save_data(stories: List[Story], path: Optional[Path] = None) -> None:
    if path is None:
        path = DATA_FILE
    atomic_save([s.to_dict() for s in stories], path)


# ---------------------------
# Utilities
# ---------------------------
def next_id(stories: List[Story]) -> int:
    if not stories:
        return 1
    return max(s.id for s in stories) + 1


def find_story_by_id(stories: List[Story], sid: int) -> Optional[Story]:
    return next((s for s in stories if s.id == sid), None)


def print_story_brief(story: Story) -> None:
    print(f"{story.id}. {story.title} ({story.culture} — {story.era})")


def list_stories(stories: List[Story]) -> None:
    if not stories:
        print("\n📚 No stories available.")
        return
    print("\n📚 AVAILABLE STORIES")
    for s in stories:
        print_story_brief(s)


# ---------------------------
# Exam generators
# ---------------------------
def generate_exam_questions(stories: List[Story], count: int = 3, qtype: str = "mcq") -> List[Dict[str, Any]]:
    """
    qtype: 'mcq' | 'tf' | 'short'
    """
    if not stories:
        return []
    pool = stories[:]
    random.shuffle(pool)
    selected = pool[:min(count, len(pool))]
    questions = []

    for s in selected:
        if qtype == "mcq":
            correct = s.moral or f"What is the lesson of '{s.title}'?"
            distractors = [o.moral for o in stories if o.id != s.id and o.moral]
            while len(distractors) < 3:
                distractors.append("Be cautious and think ahead.")
            distractors = random.sample(distractors, min(3, len(distractors)))
            options = distractors + [correct]
            random.shuffle(options)
            questions.append({
                "type": "mcq",
                "question": f"What is the moral of '{s.title}'?",
                "options": options,
                "answer_index": options.index(correct),
                "context": {"story_id": s.id, "title": s.title}
            })
        elif qtype == "tf":
            # make a claim about the story/moral and mark True/False
            true_stmt = f"The moral of '{s.title}' can be summarized as: {s.moral}"
            # create false statement by swapping morals when possible
            other = next((o for o in stories if o.id != s.id), None)
            if other:
                false_stmt = f"The moral of '{s.title}' can be summarized as: {other.moral}"
            else:
                false_stmt = f"The moral of '{s.title}' is unrelated to the story."
            if random.choice([True, False]):
                questions.append({"type": "tf", "statement": true_stmt, "answer": True, "context": {"story_id": s.id}})
            else:
                questions.append({"type": "tf", "statement": false_stmt, "answer": False, "context": {"story_id": s.id}})
        elif qtype == "short":
            # ask for a short answer — expect a free-text moral or background
            questions.append({
                "type": "short",
                "question": f"In one sentence, what is the moral of '{s.title}'?",
                "expected": s.moral,
                "context": {"story_id": s.id}
            })
        else:
            raise ValueError("Unsupported question type")
    return questions


# ---------------------------
# Exam exporters
# ---------------------------
def exam_to_markdown(questions: List[Dict[str, Any]]) -> str:
    lines = ["# Exam Questions\n"]
    for i, q in enumerate(questions, start=1):
        lines.append(f"## Question {i}")
        if q["type"] == "mcq":
            lines.append(q["question"])
            for j, opt in enumerate(q["options"], start=1):
                lines.append(f"{j}. {opt}")
        elif q["type"] == "tf":
            lines.append(f"Statement: {q['statement']}")
            lines.append("True / False")
        elif q["type"] == "short":
            lines.append(q["question"])
        lines.append("")
    return "\n".join(lines)


def exam_to_csv(questions: List[Dict[str, Any]], out_path: Path) -> None:
    import csv
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Header depends on type; keep a generic table
        writer.writerow(["type", "prompt", "options", "answer", "context"])
        for q in questions:
            if q["type"] == "mcq":
                prompt = q["question"]
                opts = " || ".join(q["options"])
                ans = q["options"][q["answer_index"]]
            elif q["type"] == "tf":
                prompt = q["statement"]
                opts = "True || False"
                ans = "True" if q["answer"] else "False"
            else:
                prompt = q.get("question", "")
                opts = ""
                ans = q.get("expected", "")
            writer.writerow([q["type"], prompt, opts, ans, json.dumps(q.get("context", {}))])


def exam_to_latex(questions: List[Dict[str, Any]]) -> str:
    # Minimal LaTeX document for printable exam
    parts = ["\\documentclass{article}", "\\usepackage[utf8]{inputenc}", "\\begin{document}", "\\section*{Exam}"]
    for i, q in enumerate(questions, start=1):
        parts.append(f"\\subsection*{{Question {i}}}")
        if q["type"] == "mcq":
            parts.append("\\begin{enumerate}")
            parts.append(f"\\item {q['question']}")
            parts.append("\\begin{enumerate}")
            for opt in q["options"]:
                parts.append(f"\\item {escape_latex(opt)}")
            parts.append("\\end{enumerate}")
            parts.append("\\end{enumerate}")
        elif q["type"] == "tf":
            parts.append(f"\\item {escape_latex(q['statement'])} (True / False)")
        else:
            parts.append(f"\\item {escape_latex(q.get('question', ''))}")
    parts.append("\\end{document}")
    return "\n".join(parts)


def escape_latex(s: str) -> str:
    replace = {
        "&": "\\&", "%": "\\%", "$": "\\$", "#": "\\#", "_": "\\_", "{": "\\{", "}": "\\}", "~": "\\textasciitilde{}", "^": "\\^{}",
        "\\": "\\textbackslash{}"
    }
    return "".join(replace.get(ch, ch) for ch in s)


# ---------------------------
# Audio / TTS
# ---------------------------
def speak(text: str) -> None:
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception:
        print("⚠ Audio not available (install pyttsx3 to enable)")


def save_story_audio(stories: List[Story], story_id: Optional[int], out_dir: Path) -> None:
    try:
        import pyttsx3
    except Exception:
        print("⚠ pyttsx3 not available. Install with `pip install pyttsx3` to enable audio export.")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    engine = pyttsx3.init()
    targets = [s for s in stories if story_id is None or s.id == story_id]
    if not targets:
        print("No stories to export.")
        return
    for s in targets:
        fname = out_dir / f"story_{s.id}_{safe_filename(s.title)}.mp3"
        try:
            engine.save_to_file(s.story, str(fname))
        except Exception:
            alt = fname.with_suffix(".wav")
            engine.save_to_file(s.story, str(alt))
            print(f"Saved as {alt}")
    try:
        engine.runAndWait()
    except Exception:
        print("⚠ Failed running TTS engine. Check pyttsx3.")


def safe_filename(s: str) -> str:
    return "".join(ch for ch in s if ch.isalnum() or ch in (" ", "_", "-")).rstrip().replace(" ", "_")


# ---------------------------
# Voice search (VOSK / SpeechRecognition)
# ---------------------------
def perform_text_search(stories: List[Story], text: str) -> None:
    q = text.lower().strip()
    results = [s for s in stories if q in s.title.lower() or q in s.culture.lower() or q in s.era.lower() or q in s.story.lower()]
    if not results:
        print("No matches.")
        return
    print(f"\nFound {len(results)} matching stories:")
    for s in results:
        print_story_brief(s)


def voice_search(stories: List[Story], model_path: Optional[Path] = None, duration: float = 5.0) -> None:
    print("\n🎙 Voice search (speak after the prompt).")
    vosk_model_env = os.environ.get("VOSK_MODEL_PATH")
    if model_path is None and vosk_model_env:
        model_path = Path(vosk_model_env)

    # Try VOSK
    if model_path and model_path.exists():
        try:
            from vosk import Model, KaldiRecognizer
            import sounddevice as sd
            import json as _json
            model = Model(str(model_path))
            samplerate = 16000
            print(f"Recording {duration} seconds (VOSK)...")
            recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype="int16")
            sd.wait()
            rec_bytes = recording.tobytes()
            rec = KaldiRecognizer(model, samplerate)
            rec.AcceptWaveform(rec_bytes)
            res = rec.Result()
            text = _json.loads(res).get("text", "")
            if not text:
                print("No speech detected.")
                return
            print(f"Recognized: {text}")
            perform_text_search(stories, text)
            return
        except Exception as exc:
            print("VOSK recognition failed:", exc)

    # Fallback to SpeechRecognition
    try:
        import speech_recognition as sr
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening (SpeechRecognition) — please speak now...")
            audio = r.listen(source, phrase_time_limit=duration)
        try:
            text = r.recognize_sphinx(audio)
            print(f"Recognized (pocketsphinx): {text}")
            perform_text_search(stories, text)
            return
        except Exception:
            try:
                print("Pocketsphinx failed; attempting Google (online fallback).")
                text = r.recognize_google(audio)
                print(f"Recognized (google): {text}")
                perform_text_search(stories, text)
                return
            except Exception as exc:
                print("Speech recognition failed:", exc)
    except Exception:
        pass

    # Manual fallback
    print("Voice search not available. Please type your query.")
    q = input("Search query: ").strip()
    if q:
        perform_text_search(stories, q)


# ---------------------------
# VOSK model download helper
# ---------------------------
def download_vosk_model(url: str, out_dir: Path, chunk_size: int = 8192, show_progress: bool = True, dry_run: bool = False) -> None:
    """
    Download and extract a VOSK model tar.gz into out_dir.
    Provide a URL for a model tarball. This function will stream download and extract.
    If dry_run=True, only validates URL and checks connectivity (no write).
    """
    out_dir = Path(out_dir)
    try:
        import requests
    except Exception:
        requests = None

    if dry_run:
        # quick check
        if requests:
            resp = requests.head(url, allow_redirects=True, timeout=10)
            if resp.status_code >= 400:
                raise RuntimeError(f"URL check failed: {resp.status_code}")
            print("URL reachable (dry-run).")
            return
        else:
            print("requests not available; cannot dry-run. Proceeding to attempt download with urllib.")
    # Download streaming
    tmp = tempfile.NamedTemporaryFile(delete=False)
    try:
        if requests:
            with requests.get(url, stream=True, timeout=30) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length") or 0)
                downloaded = 0
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        tmp.write(chunk)
                        downloaded += len(chunk)
                        if show_progress and total:
                            print(f"\rDownloaded {downloaded}/{total} bytes", end="")
            print()
        else:
            # fallback to urllib
            from urllib.request import urlopen
            with urlopen(url, timeout=30) as r:
                while True:
                    chunk = r.read(chunk_size)
                    if not chunk:
                        break
                    tmp.write(chunk)
    finally:
        tmp.flush()
        tmp.close()
    # Extract (assume tar.gz)
    import tarfile
    try:
        with tarfile.open(tmp.name, "r:gz") as tar:
            tar.extractall(path=str(out_dir))
        print(f"Model extracted to {out_dir}")
    except Exception as exc:
        raise RuntimeError("Failed to extract model: " + str(exc))
    finally:
        try:
            os.remove(tmp.name)
        except Exception:
            pass


# ---------------------------
# CLI & interactive
# ---------------------------
def cli() -> None:
    parser = argparse.ArgumentParser(prog="smart_cultural_storyteller")
    parser.add_argument("--data-file", "-d", type=Path, default=DATA_FILE, help="Path to stories JSON file")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("interactive", help="Interactive TUI")

    e_md = sub.add_parser("export-markdown", help="Export all stories to Markdown")
    e_md.add_argument("--out", "-o", type=Path, required=True, help="Output Markdown file")

    exam = sub.add_parser("generate-exam", help="Generate exams")
    exam.add_argument("--count", "-n", type=int, default=3, help="Number of questions")
    exam.add_argument("--type", "-t", choices=["mcq", "tf", "short"], default="mcq", help="Question type")
    exam.add_argument("--format", "-f", choices=["json", "md", "csv", "tex"], default="json", help="Export format")
    exam.add_argument("--out", "-o", type=Path, required=True, help="Output file")

    pdf = sub.add_parser("export-pdf", help="Export markdown to PDF (requires reportlab)")
    pdf.add_argument("--from-md", "-m", type=Path, required=True, help="Source markdown")
    pdf.add_argument("--out", "-o", type=Path, required=True, help="Output pdf")

    audio = sub.add_parser("export-audio", help="Export story audio files (pyttsx3)")
    audio.add_argument("--out-dir", "-o", type=Path, default=Path("audio"), help="Output directory")
    audio.add_argument("--id", type=int, help="Story ID to export (omit for all)")

    vs = sub.add_parser("voice-search", help="Perform voice-based search")
    vs.add_argument("--model", "-m", type=Path, help="Path to VOSK model folder")
    vs.add_argument("--duration", "-t", type=float, default=5.0, help="Recording duration seconds")

    dl = sub.add_parser("download-model", help="Download VOSK model tarball and extract")
    dl.add_argument("--url", required=True, help="Model tar.gz URL")
    dl.add_argument("--out", "-o", type=Path, required=True, help="Output directory")
    dl.add_argument("--dry-run", action="store_true", help="Only test URL; do not download")

    args = parser.parse_args()
    stories = load_data(args.data_file)

    if args.cmd is None or args.cmd == "interactive":
        main_menu_loop(stories, args.data_file)
        return

    if args.cmd == "export-markdown":
        md = export_stories_markdown(stories)
        args.out.write_text(md, encoding="utf-8")
        print(f"Wrote Markdown to {args.out}")
        return

    if args.cmd == "generate-exam":
        qs = generate_exam_questions(stories, count=args.count, qtype=args.type)
        if args.format == "json":
            args.out.write_text(json.dumps(qs, ensure_ascii=False, indent=2), encoding="utf-8")
        elif args.format == "md":
            args.out.write_text(exam_to_markdown(qs), encoding="utf-8")
        elif args.format == "csv":
            exam_to_csv(qs, args.out)
        elif args.format == "tex":
            args.out.write_text(exam_to_latex(qs), encoding="utf-8")
        print(f"Wrote exam to {args.out}")
        return

    if args.cmd == "export-pdf":
        md = args.from_md.read_text(encoding="utf-8")
        try:
            export_pdf_from_markdown(md, args.out)
            print(f"Wrote PDF to {args.out}")
        except Exception as exc:
            print("PDF export failed:", exc)
        return

    if args.cmd == "export-audio":
        save_story_audio(stories, story_id=args.id, out_dir=args.out_dir)
        return

    if args.cmd == "voice-search":
        voice_search(stories, model_path=args.model, duration=args.duration)
        return

    if args.cmd == "download-model":
        try:
            download_vosk_model(args.url, args.out, dry_run=args.dry_run)
        except Exception as exc:
            print("Download failed:", exc)
        return


# Helper: export stories markdown (kept small)
def export_stories_markdown(stories: List[Story]) -> str:
    md = ["# Cultural Stories\n"]
    for s in stories:
        md.append(f"## {s.title}  ")
        md.append(f"- Culture: {s.culture}  ")
        md.append(f"- Era: {s.era}  ")
        md.append("")
        md.append(s.story)
        md.append("")
        md.append(f"**Moral:** {s.moral}")
        md.append("")
        md.append(f"**Background:** {s.background}")
        md.append("\n---\n")
    return "\n".join(md)


def export_pdf_from_markdown(md_text: str, out_path: Path) -> None:
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except Exception as exc:
        raise RuntimeError("PDF export requires reportlab") from exc
    c = canvas.Canvas(str(out_path), pagesize=letter)
    width, height = letter
    y = height - 50
    for line in md_text.splitlines():
        if y < 50:
            c.showPage()
            y = height - 50
        c.drawString(50, y, line[:200])
        y -= 14
    c.save()


# Interactive menu (simplified)
def main_menu_loop(stories: List[Story], data_path: Optional[Path] = None) -> None:
    menu = [
        ("Hear a folk tale", hear_story),
        ("Retell story in a new style", retell_story),
        ("Generate moral lesson", show_moral),
        ("Listen to audio narration", audio_narration),
        ("Explore cultural background", cultural_background),
        ("Interactive story mode", interactive_story),
        ("Add a story", lambda s: add_story(s, data_path)),
        ("Edit a story", lambda s: edit_story(s, data_path)),
        ("Delete a story", lambda s: delete_story(s, data_path)),
        ("Search stories", search_stories),
        ("Voice search (microphone)", lambda s: voice_search(s, None, duration=5.0)),
        ("Take a short exam (MCQ)", take_exam_interactive),
        ("Export stories to markdown", lambda s: export_md_interactive(s, data_path)),
        ("Export story audio", lambda s: save_story_audio(s, None, Path("audio"))),
        ("Download VOSK model", lambda s: download_model_interactive()),
        ("List all stories", list_stories),
        ("Exit", None)
    ]
    try:
        while True:
            print("\n=== SMART CULTURAL STORYTELLER ===")
            for i, (label, _) in enumerate(menu, start=1):
                print(f"{i}. {label}")
            try:
                choice = input("Choose option: ").strip()
            except EOFError:
                print("\nInput closed. Exiting.")
                break
            if not choice.isdigit():
                print("Invalid option.")
                continue
            idx = int(choice) - 1
            if idx < 0 or idx >= len(menu):
                print("Invalid option.")
                continue
            label, func = menu[idx]
            if label == "Exit":
                print("Goodbye! 👋")
                break
            if func:
                func(stories)
    except KeyboardInterrupt:
        print("\nInterrupted. Goodbye.")


# The interactive helpers below are straightforward; keep earlier behavior for backward compat
def hear_story(stories: List[Story]) -> None:
    s = select_story(stories)
    if s:
        print(f"\n📖 {s.title}\n")
        print(s.story)


def retell_story(stories: List[Story]) -> None:
    s = select_story(stories)
    if not s:
        return
    print("\nChoose retelling style:")
    print("1. Modern")
    print("2. Dramatic")
    print("3. Kids")
    choice = input("Style: ").strip()
    base = s.story
    if choice == "1":
        ret = "In today's world, the story unfolds like this:\n" + base
    elif choice == "2":
        ret = "⚔️ A dramatic retelling:\n" + " ".join([w.upper() for w in base.split()])
    elif choice == "3":
        ret = "🌈 Once upon a time...\n" + base + "\nEveryone learned a lesson!"
    else:
        print("Invalid choice.")
        return
    print("\n🎭 RETOLD STORY\n")
    print(ret)


def show_moral(stories: List[Story]) -> None:
    s = select_story(stories)
    if s:
        print("\n🧠 MORAL LESSON")
        print(s.moral)


def audio_narration(stories: List[Story]) -> None:
    s = select_story(stories)
    if s:
        print("\n🔊 Narrating story...")
        speak(s.story)


def cultural_background(stories: List[Story]) -> None:
    s = select_story(stories)
    if s:
        print("\n🏛 CULTURAL & HISTORICAL BACKGROUND")
        print(s.background)


def interactive_story(stories: List[Story]) -> None:
    print("\n🎮 INTERACTIVE STORY MODE")
    print("You hear about a hidden treasure.")
    print("1. Investigate alone")
    print("2. Inform the king")
    choice = input("Choose: ").strip()
    if choice == "1":
        print("\nYou gain wisdom but face danger.")
    elif choice == "2":
        print("\nJustice is served through intelligence.")
    else:
        print("\nThe story fades into legend.")


# Story management
def add_story(stories: List[Story], path: Optional[Path] = None) -> None:
    print("\n➕ Add a new story (leave blank to cancel)")
    title = input("Title: ").strip()
    if not title:
        print("Cancelled.")
        return
    culture = input("Culture: ").strip() or "Unknown"
    era = input("Era: ").strip() or "Unknown"
    story_text = input("Story text: ").strip()
    moral = input("Moral: ").strip() or ""
    background = input("Background: ").strip() or ""
    sid = next_id(stories)
    story = Story(id=sid, title=title, culture=culture, era=era, story=story_text, moral=moral, background=background)
    stories.append(story)
    save_data(stories, path)
    print(f"Added story id={sid}.")


def edit_story(stories: List[Story], path: Optional[Path] = None) -> None:
    s = select_story(stories)
    if not s:
        return
    print("Leave blank to keep current value.")
    s.title = input(f"Title [{s.title}]: ").strip() or s.title
    s.culture = input(f"Culture [{s.culture}]: ").strip() or s.culture
    s.era = input(f"Era [{s.era}]: ").strip() or s.era
    s.story = input(f"Story [{s.story[:30]}...]: ").strip() or s.story
    s.moral = input(f"Moral [{s.moral}]: ").strip() or s.moral
    s.background = input(f"Background [{s.background[:30]}...]: ").strip() or s.background
    save_data(stories, path)
    print("Story updated.")


def delete_story(stories: List[Story], path: Optional[Path] = None) -> None:
    s = select_story(stories)
    if not s:
        return
    confirm = input(f"Type DELETE to remove '{s.title}': ").strip()
    if confirm == "DELETE":
        stories.remove(s)
        save_data(stories, path)
        print("Deleted.")
    else:
        print("Aborted.")


def select_story(stories: List[Story]) -> Optional[Story]:
    list_stories(stories)
    sid = input("\nEnter story ID: ").strip()
    if not sid.isdigit():
        print("Invalid id.")
        return None
    s = find_story_by_id(stories, int(sid))
    if not s:
        print("Story not found.")
    return s


def search_stories(stories: List[Story]) -> None:
    q = input("Search by title / culture / era (substring): ").strip().lower()
    if not q:
        print("Empty query.")
        return
    results = [s for s in stories if q in s.title.lower() or q in s.culture.lower() or q in s.era.lower()]
    if not results:
        print("No matches.")
        return
    print(f"\nFound {len(results)} matching stories:")
    for s in results:
        print_story_brief(s)


def download_model_interactive() -> None:
    url = input("Model tar.gz URL: ").strip()
    out = input("Output directory: ").strip()
    if not url or not out:
        print("Cancelled.")
        return
    download_vosk_model(url, Path(out))


def take_exam_interactive(stories: List[Story]) -> None:
    try:
        n = int(input("How many questions? ").strip())
    except Exception:
        n = 3
    qs = generate_exam_questions(stories, count=n, qtype="mcq")
    score = 0
    for idx, q in enumerate(qs, start=1):
        print(f"\nQ{idx}: {q['question']}")
        for i, opt in enumerate(q["options"], start=1):
            print(f"  {i}. {opt}")
        ans = input("Your answer (number): ").strip()
        if not ans.isdigit():
            print("Invalid answer.")
            continue
        if int(ans) - 1 == q["answer_index"]:
            print("Correct!")
            score += 1
        else:
            correct = q["options"][q["answer_index"]]
            print(f"Incorrect. Correct: {correct}")
    print(f"\nScore: {score}/{len(qs)}")


def export_md_interactive(stories: List[Story], path: Optional[Path] = None) -> None:
    out = input("Output markdown file path: ").strip()
    if not out:
        print("Cancelled.")
        return
    md = export_stories_markdown(stories)
    Path(out).write_text(md, encoding="utf-8")
    print(f"Wrote Markdown to {out}")


# ---------------------------
# Keep minimal tests in repo tests/
# ---------------------------
if __name__ == "__main__":
    cli()
