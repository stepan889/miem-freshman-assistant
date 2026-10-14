import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from nltk.stem.snowball import SnowballStemmer

from .config import ACADEMIC_YEAR, DATA_FILE

STEMMER = SnowballStemmer("russian")  # Не требует скачивания словарей.
STOP = set("и в во на по с со к у из за для как что это ли бы о об а но или мне мне есть какие какой какая расскажи пожалуйста".split())


def tokens(text: str) -> list[str]:
    return [STEMMER.stem(t) for t in re.findall(r"[а-яёa-z0-9]+", text.lower().replace("ё", "е"))
            if t not in STOP and len(t) > 1]


def split_text(text: str, size=1800) -> list[str]:
    parts = []
    while len(text) > size:
        end = text.rfind(" ", size // 2, size)
        if end < 0:
            end = size
        parts.append(text[:end])
        text = text[end:].lstrip()
    if text:
        parts.append(text)
    return parts


@dataclass(frozen=True)
class Chunk:
    id: str
    course_id: str
    course: str
    section: str
    text: str
    url: str

    def context(self, label: str) -> dict:
        return {"id": label, "discipline": self.course, "section": self.section,
                "academic_year": ACADEMIC_YEAR, "text": self.text}


class KnowledgeBase:
    def __init__(self, path: Path = DATA_FILE):
        self.snapshot = json.loads(path.read_text(encoding="utf-8"))
        if self.snapshot.get("academic_year") != ACADEMIC_YEAR or self.snapshot.get("study_year") != 1:
            raise ValueError("База должна содержать только 1 курс 2026/2027 года.")
        self.courses = self.snapshot["courses"]
        if not self.courses:
            raise ValueError("База дисциплин пуста. Выполните python -m miem_bot ingest.")
        self.by_id = {c["id"]: c for c in self.courses}
        self.chunks: list[Chunk] = []
        for c in self.courses:
            if c["academic_year"] != ACADEMIC_YEAR or c["study_year"] != 1:
                raise ValueError("В базе найдена дисциплина другого курса/года.")
            metadata = self.metadata(c)
            self.chunks.append(Chunk(f"{c['id']}:meta", c["id"], c["title"], "Карточка дисциплины", metadata, c["url"]))
            for i, s in enumerate(c["sections"]):
                for j, text in enumerate(split_text(s["text"])):
                    url = c["url"] + ("#" + s["anchor"] if s["anchor"] else "")
                    self.chunks.append(Chunk(f"{c['id']}:{i}:{j}", c["id"], c["title"], s["title"], text, url))
        self.term_counts = [Counter(tokens(f"{c.course} {c.section} {c.text}")) for c in self.chunks]
        df = Counter(t for counts in self.term_counts for t in counts)
        self.idf = {t: math.log(1 + (len(self.chunks) - count + 0.5) / (count + 0.5)) for t, count in df.items()}
        self.avg_len = sum(sum(c.values()) for c in self.term_counts) / len(self.chunks)

    @staticmethod
    def metadata(c):
        published = "; ".join(s["title"] for s in c["sections"])
        return (f"{c['title']}. Учебный год {c['academic_year']}, {c['when']}. "
                f"Кредиты: {c['credits'] or 'не указаны'}. {c['status']}. {c['department']}. "
                f"Преподаватели: {', '.join(c['teachers']) or 'не указаны'}. "
                f"Опубликованные разделы: {published}.")

    def catalog(self):
        return [{"id": c["id"], "title": c["title"], "when": c["when"],
                 "sections": sorted({s["title"] for s in c["sections"]})} for c in self.courses]

    def scores(self, query):
        terms = set(tokens(query))
        scores = []
        for count in self.term_counts:
            length = sum(count.values())
            score = sum(self.idf.get(t, 0) * count[t] * 2.5 /
                        (count[t] + 1.5 * (0.25 + 0.75 * length / self.avg_len))
                        for t in terms if count[t])
            scores.append(score)
        return scores

    def search(self, query, course_ids=None, sections=None, limit=14):
        scores = self.scores(query)
        wanted = set(course_ids or [])
        section_terms = set(tokens(" ".join(sections or [])))
        ranked = []
        for chunk, score in zip(self.chunks, scores):
            if wanted and chunk.course_id not in wanted:
                continue
            if section_terms and section_terms & set(tokens(chunk.section)):
                score += 12
            if score > 0:
                ranked.append((score, chunk))
        return [c for _, c in sorted(ranked, key=lambda x: x[0], reverse=True)[:limit]]

    def context_for_plan(self, query, course_ids, sections, catalog=False):
        if catalog:
            return [c for c in self.chunks if c.section == "Карточка дисциплины"]
        if not course_ids:
            return self.search(query, sections=sections, limit=16)
        scores = dict(zip((c.id for c in self.chunks), self.scores(query)))
        result = []
        # Каждой названной дисциплине выделяется равная доля контекста.
        budget = min(20000, 48000 // len(course_ids))
        for course_id in course_ids:
            candidates = [c for c in self.chunks if c.course_id == course_id]
            desired = set(sections)
            focused = [c for c in candidates if c.section in desired]
            # Если раздел отсутствует, метаданные покажут это модели.
            preferred = focused if desired else candidates
            preferred = sorted(preferred, key=lambda c: scores[c.id], reverse=True)
            meta = next(c for c in candidates if c.section == "Карточка дисциплины")
            chosen, used = [meta], len(meta.text)
            for chunk in preferred:
                if chunk.id != meta.id and used + len(chunk.text) <= budget:
                    chosen.append(chunk)
                    used += len(chunk.text)
            result.extend(chosen)
        return result
