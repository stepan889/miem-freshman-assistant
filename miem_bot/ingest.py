"""Публичные карточки ВШЭ: пагинация, проверка курса/года, атомарный снимок."""
import asyncio
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urljoin, urlparse, urlencode

import httpx
from bs4 import BeautifulSoup, Tag

from .config import ACADEMIC_YEAR, CATALOG_URL, DATA_FILE


def clean_text(node: Tag) -> str:
    # SVG title содержит важные признаки «блокирующий» / «неблокирующий».
    return re.sub(r"[ \t\xa0]+", " ", node.get_text(" ", strip=True)).strip()


def section_text(node: Tag) -> str:
    items = node.select("ul.pud__list > li")
    if items:
        return "\n".join(clean_text(item) for item in items)
    paragraphs = node.find_all("p", recursive=False)
    if paragraphs:
        return "\n".join(clean_text(p) for p in paragraphs)
    return clean_text(node)


def catalog_links(html: str, url: str) -> tuple[list[str], str | None]:
    soup = BeautifulSoup(html, "html.parser")
    heading = soup.find("h1")
    if not heading or not re.search(r"2026\s*[/–-]\s*2027", clean_text(heading)):
        raise ValueError("Каталог ВШЭ не подтверждает 2026/2027 год (возможна страница защиты).")
    links = sorted({urljoin(url, a["href"]) for a in soup.select("a[href]")
                    if re.search(r"/ba/isct/courses/\d+\.html$", a["href"])})
    next_link = soup.select_one('link[rel="next"]')
    return links, urljoin(url, next_link["href"]) if next_link else None


def parse_course(html: str, url: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.select_one("h1.b-program__header-title")
    header = soup.select_one(".b-program__header-info")
    metadata = soup.select_one(".doc-meta")
    if not title or not header or not metadata:
        raise ValueError(f"Не найдена структура карточки: {url}")
    if ACADEMIC_YEAR not in clean_text(header):
        raise ValueError(f"Карточка другого учебного года: {url}")
    meta = {clean_text(dt).rstrip(":"): clean_text(dt.find_next_sibling("dd"))
            for dt in metadata.select("dt") if dt.find_next_sibling("dd")}
    when = meta.get("Когда читается", "")
    if not re.search(r"(?<!\d)1\s*[-–]?\s*й\s+курс\b", when):
        raise ValueError(f"Карточка не относится к 1 курсу: {url}: {when}")
    program = soup.select_one(".pud")
    sections = []
    if program:
        for h in program.select("h4.pud__title"):
            content = h.find_next_sibling("div", class_="pud__content")
            # Не переносим содержимое следующего раздела под пустой заголовок.
            sibling = h.find_next_sibling()
            if not content or sibling is not content:
                continue
            anchor = h.select_one("a[id]")
            text = section_text(content)
            if text:
                sections.append({"title": clean_text(h), "anchor": anchor["id"] if anchor else "",
                                 "text": text})
    if not sections:
        # Некоторые карточки (экзамены, физкультура) имеют только метаданные.
        sections = [{"title": "Программа дисциплины", "anchor": "",
                     "text": "Подробная программа на публичной карточке не опубликована."}]
    blocks = header.select(".b-program__header-info-block3 .b-program_big")
    teachers = [clean_text(n) for n in soup.select(".b-peoples__person span.b")]
    return {
        "id": re.search(r"/(\d+)\.html", url).group(1), "title": clean_text(title),
        "url": url, "academic_year": ACADEMIC_YEAR, "study_year": 1,
        "when": when, "credits": clean_text(blocks[0]) if blocks else "",
        "department": meta.get("Кто читает", ""), "status": meta.get("Статус", ""),
        "teachers": teachers, "sections": sections,
        "html_sha256": hashlib.sha256(html.encode()).hexdigest(),
    }


def safe_catalog_url(url: str) -> str:
    p = urlparse(url)
    if p.hostname != "www.hse.ru" or not p.path.startswith("/ba/isct/courses"):
        raise ValueError("Неожиданная ссылка пагинации ВШЭ.")
    params = parse_qs(p.query)
    params.update({"course": ["1.1.1.4"], "year": ["2026"]})
    return p._replace(scheme="https", query=urlencode(params, doseq=True)).geturl()


async def ingest(output: Path = DATA_FILE) -> dict:
    raw_dir = output.parent / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=45, follow_redirects=True,
                                 headers={"User-Agent": "MIEM-Freshman-Prototype/0.1"}) as client:
        async def fetch(url):
            for attempt in range(3):
                try:
                    response = await client.get(url)
                    response.raise_for_status()
                    return response.text
                except (httpx.TransportError, httpx.HTTPStatusError):
                    if attempt == 2:
                        raise
                    await asyncio.sleep(1 + attempt)

        url, visited, course_urls = CATALOG_URL, set(), set()
        while url:
            url = safe_catalog_url(url)
            if url in visited or len(visited) >= 20:
                raise ValueError("Зацикливание пагинации: снимок не изменён.")
            visited.add(url)
            html = await fetch(url)
            (raw_dir / f"catalog-{len(visited)}.html").write_text(html, encoding="utf-8")
            links, url = catalog_links(html, url)
            course_urls.update(links)
        if not course_urls:
            raise ValueError("Каталог пуст: предыдущий снимок не изменён.")
        courses = []
        for course_url in sorted(course_urls):
            html = await fetch(course_url)
            course = parse_course(html, course_url)
            (raw_dir / f"{course['id']}.html").write_text(html, encoding="utf-8")
            courses.append(course)
            print(f"✓ {course['title']}", flush=True)
            await asyncio.sleep(0.25)
    result = {"academic_year": ACADEMIC_YEAR, "study_year": 1, "catalog_url": CATALOG_URL,
              "fetched_at": datetime.now(timezone.utc).isoformat(),
              "catalog_pages": sorted(visited), "courses": sorted(courses, key=lambda c: c["title"].casefold())}
    tmp = output.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(output)
    return result
