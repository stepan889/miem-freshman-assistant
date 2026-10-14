import json
from pathlib import Path

import pytest

from miem_bot.config import DATA_FILE
from miem_bot.ingest import catalog_links, parse_course, safe_catalog_url

URL = "https://www.hse.ru/ba/isct/courses/123.html"


def card(year="2026/2027", when="1-й курс, 1-4 модуль"):
    return f'''<h1 class="b-program__header-title">Алгебра и геометрия</h1>
    <div class="b-program__header-info">{year}</div>
    <div class="doc-meta"><dl><dt>Когда читается:</dt><dd>{when}</dd></dl></div>
    <div class="pud"><h4 class="pud__title">Промежуточная аттестация<a id="grade"></a></h4>
    <div class="pud__content"><ul class="pud__list">
    <li>2 модуль: 0.275 * Контрольная + 0.5 * Экзамен</li><li>4 модуль: 0.2 * Оценка 2 модуля</li>
    </ul></div><h4 class="pud__title">Список литературы</h4>
    <h4 class="pud__title">Основная литература<a id="library"></a></h4>
    <div class="pud__content">Беклемишев, 1998</div></div>'''


def test_parser_preserves_formula_modules_and_heading_boundaries():
    c = parse_course(card(), URL)
    assert c["study_year"] == 1
    assert c["id"] == "123"
    assert len(c["sections"]) == 2
    assert c["sections"][0]["text"] == "2 модуль: 0.275 * Контрольная + 0.5 * Экзамен\n4 модуль: 0.2 * Оценка 2 модуля"
    assert c["sections"][1]["text"] == "Беклемишев, 1998"


@pytest.mark.parametrize("year,when", [("2025/2026", "1-й курс, 1 модуль"),
                                       ("2026/2027", "2-й курс, 1 модуль"),
                                       ("2026/2027", "11-й курс, 1 модуль")])
def test_reject_wrong_year_or_study_year(year, when):
    with pytest.raises(ValueError):
        parse_course(card(year, when), URL)


def test_catalog_follows_next_page_and_deduplicates():
    html = '''<h1>Учебные курсы 2026/2027 учебного года</h1>
    <a href="/ba/isct/courses/123.html">Курс</a><a href="/ba/isct/courses/123.html">Преподаватель</a>
    <a href="/ba/isct/courses/old.html">Не карточка</a>
    <link rel="next" href="/ba/isct/courses/page2.html?year=2026&amp;course=1.1.1.4">'''
    links, next_page = catalog_links(html, URL)
    assert links == [URL]
    assert "page2.html" in next_page
    assert "course=1.1.1.4" in safe_catalog_url(next_page)


def test_no_silent_acceptance_of_antibot_page():
    with pytest.raises(ValueError):
        catalog_links("<h1>Access denied</h1>", URL)


def test_pagination_preserves_scope():
    assert "year=2026" in safe_catalog_url("https://www.hse.ru/ba/isct/courses/page2.html?year=2025")
    with pytest.raises(ValueError):
        safe_catalog_url("https://evil.example/page")


def test_shipped_snapshot_scope_and_coverage():
    snapshot = json.loads(DATA_FILE.read_text())
    assert len(snapshot["catalog_pages"]) == 2
    assert len(snapshot["courses"]) == 25
    assert len({c["id"] for c in snapshot["courses"]}) == 25
    assert all(c["academic_year"] == "2026/2027" and c["study_year"] == 1 for c in snapshot["courses"])
    assert all(c["sections"] for c in snapshot["courses"])
    titles = {c["title"] for c in snapshot["courses"]}
    assert "Физика" in titles and "Физическая культура" in titles  # Вторая страница.
