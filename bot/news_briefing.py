"""관심 주제 뉴스를 Claude 로 골라 요약해 카카오톡 '나에게 보내기'로 보낸다.

받는 사람마다 한 번:
             python news_briefing.py --auth                          (로그인 주소 출력)
             python news_briefing.py --code "<URL>" --name <이름>   (이동된 주소로 토큰 저장)
매일 실행:   python news_briefing.py

PAGES_DIR·PAGES_URL 이 있으면 관심사별 탭 HTML 을 GitHub Pages 에 올리고 링크 한 통만 보낸다.
"""

import datetime
import html
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

import anthropic
import feedparser
import requests
import trafilatura
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

BASE_DIR = Path(__file__).resolve().parent
TOKEN_DIR = BASE_DIR / "tokens"  # 받는 사람마다 <이름>.json
LEGACY_TOKEN_FILE = BASE_DIR / "kakao_tokens.json"
load_dotenv(BASE_DIR / ".env")

KAKAO_REST_KEY = os.environ.get("KAKAO_REST_KEY", "")
KAKAO_REDIRECT_URI = os.environ.get("KAKAO_REDIRECT_URI", "https://localhost")
KAKAO_CLIENT_SECRET = os.environ.get("KAKAO_CLIENT_SECRET")  # 콘솔에서 켠 경우만
MY_INTERESTS = os.environ.get("MY_INTERESTS") or "경제, 부동산, AI, IT·테크, 건강"
INTEREST_LIST = [s.strip() for s in MY_INTERESTS.split(",") if s.strip()]
PAGES_DIR = os.environ.get("PAGES_DIR")  # GitHub Pages 저장소 로컬 클론
PAGES_URL = os.environ.get("PAGES_URL", "").rstrip("/")  # 예: https://아이디.github.io/news-briefing
SELECT_MODEL = "claude-haiku-5-5"  # 기사 고르기는 단순 분류라 가벼운 모델
SUMMARY_MODEL = os.environ.get("SUMMARY_MODEL") or "claude-sonnet-5-5"
PRICES = {"claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50)}  # 100만 토큰당 달러

RSS_FEEDS = [
    "https://feeds.bbci.co.uk/korean/rss.xml",  # BBC 코리아
    "https://rss.donga.com/total.xml",  # 동아일보
    "https://www.yna.co.kr/rss/news.xml",  # 연합뉴스
    "https://www.yna.co.kr/rss/economy.xml",  # 연합뉴스 경제
    "https://www.yna.co.kr/rss/industry.xml",  # 연합뉴스 산업
    "https://www.yna.co.kr/rss/health.xml",  # 연합뉴스 건강
    "https://rss.donga.com/health.xml",  # 동아일보 건강
    "https://www.hankyung.com/feed/realestate",  # 한국경제 부동산
]

KAKAO_TEXT_LIMIT = 200  # 텍스트 템플릿 본문 최대 길이


# ── 뉴스 수집 ────────────────────────────────
def parse_feed(url, tries=2):
    """사이트가 접속을 끊으면 한 번 더 시도하고, 그래도 안 되면 빈 피드로 건너뛴다."""
    for i in range(tries):
        try:
            return feedparser.parse(url, agent="Mozilla/5.0")
        except Exception as e:
            print(f"[경고] 피드 접속 실패 ({i + 1}/{tries}): {url} ({e})")
            time.sleep(3)
    return feedparser.FeedParserDict(entries=[])


def get_news(max_per_feed=20):
    articles = []
    seen = set()  # 종합·분야 피드에 같은 기사가 겹친다
    for url in RSS_FEEDS:
        feed = parse_feed(url)
        if not feed.entries:
            print(f"[경고] 기사를 못 가져옴: {url}")
        for entry in feed.entries[:max_per_feed]:
            link = entry.get("link", "")
            if link in seen:
                continue
            seen.add(link)
            articles.append(
                {
                    "title": entry.get("title", ""),
                    "summary": entry.get("summary", "")[:300],
                    "link": entry.get("link", ""),
                }
            )
    return articles


# ── Claude 요약 ──────────────────────────────
PER_CATEGORY = 3
BODY_LIMIT = 2500  # 기사 본문을 Claude 에 넘길 최대 글자 수


class Choice(BaseModel):
    category: str
    id: int


class Selection(BaseModel):
    choices: list[Choice]


class Brief(BaseModel):
    id: int
    title: str
    lead: str
    points: list[str]
    detail: str


class Briefs(BaseModel):
    briefs: list[Brief]
    cheer: str


class Pick(BaseModel):
    category: str
    title: str
    lead: str
    points: list[str]
    detail: str
    link: str
    image: str | None = None
    source: str | None = None


def ask_claude(prompt, schema, max_tokens, model):
    client = anthropic.Anthropic()  # ANTHROPIC_API_KEY 환경 변수 사용
    extra = {}
    if model != "claude-haiku-5-5":  # Haiku 는 서버측 대체 모델을 지원하지 않는다
        # 안전 필터가 거절하면 다른 모델이 이어서 처리
        extra = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
    try:
        response = client.beta.messages.parse(
            model=model,
            max_tokens=max_tokens,
            output_config={"effort": "low"},  # 고르고 요약하는 정도라 낮게 두어 비용 절약
            messages=[{"role": "user", "content": prompt}],
            output_format=schema,
            **extra,
        )
    except ValidationError:
        # 응답 길이 상한에 걸려 JSON 이 중간에 끊긴 경우
        sys.exit(f"Claude 요약 실패: [{model}] 응답이 잘렸습니다 (max_tokens={max_tokens})")
    u = response.usage
    price_in, price_out = PRICES.get(model, PRICES["claude-sonnet-5-5"])
    cost = (u.input_tokens * price_in + u.output_tokens * price_out) / 1_000_000
    print(f"  [{model}] 토큰 입력 {u.input_tokens:,} · 출력 {u.output_tokens:,} → 약 ${cost:.4f}")
    if response.stop_reason == "refusal" or response.parsed_output is None:
        sys.exit(f"Claude 요약 실패: stop_reason={response.stop_reason}")
    return response.parsed_output


def select_articles(articles, interests, today):
    news_text = "\n".join(
        f"[{i}] {a['title']}: {a['summary']}" for i, a in enumerate(articles)
    )
    prompt = f"""오늘은 {today} 입니다. 당신은 뉴스 큐레이터입니다.
아래 뉴스 목록에서 관심 주제 [{", ".join(interests)}] 각각에 관련된 기사를 주제마다 최대 {PER_CATEGORY}개 고르세요.
한 기사는 가장 가까운 주제 하나에만 넣고, 관련 기사가 없는 주제는 비워 두세요.
비슷한 사건을 다룬 기사가 여럿이면 하나만 고르세요.
'주요 뉴스 헤드라인'처럼 여러 소식을 모아 놓은 기사는 고르지 마세요.
건강처럼 생활 주제는 인사·실적 같은 기업 소식보다 실생활에 도움이 되는 정보·연구 결과를 우선하세요.

- category: 위 관심 주제 중 하나 (글자 그대로)
- id: 목록의 [번호]

뉴스 목록:
{news_text}
"""
    choices = ask_claude(prompt, Selection, 4000, SELECT_MODEL).choices
    picked, counts = [], {}
    for c in choices:
        if 0 <= c.id < len(articles) and counts.get(c.category, 0) < PER_CATEGORY:
            counts[c.category] = counts.get(c.category, 0) + 1
            picked.append((c.category, articles[c.id]))
    return picked


def fetch_article(link):
    """기사 본문·대표 이미지·언론사명. 실패하면 빈 값."""
    try:
        doc = trafilatura.bare_extraction(trafilatura.fetch_url(link), with_metadata=True)
        doc = doc.as_dict() if hasattr(doc, "as_dict") else (doc or {})
        return doc.get("text") or "", doc.get("image"), doc.get("sitename")
    except Exception as e:  # 한 기사 실패로 전체를 멈추지 않는다
        print(f"[경고] 본문 수집 실패: {link} ({e})")
        return "", None, None


def summarize_with_claude(articles, interests, today):
    picked = select_articles(articles, interests, today)
    fetched = [fetch_article(a["link"]) for _, a in picked]
    return write_briefs(picked, fetched, today, SUMMARY_MODEL)


def write_briefs(picked, fetched, today, model):
    blocks = "\n\n".join(
        f"=== [{i}] {a['title']}\n{(body or a['summary'])[:BODY_LIMIT]}"
        for i, ((_, a), (body, _, _)) in enumerate(zip(picked, fetched))
    )
    prompt = f"""아래 기사들을 읽고 각각 한국어 브리핑을 써 주세요. 기사에 없는 내용은 쓰지 마세요.

- id: 기사 [번호]
- title: 기사 제목 (40자 이내로 다듬기)
- lead: 기사 핵심 한 문장 (60자 이내, '~다'로 끝내기)
- points: 핵심 사실 3~4개, 각각 한 문장 (숫자·주체를 구체적으로, '~다'로 끝내기)
- detail: 배경과 의미를 담은 3~4문장 문단 (모든 문장을 '~습니다'체로)

그리고 cheer 에는 오늘({today}) 하루를 응원하는 힘나는 한마디를 직접 지어 주세요.
30자 이내, 따뜻하고 담백하게. 유명인 명언을 인용하거나 출처를 붙이지 마세요.

기사:
{blocks}
"""
    result = ask_claude(prompt, Briefs, 20000, model)
    briefs = {b.id: b for b in result.briefs}

    picks = []
    for i, ((category, a), (_, image, source)) in enumerate(zip(picked, fetched)):
        b = briefs.get(i)
        if not b:
            continue
        picks.append(
            Pick(
                category=category, title=b.title, lead=b.lead, points=b.points,
                detail=b.detail, link=a["link"], image=image, source=source,
            )
        )
    return picks, result.cheer.strip()


# ── 카카오 토큰 ──────────────────────────────
def _token_request(data):
    if KAKAO_CLIENT_SECRET:
        data["client_secret"] = KAKAO_CLIENT_SECRET
    res = requests.post("https://kauth.kakao.com/oauth/token", data=data, timeout=10)
    if not res.ok:
        raise RuntimeError(f"카카오 토큰 요청 실패: {res.status_code} {res.text}")
    return res.json()


def token_path(name):
    return TOKEN_DIR / f"{name}.json"


def save_tokens(name, tokens):
    TOKEN_DIR.mkdir(exist_ok=True)
    token_path(name).write_text(
        json.dumps(tokens, ensure_ascii=False, indent=2), "utf-8"
    )


def recipients():
    # 예전 단일 토큰 파일은 'me' 로 옮긴다
    if LEGACY_TOKEN_FILE.exists() and not token_path("me").exists():
        TOKEN_DIR.mkdir(exist_ok=True)
        LEGACY_TOKEN_FILE.rename(token_path("me"))
    return sorted(p.stem for p in TOKEN_DIR.glob("*.json")) if TOKEN_DIR.exists() else []


def print_auth_url():
    url = "https://kauth.kakao.com/oauth/authorize?" + urlencode(
        {
            "client_id": KAKAO_REST_KEY,
            "redirect_uri": KAKAO_REDIRECT_URI,
            "response_type": "code",
            "scope": "talk_message",
        }
    )
    print("1) 아래 주소를 브라우저에서 열고 카카오 로그인·동의하세요.\n")
    print(url)
    print('\n2) 이동된 주소창의 URL 전체로 실행: python news_briefing.py --code "<URL>" --name <이름>')


def save_auth_code(redirected, name):
    """로그인 후 이동된 주소의 인가 코드로 그 사람의 토큰을 처음 저장한다."""
    code = parse_qs(urlparse(redirected).query).get("code", [redirected])[0]

    tokens = _token_request(
        {
            "grant_type": "authorization_code",
            "client_id": KAKAO_REST_KEY,
            "redirect_uri": KAKAO_REDIRECT_URI,
            "code": code,
        }
    )
    save_tokens(name, tokens)
    print(f"토큰 저장 완료: {token_path(name)}")


def get_access_token(name):
    """저장된 리프레시 토큰으로 매번 새 액세스 토큰을 받는다."""
    tokens = json.loads(token_path(name).read_text("utf-8"))
    fresh = _token_request(
        {
            "grant_type": "refresh_token",
            "client_id": KAKAO_REST_KEY,
            "refresh_token": tokens["refresh_token"],
        }
    )
    # 리프레시 토큰은 만료가 가까울 때만 새로 내려온다
    tokens.update(fresh)
    save_tokens(name, tokens)
    return tokens["access_token"]


# ── 카카오톡 전송 ────────────────────────────
def send_kakao(access_token, text, link, button_title="기사 보기"):
    template = {
        "object_type": "text",
        "text": text[:KAKAO_TEXT_LIMIT],
        "link": {"web_url": link, "mobile_web_url": link},
        "button_title": button_title,
    }
    res = requests.post(
        "https://kapi.kakao.com/v2/api/talk/memo/default/send",
        headers={"Authorization": f"Bearer {access_token}"},
        data={"template_object": json.dumps(template, ensure_ascii=False)},
        timeout=10,
    )
    if not res.ok:
        raise RuntimeError(f"카카오톡 전송 실패: {res.status_code} {res.text}")


# ── HTML 페이지 ──────────────────────────────
FONT_CSS = "https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/variable/pretendardvariable-dynamic-subset.min.css"
WEEKDAYS = "월화수목금토일"

PAGE_CSS = """
:root{--bg:#f2f3f6;--card:#fff;--text:#171a21;--sub:#4a5260;--muted:#8a93a0;--line:#e7e9ee;
--accent:#3d6ef5;--accent2:#8b5cf6;--accent-soft:#eef2ff;--chip:#f3f4f7;
--hero:linear-gradient(135deg,#3d6ef5 0%,#7c5cf6 55%,#c06cf0 100%);
--shadow:0 1px 2px rgba(17,24,39,.04),0 8px 24px rgba(17,24,39,.06)}
@media (prefers-color-scheme:dark){:root{--bg:#0d0e12;--card:#17191f;--text:#eef0f3;--sub:#b3bac4;--muted:#7a838f;
--line:#262932;--accent:#7b9bff;--accent2:#a78bfa;--accent-soft:#1a2036;--chip:#22252d;
--hero:linear-gradient(135deg,#2d4fc4 0%,#5b3fc7 55%,#8a45b8 100%);--shadow:0 0 0 1px #20232b}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);
font-family:"Pretendard Variable",Pretendard,-apple-system,"Apple SD Gothic Neo","Malgun Gothic",sans-serif;
font-size:16px;line-height:1.7;letter-spacing:-.012em;word-break:keep-all;overflow-wrap:anywhere;-webkit-font-smoothing:antialiased}
main{max-width:640px;margin:0 auto;padding:16px 16px 64px}
.hero{position:relative;overflow:hidden;padding:28px 24px 24px;border-radius:28px;background:var(--hero);color:#fff;margin-bottom:8px}
.hero::after{content:"";position:absolute;right:-60px;top:-60px;width:220px;height:220px;border-radius:50%;
background:radial-gradient(circle,rgba(255,255,255,.28),transparent 70%)}
.eyebrow{margin:0 0 6px;font-size:12px;font-weight:700;letter-spacing:.12em;opacity:.85}
h1{margin:0;font-size:30px;line-height:1.25;font-weight:800;letter-spacing:-.035em}
.cheer{position:relative;z-index:1;margin:16px 0 0;padding:12px 14px;border-radius:14px;background:rgba(255,255,255,.16);
font-size:15px;font-weight:600;line-height:1.55}
.cheer span{display:block;margin-bottom:2px;font-size:11px;font-weight:700;letter-spacing:.08em;opacity:.75}
.stats{display:flex;flex-wrap:wrap;gap:6px;margin:18px 0 0;padding:0;list-style:none}
.stats li{padding:5px 11px;border-radius:999px;background:rgba(255,255,255,.18);font-size:13px;font-weight:600}
.stats b{font-weight:800;margin-left:4px}
input[type=radio]{position:fixed;top:0;left:0;opacity:0;pointer-events:none}
.tabs{position:sticky;top:0;z-index:5;display:flex;gap:8px;overflow-x:auto;scrollbar-width:none;
margin:0 -16px 12px;padding:12px 16px;background:color-mix(in srgb,var(--bg) 85%,transparent);
-webkit-backdrop-filter:blur(14px);backdrop-filter:blur(14px)}
.tabs::-webkit-scrollbar{display:none}
.tabs label{flex:none;display:flex;align-items:center;gap:6px;padding:9px 16px;border-radius:14px;background:var(--card);
color:var(--sub);font-size:15px;font-weight:700;cursor:pointer;box-shadow:inset 0 0 0 1px var(--line);transition:all .2s}
.tabs .count{min-width:20px;padding:0 6px;border-radius:999px;background:var(--chip);color:var(--muted);
font-size:12px;line-height:20px;text-align:center}
.panel{display:none}
.panel>*{animation:rise .35s ease both}
.panel>*:nth-child(2){animation-delay:.05s}.panel>*:nth-child(3){animation-delay:.1s}
.panel>*:nth-child(4){animation-delay:.15s}.panel>*:nth-child(5){animation-delay:.2s}
@keyframes rise{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
@media (prefers-reduced-motion:reduce){.panel>*{animation:none}}
article{background:var(--card);border-radius:24px;overflow:hidden;margin-bottom:16px;box-shadow:var(--shadow)}
.media{position:relative;display:block;aspect-ratio:16/9;background:var(--hero)}
.media img{display:block;width:100%;height:100%;object-fit:cover}
.media::after{content:"";position:absolute;inset:0 0 auto 0;height:40%;background:linear-gradient(rgba(0,0,0,.35),transparent)}
.media .tag{position:absolute;left:14px;top:14px;z-index:1;display:flex;align-items:center;gap:8px;padding:4px 12px 4px 4px;
border-radius:12px;background:rgba(15,17,22,.55);-webkit-backdrop-filter:blur(8px);backdrop-filter:blur(8px);color:#fff;font-size:13px;font-weight:700}
.media .no{display:grid;place-items:center;width:26px;height:26px;border-radius:8px;background:linear-gradient(135deg,var(--accent),var(--accent2));font-size:12px;font-weight:800}
.body{padding:20px}
h2{margin:0 0 14px;font-size:21px;line-height:1.42;font-weight:800;letter-spacing:-.03em}
h2 a{color:inherit;text-decoration:none}
.lead{position:relative;margin:0 0 18px;padding:14px 16px 14px 18px;border-radius:14px;background:var(--accent-soft);
font-size:15.5px;font-weight:600;line-height:1.6}
.lead::before{content:"";position:absolute;left:0;top:12px;bottom:12px;width:3px;border-radius:0 3px 3px 0;
background:linear-gradient(var(--accent),var(--accent2))}
.label{margin:0 0 10px;color:var(--muted);font-size:12px;font-weight:700;letter-spacing:.06em}
ol{margin:0 0 18px;padding:0;list-style:none;counter-reset:pt}
ol li{position:relative;padding-left:30px;margin-bottom:10px;font-size:15.5px;line-height:1.65;counter-increment:pt}
ol li::before{content:counter(pt);position:absolute;left:0;top:3px;display:grid;place-items:center;width:20px;height:20px;
border-radius:6px;background:var(--accent-soft);color:var(--accent);font-size:11.5px;font-weight:800}
.detail{margin:0 0 20px;padding-top:16px;border-top:1px dashed var(--line);color:var(--sub);font-size:15px;line-height:1.85}
.foot{display:flex;align-items:center;justify-content:space-between;gap:12px}
.time{color:var(--muted);font-size:13px}
.more{display:inline-flex;align-items:center;gap:4px;padding:10px 16px;border-radius:12px;background:var(--text);
color:var(--bg);font-size:14px;font-weight:700;text-decoration:none}
.empty{padding:64px 20px;color:var(--muted);text-align:center;background:var(--card);border-radius:24px}
.empty b{display:block;font-size:32px;margin-bottom:8px}
footer{margin-top:28px;color:var(--muted);font-size:12px;text-align:center}
"""


def group_by_category(picks):
    groups = {name: [] for name in INTEREST_LIST}
    for p in picks:
        groups.setdefault(p.category.strip() or "기타", []).append(p)
    return groups


def render_card(n, p):
    e = html.escape
    link = e(p.link)
    img = (
        f'<img src="{e(p.image)}" alt="" loading="lazy" referrerpolicy="no-referrer">'
        if p.image else ""
    )
    src = e(p.source) if p.source else ""
    points = "".join(f"<li>{e(t)}</li>" for t in p.points)
    minutes = max(1, round(len(p.lead + p.detail + "".join(p.points)) / 500))
    return (
        f'<article><a class="media" href="{link}" target="_blank" rel="noopener">{img}'
        f'<span class="tag"><span class="no">{n}</span>{src}</span></a>'
        f'<div class="body"><h2><a href="{link}" target="_blank" rel="noopener">{e(p.title)}</a></h2>'
        f'<p class="lead">{e(p.lead)}</p>'
        f'<p class="label">핵심 포인트</p><ol>{points}</ol>'
        f'<p class="detail">{e(p.detail)}</p>'
        f'<div class="foot"><span class="time">{minutes}분 읽기</span>'
        f'<a class="more" href="{link}" target="_blank" rel="noopener">원문 보기 ↗</a></div>'
        f"</div></article>"
    )


def render_page(groups, today, cheer=""):
    e = html.escape
    names = list(groups)
    d = datetime.date.fromisoformat(today)
    heading = f"{d.month}월 {d.day}일 {WEEKDAYS[d.weekday()]}요일"
    # 탭은 radio + CSS 로만 전환
    tab_css = "".join(
        f"#t{i}:checked~.tabs label[for=t{i}]{{background:var(--text);color:var(--bg);box-shadow:none}}"
        f"#t{i}:checked~.tabs label[for=t{i}] .count{{background:var(--accent);color:#fff}}"
        f"#t{i}:checked~#p{i}{{display:block}}"
        for i in range(len(names))
    )
    radios = "".join(
        f'<input type="radio" name="tab" id="t{i}"{" checked" if i == 0 else ""}>'
        for i in range(len(names))
    )
    labels = "".join(
        f'<label for="t{i}">{e(n)}<span class="count">{len(groups[n])}</span></label>'
        for i, n in enumerate(names)
    )
    stats = "".join(f"<li>{e(n)}<b>{len(groups[n])}</b></li>" for n in names)
    cheer_html = f'<p class="cheer"><span>오늘의 한마디</span>{e(cheer)}</p>' if cheer else ""
    panels = "".join(
        f'<section class="panel" id="p{i}">'
        + ("".join(render_card(k, p) for k, p in enumerate(groups[n], 1))
           or '<div class="empty"><b>🌙</b>오늘은 관련 뉴스가 없습니다.</div>')
        + "</section>"
        for i, n in enumerate(names)
    )

    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>뉴스 브리핑 {today}</title>
<link rel="stylesheet" href="{FONT_CSS}">
<style>{PAGE_CSS}{tab_css}</style></head>
<body><main>
<header class="hero"><p class="eyebrow">TODAY'S BRIEFING</p><h1>{heading}<br>뉴스 브리핑</h1>
{cheer_html}<ul class="stats">{stats}</ul></header>
{radios}<nav class="tabs">{labels}</nav>{panels}
<footer>Claude 가 기사 본문을 읽고 요약했습니다 · 자세한 내용은 원문을 확인하세요</footer>
</main>
<script>
var hero = document.querySelector(".hero");
document.querySelectorAll("input[name=tab]").forEach(function (r) {{
  r.addEventListener("change", function () {{
    var top = hero.offsetTop + hero.offsetHeight;
    if (window.scrollY > top) window.scrollTo({{ top: top, behavior: "smooth" }});
  }});
}});
</script>
</body></html>
"""


def publish_page(page, today):
    """archive/날짜.html 과 index.html 을 쓰고 GitHub 에 올린다. 공개 주소를 돌려준다."""
    repo = Path(PAGES_DIR)
    (repo / "archive").mkdir(exist_ok=True)
    (repo / "archive" / f"{today}.html").write_text(page, "utf-8")
    (repo / "index.html").write_text(page, "utf-8")

    def git(*args):
        subprocess.run(["git", "-C", str(repo), *args], check=True)

    git("add", f"archive/{today}.html", "index.html")
    # 같은 날 다시 돌려 내용이 같으면 커밋할 게 없다
    if subprocess.run(["git", "-C", str(repo), "diff", "--cached", "--quiet"]).returncode:
        git("commit", "-m", f"briefing {today}")
    git("push")
    return f"{PAGES_URL}/archive/{today}.html"


def send_page_briefing(token, groups, url, today, cheer=""):
    counts = " · ".join(f"{n} {len(ps)}" for n, ps in groups.items())
    line = f"\n💬 {cheer}" if cheer else ""
    send_kakao(token, f"📰 {today} 뉴스 브리핑{line}\n{counts}", url, "브리핑 보기")


# ── 메인 실행 ────────────────────────────────
def send_briefing(token, picks, today):
    if not picks:
        send_kakao(
            token,
            f"📰 {today} 뉴스 브리핑\n오늘은 [{MY_INTERESTS}] 관련 뉴스가 없습니다.",
            "https://news.naver.com",
            "뉴스 보기",
        )
    else:
        picks = picks[:5]  # 기사마다 한 통이라 너무 많아지지 않게
        send_kakao(
            token,
            f"📰 {today} 뉴스 브리핑\n[{MY_INTERESTS}] 관련 {len(picks)}건",
            picks[0].link,
        )
        for i, p in enumerate(picks, 1):
            send_kakao(token, f"{i}️⃣ {p.title}\n📌 {p.lead}", p.link)


def main(preview=False):
    # 설정이 빠졌으면 Claude 비용을 쓰기 전에 멈춘다
    missing = [k for k in ("ANTHROPIC_API_KEY", "KAKAO_REST_KEY") if not os.environ.get(k)]
    if missing:
        sys.exit(f"환경 변수가 비어 있습니다: {', '.join(missing)}")
    names = recipients()
    if not names and not preview:
        sys.exit("받는 사람이 없습니다. 먼저 `python news_briefing.py --auth` 를 실행하세요.")
    today = datetime.date.today().strftime("%Y-%m-%d")

    print("뉴스 수집 중...")
    articles = get_news()
    print(f"{len(articles)}개 기사 수집 완료. Claude 요약 중...")
    picks, cheer = summarize_with_claude(articles, INTEREST_LIST, today)
    groups = group_by_category(picks)

    if preview:
        out = BASE_DIR / "preview.html"
        out.write_text(render_page(groups, today, cheer), "utf-8")
        print(f"미리보기 저장: {out}")
        return

    url = None
    if PAGES_DIR and PAGES_URL:
        url = publish_page(render_page(groups, today, cheer), today)
        print(f"페이지 게시: {url}")

    # 한 사람이 실패해도 나머지에게는 보낸다
    failed = []
    for name in names:
        try:
            token = get_access_token(name)
            if url:
                send_page_briefing(token, groups, url, today, cheer)
            else:
                send_briefing(token, picks, today)
            print(f"[{name}] 전송 완료 ({len(picks)}건)")
        except RuntimeError as e:
            failed.append(name)
            print(f"[{name}] {e}")
    if failed:
        sys.exit(f"전송 실패: {', '.join(failed)}")


def arg(flag):
    return sys.argv[sys.argv.index(flag) + 1].strip() if flag in sys.argv else None


if __name__ == "__main__":
    if "--auth" in sys.argv:
        print_auth_url()
    elif "--code" in sys.argv:
        save_auth_code(arg("--code"), arg("--name") or "me")
    else:
        main(preview="--preview" in sys.argv)
