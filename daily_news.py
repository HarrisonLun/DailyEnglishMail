#!/usr/bin/env python3
"""
Daily bilingual (EN/ZH-TW) international + financial news digest, with an
English vocabulary-learning section, sent by email via Resend.

Runs standalone (e.g. on GitHub Actions cron) — no dependency on any
desktop app being open.

Required environment variables (set as GitHub Actions secrets):
  GEMINI_API_KEY     - Google Gemini API key (free tier), used to write the digest.
                       Get one for free at https://aistudio.google.com/apikey
  RESEND_API_KEY     - Resend API key, used to send the email
  TO_EMAIL           - destination inbox, e.g. b0987796977@gmail.com
  FROM_EMAIL          (optional) sender, defaults to Resend's shared test
                       sender "onboarding@resend.dev". That sender can only
                       deliver to the email address on your Resend account.
                       Once you verify your own domain in Resend, set this
                       to something like "News <news@yourdomain.com>".
"""

import os
import sys
import json
import time
import datetime
from html import unescape
import re
import xml.etree.ElementTree as ET

import requests

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; DailyNewsBot/1.0)"}
# Gemini's free-tier lineup and the "gemini-flash-latest" alias both drift
# over time — models get deprecated, and the alias can get repointed to a
# model version with different request requirements (seen in practice: a
# working request started returning 400 INVALID_ARGUMENT after Google
# rotated what the alias points to). Rather than pin one name and have the
# whole script break on the next rotation, try several candidates in order
# and use whichever one actually accepts the request.
CANDIDATE_MODELS = [
    "gemini-flash-latest",
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
]

# ---- Configuration -----------------------------------------------------

RSS_FEEDS = [
    ("BBC World", "http://feeds.bbci.co.uk/news/world/rss.xml"),
    ("NYT World", "https://rss.nytimes.com/services/xml/rss/nyt/World.xml"),
    ("NYT Business", "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml"),
    ("CNBC Top News", "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
    ("Yahoo Finance", "https://finance.yahoo.com/news/rssindex"),
    ("MarketWatch Top Stories", "https://feeds.content.dowjones.io/public/rss/mw_topstories"),
]

ENTRIES_PER_FEED = 8


def strip_html(raw: str) -> str:
    text = re.sub(r"<[^>]+>", " ", raw or "")
    text = unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def parse_rss(xml_bytes: bytes):
    """Minimal RSS 2.0 / Atom parser using only the stdlib. Yields (title, summary, link)."""
    root = ET.fromstring(xml_bytes)
    # RSS 2.0: rss/channel/item ; Atom: feed/entry (with namespace)
    items = root.findall(".//item")
    if items:
        for item in items:
            title = (item.findtext("title") or "").strip()
            summary = (item.findtext("description") or "").strip()
            link = (item.findtext("link") or "").strip()
            yield title, summary, link
        return
    # Fallback: Atom
    ns = {"a": "http://www.w3.org/2005/Atom"}
    for entry in root.findall(".//a:entry", ns):
        title = (entry.findtext("a:title", default="", namespaces=ns) or "").strip()
        summary = (entry.findtext("a:summary", default="", namespaces=ns) or "").strip()
        link_el = entry.find("a:link", ns)
        link = link_el.get("href") if link_el is not None else ""
        yield title, summary, link


def fetch_headlines() -> str:
    """Pull recent items from each RSS feed and format as a text blob."""
    blocks = []
    for source, url in RSS_FEEDS:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
        except Exception as e:
            print(f"WARN: failed to fetch {source}: {e}", file=sys.stderr)
            continue
        items = []
        try:
            for title, summary, link in list(parse_rss(resp.content))[:ENTRIES_PER_FEED]:
                title = strip_html(title)
                summary = strip_html(summary)
                if not title:
                    continue
                items.append(f"- {title}. {summary} ({link})")
        except Exception as e:
            print(f"WARN: failed to parse {source}: {e}", file=sys.stderr)
            continue
        if items:
            blocks.append(f"### {source}\n" + "\n".join(items))
    return "\n\n".join(blocks)


def _call_gemini(model: str, prompt: str):
    """
    Call one Gemini model. Returns a parsed dict on success, or None if this
    model couldn't produce a usable response (bad request it doesn't accept,
    or output that wasn't valid JSON) — callers should try the next
    candidate model in that case rather than giving up entirely.
    """
    max_attempts = 3
    resp = None
    for attempt in range(1, max_attempts + 1):
        resp = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            params={"key": os.environ["GEMINI_API_KEY"].strip()},
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                # 8000 and 16000 (alone, with no other generationConfig
                # fields) are both confirmed to work. Raising further to fit
                # richer content — keeping the config otherwise minimal,
                # since adding responseMimeType/thinkingConfig alongside
                # maxOutputTokens is what previously triggered 400 errors.
                "generationConfig": {"temperature": 0.4, "maxOutputTokens": 24000},
            },
            timeout=120,
        )
        # 429 (rate limited) and 503 (temporarily overloaded) are transient —
        # retry the SAME model with backoff instead of moving on immediately.
        if resp.status_code in (429, 503) and attempt < max_attempts:
            wait = 10 * attempt
            print(
                f"WARN: {model} returned {resp.status_code} (attempt {attempt}/{max_attempts}), "
                f"retrying in {wait}s: {resp.text}",
                file=sys.stderr,
            )
            time.sleep(wait)
            continue
        break

    if resp.status_code == 400:
        # Model-specific incompatibility (e.g. this alias currently points to
        # a model version that rejects our request shape). Not worth retrying
        # this model — signal the caller to move on to the next candidate.
        print(f"WARN: {model} returned 400, trying next candidate model: {resp.text}", file=sys.stderr)
        return None
    if resp.status_code >= 300:
        print(f"ERROR calling {model}: {resp.status_code} {resp.text}", file=sys.stderr)
        resp.raise_for_status()

    data = resp.json()
    candidate = data["candidates"][0]
    finish_reason = candidate.get("finishReason")
    if finish_reason and finish_reason not in ("STOP", "MAX_TOKENS"):
        print(f"WARN: {model} unexpected finishReason={finish_reason}: {data}", file=sys.stderr)
    if finish_reason == "MAX_TOKENS":
        print(f"WARN: {model} response was cut off (finishReason=MAX_TOKENS).", file=sys.stderr)

    raw_text = candidate["content"]["parts"][0]["text"].strip()
    # Strip accidental markdown code fences if the model adds them anyway.
    if raw_text.startswith("```"):
        raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
        raw_text = re.sub(r"\s*```$", "", raw_text)
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as e:
        print(f"WARN: {model} output was not valid JSON ({e}), trying next candidate model.", file=sys.stderr)
        print(f"Raw output length: {len(raw_text)} chars; last 300 chars: ...{raw_text[-300:]}", file=sys.stderr)
        return None


def build_digest(headlines_blob: str) -> dict:
    """Ask Gemini (free tier) to turn raw headlines into the bilingual digest + vocab list.

    Tries a list of candidate models in order and uses whichever one actually
    works. Gemini's free-tier model lineup and the "gemini-flash-latest" alias
    both change over time (models get deprecated, aliases get repointed to
    models with different request requirements), so hardcoding a single model
    name is fragile. This fallback chain absorbs that churn.
    """
    today = datetime.date.today().strftime("%Y/%m/%d")
    today_zh = datetime.date.today().strftime("%Y年%m月%d日")

    prompt = f"""You are preparing a daily bilingual (English + Traditional Chinese) news
briefing email from today's raw headlines below. Output ONLY a single JSON object
(no markdown fences, no commentary) with exactly these keys: "subject", "html", "text".

Requirements for the content:
- Pick the most important 7-8 INTERNATIONAL news stories and 7-8 FINANCIAL/MARKETS
  news stories from the raw headlines provided. Skip duplicates, gossip, or trivial items.
- For each story: first write a short, natural English sentence or two (like a wire
  service would phrase it), then immediately below it a Traditional Chinese (繁體中文)
  translation/summary, then the source name in parentheses.
- After both sections, add a "📚 Today's Vocabulary 今日單字學習" section: pick 15-20
  useful English words/phrases that actually appear in the English sentences you wrote
  above (prefer business/finance/news vocabulary a Chinese-speaking learner might not
  know, e.g. "tariff," "hawkish," "volatility," "ceasefire," "antitrust"). For each,
  give: the word/phrase, its part of speech, a concise Traditional Chinese definition,
  and the exact sentence from the briefing where it appears (bold the word/phrase
  within the sentence). Present this as an HTML table in the html version, and as a
  simple list in the text version.
- Be reasonably concise in the HTML markup itself (inline styles, no long comments,
  no unnecessary nesting) so more of the token budget goes to actual content rather
  than markup overhead.
- Title: "📰 Daily Global & Financial News Briefing 每日國際與金融新聞摘要 — {today_zh}"
- subject should be: "📰 Daily News & Vocabulary 每日國際金融新聞與單字 — {today}"
- "html" must be a complete, self-contained HTML email body (inline styles, no
  external CSS/JS), with clear section headers and a horizontal rule between the
  three sections. Keep the HTML compact — no long comments, no unnecessary nesting.
- "text" must be an equivalent plain-text version (no HTML tags), for the plain-text
  fallback part of the email.
- Keep the whole thing readable in about 6-8 minutes. Be factual, no speculation.

Raw headlines (source, title, short summary, link):

{headlines_blob}
"""

    errors = []
    for model in CANDIDATE_MODELS:
        try:
            result = _call_gemini(model, prompt)
        except Exception as e:
            print(f"WARN: {model} raised {type(e).__name__}: {e}", file=sys.stderr)
            errors.append(f"{model}: {e}")
            continue
        if result is not None:
            print(f"INFO: digest generated successfully using model '{model}'")
            return result
        errors.append(f"{model}: returned no usable result (see warnings above)")

    raise RuntimeError(
        "All candidate Gemini models failed to produce a usable digest:\n"
        + "\n".join(errors)
    )


def send_email(subject: str, html: str, text: str):
    # .strip() defensively: GitHub secrets pasted from a browser/clipboard
    # can end up with a trailing newline or leading/trailing whitespace,
    # which `requests` rejects outright when the value goes into a header
    # ("Invalid leading whitespace, reserved character(s), or return
    # character(s) in header value").
    to_email = os.environ["TO_EMAIL"].strip()
    from_email = os.environ.get("FROM_EMAIL", "每日新聞摘要 <onboarding@resend.dev>").strip()
    resend_api_key = os.environ["RESEND_API_KEY"].strip()
    resp = requests.post(
        "https://api.resend.com/emails",
        headers={
            "Authorization": f"Bearer {resend_api_key}",
            "Content-Type": "application/json",
        },
        json={
            "from": from_email,
            "to": [to_email],
            "subject": subject,
            "html": html,
            "text": text,
        },
        timeout=30,
    )
    if resp.status_code >= 300:
        print(f"ERROR sending email: {resp.status_code} {resp.text}", file=sys.stderr)
        resp.raise_for_status()
    print(f"Email sent: {resp.json()}")


def main():
    headlines_blob = fetch_headlines()
    if not headlines_blob.strip():
        print("ERROR: no headlines fetched from any feed", file=sys.stderr)
        sys.exit(1)

    digest = build_digest(headlines_blob)
    send_email(digest["subject"], digest["html"], digest["text"])


if __name__ == "__main__":
    main()
