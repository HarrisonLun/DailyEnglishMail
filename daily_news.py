#!/usr/bin/env python3
"""
Daily bilingual (EN/ZH-TW) international + financial news digest, with an
English vocabulary-learning section, sent by email via Resend.

Runs standalone (e.g. on GitHub Actions cron) — no dependency on any
desktop app being open.

Required environment variables (set as GitHub Actions secrets):
  ANTHROPIC_API_KEY  - Anthropic API key, used to write the digest
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
import datetime
from html import unescape
import re
import xml.etree.ElementTree as ET

import requests
from anthropic import Anthropic

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; DailyNewsBot/1.0)"}

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
MODEL = "claude-sonnet-5"


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


def build_digest(headlines_blob: str) -> dict:
    """Ask Claude to turn raw headlines into the bilingual digest + vocab list."""
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    today = datetime.date.today().strftime("%Y/%m/%d")
    today_zh = datetime.date.today().strftime("%Y年%m月%d日")

    prompt = f"""You are preparing a daily bilingual (English + Traditional Chinese) news
briefing email from today's raw headlines below. Output ONLY a single JSON object
(no markdown fences, no commentary) with exactly these keys: "subject", "html", "text".

Requirements for the content:
- Pick the most important 4-6 INTERNATIONAL news stories and 4-6 FINANCIAL/MARKETS
  news stories from the raw headlines provided. Skip duplicates, gossip, or trivial items.
- For each story: first write a short, natural English sentence or two (like a wire
  service would phrase it), then immediately below it a Traditional Chinese (繁體中文)
  translation/summary, then the source name in parentheses.
- After both sections, add a "📚 Today's Vocabulary 今日單字學習" section: pick 8-12
  useful English words/phrases that actually appear in the English sentences you wrote
  above (prefer business/finance/news vocabulary a Chinese-speaking learner might not
  know, e.g. "tariff," "hawkish," "volatility," "ceasefire," "antitrust"). For each,
  give: the word/phrase, its part of speech, a concise Traditional Chinese definition,
  and the exact sentence from the briefing where it appears (bold the word/phrase
  within the sentence). Present this as an HTML table in the html version, and as a
  simple list in the text version.
- Title: "📰 Daily Global & Financial News Briefing 每日國際與金融新聞摘要 — {today_zh}"
- subject should be: "📰 Daily News & Vocabulary 每日國際金融新聞與單字 — {today}"
- "html" must be a complete, self-contained HTML email body (inline styles, no
  external CSS/JS), with clear section headers and a horizontal rule between the
  three sections.
- "text" must be an equivalent plain-text version (no HTML tags), for the plain-text
  fallback part of the email.
- Keep the whole thing readable in about 5-7 minutes. Be factual, no speculation.

Raw headlines (source, title, short summary, link):

{headlines_blob}
"""

    resp = client.messages.create(
        model=MODEL,
        max_tokens=8000,
        messages=[{"role": "user", "content": prompt}],
    )
    raw_text = "".join(block.text for block in resp.content if block.type == "text")
    raw_text = raw_text.strip()
    # Strip accidental markdown code fences if the model adds them anyway.
    if raw_text.startswith("```"):
        raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
        raw_text = re.sub(r"\s*```$", "", raw_text)
    return json.loads(raw_text)


def send_email(subject: str, html: str, text: str):
    to_email = os.environ["TO_EMAIL"]
    from_email = os.environ.get("FROM_EMAIL", "每日新聞摘要 <onboarding@resend.dev>")
    resp = requests.post(
        "https://api.resend.com/emails",
        headers={
            "Authorization": f"Bearer {os.environ['RESEND_API_KEY']}",
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
