# Design: AI Video Lectures per Section (Tavus)

**Status:** Proposal · **Owner:** Siddharth · **Scope:** 15 section lectures (v1), per-problem walkthroughs and a live tutor later

## 1. Goal

Each section page gets a short video lecture (3–5 min) that explains the section's theory before the learner starts coding. A Tavus replica (an AI presenter) delivers it. The lecture complements the written introduction in `src/data/sections.ts`. It does not replace it.

Non-goals for v1: live conversation, per-problem videos, user accounts.

## 2. Hard constraint: the API key never reaches the browser

The site is a static GitHub Pages build. Anything in the JS bundle is public, including any env var prefixed `VITE_`, which Vite inlines at build time. So:

- Videos are **pre-generated offline** by a Node script that reads `TAVUS_API_KEY` from the environment (a local shell or a GitHub Actions secret).
- The site ships only a **manifest of public video URLs**. No key, no Tavus API calls from the client.
- Never name the variable `VITE_TAVUS_API_KEY`. Never commit `.env` (add `.env*` to `.gitignore`).

## 3. Architecture

```
content/lectures/<section-id>.md      ← narration script (human-reviewed, committed)
        │
        ▼
scripts/tavus/generate-lectures.mjs   ← offline; reads TAVUS_API_KEY
        │  POST /v2/videos  (replica_id + script)
        │  poll GET /v2/videos/{id} until status = "ready"
        ▼
src/data/lectures.json                ← manifest committed to repo
        │
        ▼
<LecturePlayer sectionId=… />         ← SectionPage, renders only if manifest has an entry
```

### 3.1 Narration scripts: `content/lectures/<section-id>.md`

Section intros are written for reading. They contain code blocks, tables and LaTeX-ish formulas that sound terrible read aloud. Each lecture gets its own **spoken** script:

```markdown
---
section: transformers
replica: default          # key into scripts/tavus/config.json
target_minutes: 4
chapters:                 # optional, for the chapter list under the player
  - { t: 0,   title: "Why attention?" }
  - { t: 55,  title: "Queries, keys, values" }
  - { t: 150, title: "Scaling and softmax" }
  - { t: 210, title: "Multi-head attention" }
---
Imagine you're reading the sentence "the animal didn't cross the street because it was too tired"...
```

Authoring rules:
- Say formulas in words ("divide by the square root of d-k"), and keep the symbolic form in the on-page intro.
- No code read aloud. Say "you'll implement this in the first problem" and link it.
- About 130–150 spoken words per minute, so 4 minutes is roughly 550 words.
- End with a hand-off: "Start with Scaled Dot-Product Attention below."
- Workflow: Claude drafts each script from the section intro and problem list, then you review and edit before generating. The scripts are educational content under the "All Rights Reserved" part of `LICENSE`.

### 3.2 Generator: `scripts/tavus/generate-lectures.mjs`

```
node scripts/tavus/generate-lectures.mjs              # generate new/changed lectures
node scripts/tavus/generate-lectures.mjs transformers # one section
node scripts/tavus/generate-lectures.mjs --dry-run    # show what would be generated + est. minutes
```

Behaviour:
1. For each `content/lectures/*.md`, compute `sha256(script body + replica_id)`.
2. Skip it if `lectures.json` already has a `ready` entry with the same hash. This makes the script idempotent, so you never pay twice for unchanged scripts.
3. `POST https://tavusapi.com/v2/videos` with header `x-api-key: $TAVUS_API_KEY` and body:
   ```json
   { "replica_id": "<from config>", "script": "<body>", "video_name": "mlcl-<section>-<hash8>",
     "background_url": "<optional: section landing page for a screen-share style background>" }
   ```
4. Poll `GET /v2/videos/{video_id}` with backoff (every 15s, give up after 30 min) until `status` is `ready` or `error`.
5. Write the entry to `src/data/lectures.json` after each video, so a crash doesn't lose finished work.
6. Print a summary: generated, skipped and failed counts, plus total minutes.

Use `--dry-run` before the first real run to sanity-check scope and cost.

*Verify the field names (`hosted_url`, `stream_url`, `download_url`, `thumbnail_url`, `status`) against the current Tavus API reference when implementing. This design was written without live API access.*

### 3.3 Manifest: `src/data/lectures.json`

```json
{
  "transformers": {
    "videoId": "abc123",
    "hostedUrl": "https://videos.tavus.io/video/abc123",
    "streamUrl": "https://.../abc123.m3u8",
    "thumbnailUrl": "https://.../abc123.jpg",
    "durationSec": 241,
    "scriptHash": "9f2c…",
    "generatedAt": "2026-11-08T20:00:00Z",
    "chapters": [{ "t": 0, "title": "Why attention?" }]
  }
}
```

Typed as `Record<string, Lecture>` in `src/types/index.ts`. A section without an entry renders exactly as today.

### 3.4 Frontend: `LecturePlayer`

Location: `src/components/Lecture/LecturePlayer.tsx`, rendered on `SectionPage` above the collapsible introduction.

- **Collapsed by default:** a card with the thumbnail, a "▶ Watch the 4-min lecture" label and the duration. The player loads only on click, so no third-party iframe or video weight lands on page load. This keeps Lighthouse scores and privacy clean.
- **Player:** a `<video>` with `streamUrl`/MP4 if available (needs hls.js for `.m3u8` outside Safari), otherwise an `<iframe src={hostedUrl}>`. Start with the iframe because it needs zero dependencies.
- **Chapters** under the player (seekable when using `<video>`).
- **Transcript** in a "Read transcript" disclosure, rendered from the narration script. This covers accessibility and SEO, and works when the video is blocked.
- **Progress:** add `lecturesWatched: Record<sectionId, true>` to `ProgressContext`, set at about 80% watched (`<video>` `timeupdate`) or on click (iframe). Show a small ✓ on section cards and in the sidebar.
- **Privacy page:** add a line that the lecture player loads video from Tavus's CDN once you click play.

### 3.5 Video hosting

| Option | Pros | Cons |
|---|---|---|
| **Tavus `hosted_url` (v1)** | No extra infra | Depends on Tavus plan and retention; Tavus branding/player |
| Download MP4 to Cloudflare R2 / Bunny | Cheap, own player, durable | Small extra step in the generator |
| YouTube (unlisted/public) | Free, discovery, captions | YouTube embed and tracking |
| Commit MP4s to `gh-pages` | — | **No**: repo bloat, 100 MB file limit |

Recommendation: use Tavus-hosted for v1. If you rely on lectures long-term, add a `--mirror r2` flag that downloads `download_url` and uploads it to R2, so you aren't locked to Tavus retention.

### 3.6 Automation (optional)

A `.github/workflows/lectures.yml` with `workflow_dispatch` only, never on push, so spend stays manual:
- Store `TAVUS_API_KEY` as a repo **Actions secret**.
- It runs the generator and opens a PR with the updated `lectures.json`.

## 4. Cost and scope

- v1: 15 sections × about 4 min = **about 60 minutes** of generated video. Check the minutes included in your Tavus plan.
- Regenerating happens only when a script changes, thanks to the hash.
- v2 (per-problem 60–90s walkthroughs): 78 problems × about 1.25 min = about 100 more minutes. Do this only after watch-rate data from v1 shows learners use the lectures. Tracking this needs analytics, which the site doesn't have yet, so a privacy-friendly counter like Plausible would be a prerequisite.

## 5. Phase 3 idea: "Ask the tutor" (Tavus Conversational Video)

Tavus also offers real-time conversational replicas: `POST /v2/conversations` with a persona returns a `conversation_url` you embed. A "Stuck? Talk it through" button on the problem page could open a live tutor primed with the problem description, the learner's current code and the latest test failures.

This **requires a backend**: a small Cloudflare Worker or Vercel function that holds the key, creates conversations, enforces per-visitor rate limits and caps session length. It is billed per minute, so it needs abuse protection (Turnstile + quotas) before you launch. It's a separate project. Don't start it until v1 is live.

## 6. Implementation plan

1. `content/lectures/` with Claude-drafted scripts for 2 sections (e.g. NumPy Fundamentals and Transformers). You review them.
2. `scripts/tavus/generate-lectures.mjs` + `config.json` (replica id) + `--dry-run`.
3. Generate those 2. Watch them. Tune script style and replica.
4. `LecturePlayer`, manifest type, SectionPage integration, transcript, privacy-page line.
5. Write the remaining 13 scripts, review, generate, ship.
6. (Optional) Actions workflow, R2 mirroring, watched-state in progress.

## 7. Open questions

- Which replica? A stock Tavus replica, or a personal replica trained on your own footage (more on-brand, needs consent recording)?
- Presenter only, or a background showing the section page or diagrams (`background_url`)?
- Language: English only for v1?
