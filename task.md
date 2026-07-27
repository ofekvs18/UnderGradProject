# Claude Code Prompt — Coprolite Micro-CT Article (Thesis Chapter)

Paste everything below into Claude Code in the project's root directory. Fill in the **STATUS / TO-FILL** section at the bottom before running. Once filled, run once, uninterrupted.

---

## 0. Role & Context

You are running an end-to-end scientific writing pipeline for a graduate thesis chapter on **micro-CT scanning of coprolites for archeobotanical knowledge**. You have full read access to this repository. Two source locations matter:

- `docs/article/` — the article source, already in **LaTeX**: `.tex` file(s) for the literature review and methodology, a `.bib` file for references, and `.jpg`/`.png` figures already placed/referenced. There is no separate instructions file — the existing `.tex` structure (sections, preamble, document class, formatting conventions already in use) *is* the spec to follow and extend. **Read all `.tex` files and the `.bib` file first, in full, before writing anything.**
- The project data directory (path to be confirmed below) — micro-CT scans, processed segmentation results, quantitative outputs, figures, and any existing analysis scripts/notebooks.

**Output format: LaTeX, staying inside the existing `docs/article/` project.** Do not convert anything to Word/PDF as the deliverable — the final artifact is an updated, complete, compilable `.tex` document (using the existing `.bib` for citations and existing/new figures in the same directory) that Ofek can keep editing directly. If a LaTeX toolchain is available, compile it at the end as a correctness check (catches broken refs/citations/figures) — but the `.tex` source is the deliverable, not the PDF.

Do not ask me questions mid-run. If you hit a genuine blocker (a referenced file doesn't exist, a required number can't be found anywhere, a script fails and can't be fixed after 2 attempts), stop, log it clearly in `article_run_log.md`, and move to the next task rather than halting the whole pipeline.

## 1. Step-by-Step Plan

**Phase 0 — Discovery (read-only, ~15–30 min)**
1. Read every `.tex` file in `docs/article/` in full (including the preamble/document class — note packages already loaded, custom macros, section structure, heading conventions). Read the `.bib` file and note what references already exist and in what key format. List the `.jpg`/`.png` figures present and check whether each is actually `\includegraphics`'d somewhere in the `.tex` already, or sitting unused.
2. From the `.tex` content itself, infer the implicit spec: expected chapter length (based on what's drafted so far and any comments/TODOs in the source), citation style (from `\bibliographystyle` and how `\cite` is used), section/subsection conventions already established. Treat this as the authoritative structure to continue in — don't introduce a different LaTeX style or package set than what's already there.
3. Note what's already written (literature review, methodology) vs. what's missing (presumably Results, Discussion, Conclusion — confirm against what actually exists in the `.tex` files).
4. Inventory the project data directory: list all result files, figures, tables, scripts, and datasets. Build a manifest (`data_manifest.md`) mapping each expected result (per the methodology section) to an actual file, or flagging it as **missing**.
5. Produce a gap report: what's fully analyzed, what's partially done, what's missing entirely.

**Phase 1 — Planning**
6. Generate a detailed outline for the remaining chapter content (sections/subsections still needed, target length per section, which figures/tables go where and their `\label`/`\ref` keys, which `.bib` entries are needed where — flag any citation needed that isn't in the `.bib` yet) based on the existing `.tex`/`.bib` + data manifest.
7. Write this outline to `article_outline.md` and treat it as the shared plan all writer subagents follow.

**Phase 2 — Gap Filling (only if data is incomplete)**
7. For any missing-but-scriptable result (e.g., a summary statistic, a derived figure from existing processed data), attempt to generate it from existing scripts/data. Log what was generated and how.
8. For any missing result that requires new analysis you cannot responsibly perform (e.g., needs new segmentation, new scanning, human judgment call), do **not** fabricate it. Insert a clearly marked placeholder: `[[GAP: description — needs Ofek's input]]` and log it in `article_run_log.md`. This must never silently become a real-looking number in the final draft.

**Phase 3 — Writing (multi-agent, see Section 2 below)**
9. Draft section-by-section per the outline, using the Writer agent(s).
10. Every claim tied to a number, figure, or data result must cite its exact source file/line or figure ID internally (as an inline `<!-- source: ... -->` comment) — this is what the Fact-Checker uses in Phase 4, and it's stripped before final export.

**Phase 4 — Fact-Check & Consistency Loop**
11. Fact-Checker agent reviews the full draft against: (a) the data manifest and source files, (b) the literature review for citation accuracy, (c) internal consistency (numbers/terms/claims matching across sections), (d) the instructions .docx for structural/formatting compliance.
12. Any issue found is sent back to the Writer agent with a specific correction request. Repeat until the Fact-Checker reports zero issues or a defined max of 3 loops — whichever comes first. If 3 loops pass with unresolved issues, log them explicitly rather than looping forever.

**Phase 5 — Formatting & Export**
13. Apply final formatting per the instructions .docx (headings, citation style, figure numbering, page limits).
14. Export to `.docx` using the docx skill, matching the required template/structure.
15. Strip all internal source-tracking comments and `[[GAP...]]` markers into a separate `open_items.md` for Ofek to review — the exported .docx should read cleanly, but every remaining gap must also be visible in this summary file.

**Phase 6 — Final Verification**
16. Run a final full read-through pass: word count vs. limit, all required sections present, all figures/tables referenced and numbered correctly, all citations present in the reference list.
17. Produce `article_run_log.md` summarizing: what was written, what was fact-checked and corrected, what gaps remain, final word count, and a go/no-go recommendation for submission.

## 2. Agentic Architecture

This follows a **planner → parallel specialist workers → critic/verifier loop** pattern, similar to Anthropic's orchestrator-worker multi-agent research pattern, Stanford's STORM (outline-first, then section-level drafting), and generate-critique-revise loops used in AI co-scientist–style systems. Adapted here for a single-author thesis chapter (no need for multi-perspective debate — the emphasis is data fidelity and consistency, not adjudicating conflicting viewpoints):

- **Orchestrator/Planner** — owns `article_outline.md` and `article_run_log.md`. Sequences phases, dispatches writing tasks per section, decides when the fact-check loop is done, never writes prose itself.
- **Data-Analyst agent** — runs only in Phase 2. Reads existing scripts/notebooks, regenerates missing derived results where safely possible, never invents data.
- **Writer agent(s)** — one per major section (Intro, Methods, Results, Discussion, Conclusion) or sequential if you'd rather keep voice consistent — draft prose from the outline + source material, tag every factual claim with its source.
- **Fact-Checker/Consistency agent** — the critic. Cross-references every tagged claim against source files and the literature review, checks citation format, flags contradictions between sections. Does not fix issues itself — sends structured feedback back to the relevant Writer agent.
- **Formatter/Export agent** — final pass only, applies the .docx template and produces the deliverable.

Keep Writer and Fact-Checker as separate agent invocations (not the same context) — this is the part that most improves output quality in these architectures, since a model checking its own freshly-written claims tends to rubber-stamp them.

## 3. Skills/Plugins Worth Having Enabled

- **docx** — required for reading the instructions .docx and exporting the final chapter.
- **pdf** — only if any source literature is PDF-only and needs text/figure extraction.
- **xlsx** — only if any processed data/results live in spreadsheets you'll pull numbers or tables from.

Fact-checking was scoped to internal consistency + citation accuracy only (no live literature lookups), so no web-search/PubMed tooling is required for this run. If that changes later, add literature-search access and a step in Phase 4 for external verification.

## 4. Open Questions / Assumptions Baked Into This Plan

- Assumes `docs/article/` contains the actual literature review + methodology + instructions .docx today — confirm paths below.
- Assumes the project data directory path (not yet specified — fill in below).
- Assumes citation style, word/page limit, and required structure come entirely from the instructions .docx — if your advisor gave verbal/email instructions not in that file, add them below.
- Assumes "verified" = internal consistency + citation accuracy, not external fact-checking against new literature searches.
- Assumes full autonomy: no mid-run check-ins, single pass through Phase 4's loop (max 3 iterations), then hand-off.
- Assumes gaps in data get flagged, never fabricated.

---

## STATUS / TO-FILL (fill this in, then run)

| Item | Value |
|---|---|
| Project data directory path | `data/` and `results/` |
| Instructions .docx filename (in `docs/article/`) | instructions.docx |
| Deadline (exact date/time) | 7 am, July 28th |
| Word/page limit (if not in .docx) | see .docx |
| Citation style (if not in .docx) | see .docx |
| Advisor/program name (for chapter framing) | Prof. Noa Dagan, Dr. Seffi Cohen, Dr. Liat Antwarg  |
| Known missing data/results (if you already know) | let the gap report find it |
| Anything the instructions .docx does NOT cover that matters (e.g. specific formatting quirks, required appendices) | None |
| Any sections already drafted elsewhere Claude should incorporate as-is | *(fill in or "none")* |