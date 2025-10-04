# Implementation Plan: Competition Judge Selector & Booking Assistant

## Goal
Provide a central tool to manage judge selection, communication, and scheduling for club competitions (PDI, print, panels), leveraging existing lists and automating reminders while allowing personal touches.

## Why it matters
- Reduces manual effort and errors when booking judges.
- Ensures variety and avoids overusing the same judges or problematic experiences.
- Keeps the committee informed with up-to-date schedules and reminders.

## Scope
- Data ingestion (EAF/PAGB lists, existing spreadsheet, manual additions).
- Judge directory with filtering and recommendation logic.
- Communication workflow with templated emails and reminder scheduling.
- Reports/briefing packs; integration with club calendar/website for multi-user access.

## Requirements & Acceptance
- **R1:** Create unified judge directory with searchable profile fields and notes.
  **AC1:** Import run populates directory; duplicates resolved; manual additions supported.
- **R2:** Suggest suitable judges based on competition type, travel radius, and booking history.
  **AC2:** For sample competition, system offers multiple candidates with reasons (specialty, availability window, last booked date).
- **R3:** Track communication stages (invite → confirmed → reminder → thank-you) using templates.
  **AC3:** Workflow log shows timestamps and editable entries for a demo judge.
- **R4:** Automate reminder scheduling (emails/calendar entries) while allowing manual adjustments.
  **AC4:** Reminders generated for selected judges, optionally exported to ICS/email drafts.
- **R5:** Produce reports: per-competition pack (bio, instructions), quarterly committee summary.
  **AC5:** Sample report renders in Markdown/PDF showing booked judges, gaps, actions needed.
- **R6:** Support multi-user web access (integrate with club site login), respecting UK data protection.
  **AC6:** Authentication via club website; audit who accessed/edited judge records.

## Inputs & Outputs
- **Inputs:** EAF/PAGB judge lists (PDF/CSV), existing booking spreadsheet, manual entries, email templates (OneNote/Word), competition calendar.
- **Outputs:**
  - Judge directory database (JSON/SQLite).
  - Competition schedule with assigned judges.
  - Email/text templates (pre-populated) ready for personalisation.
  - Reminder exports (ICS, email drafts).
  - Reports (Markdown/PDF) for upcoming competitions and quarterly reviews.

## Design Sketch
1. **Data ingestion module:** Parse PDF/CSV/Excel; manual upload interface; store in central DB with field normalisation.
2. **Judge recommendation engine:** Filter by competition type, travel distance, last booked date, "avoid" flags; suggest variety.
3. **Communication workflow:** Template engine for emails (invite, confirm, reminder, thank-you); track statuses; integrate with club calendar.
4. **Reminder automation:** Allow configuration of lead times; generate ICS files/email drafts (Gmail/Outlook friendly); optionally integrate with website scheduler.
5. **Reporting:** Generate per-competition briefing pack (judge bio, instructions) and committee summaries.
6. **Integration layer:** Extend club website (Django/WordPress?) or provide secure web UI with existing login; manage permissions.

## Tasks & Sizes
1. **T1 (S):** Set up project structure, define data schema (judge profile, booking records, communication log).
2. **T2 (M):** Implement data import pipeline (PDF/CSV/Excel) + manual entry forms.
3. **T3 (M):** Build judge directory UI/CLI and recommendation filters.
4. **T4 (M):** Develop communication workflow (template engine, status tracking); integrate email/ICS export options.
5. **T5 (S):** Implement reminder scheduler with configurable lead times (generate calendar/email artifacts).
6. **T6 (S):** Produce reports (Markdown/HTML/PDF) for competitions and quarterly summaries.
7. **T7 (M):** Integrate with club website login; set up permissions and simple web dashboards.
8. **T8 (S):** Testing with historical data; document workflows; ensure data protection compliance (storage, access logs).

## Risks & Mitigations
- **Data ingestion quality:** PDF parsing may be imperfect → allow manual correction and import logs.
- **Email/calendar integration complexity:** Provide multiple export formats (copy-paste text, ICS) to cover Gmail/Outlook.
- **Multi-user adoption:** Ensure web UI is simple; provide documentation; log edits for accountability.

## Savepoint
- **Current status:** Requirements & plan documented; implementation pending.
- **Next action:** Create branch `feature/judge-booker`; prototype data schema and simple import from existing spreadsheet (T2).
- **Branch:** _feature/judge-booker_ (to be created).
- **Files in flight:** None yet.
- **Open questions:** Confirm club website tech stack, desired email output formats, process for storing judge contact consent.
