# Feature Request: Competition Judge Selector & Booking Assistant

## 0) TL;DR (1–2 sentences)
Create a system that consolidates judge information, suggests balanced selections, and automates reminders/emails for booking eight competitions per year (PDI, print, panels).

---

## 1) Why / Outcome
- **Why now:** Current process relies on scattered PDFs, spreadsheets, and manual emails; hard to avoid repeat bookings or manage reminders.
- **Definition of success:** For each competition slot, the organiser sees recommended judges with context, tracks communications, and the system nudges them with timely reminder emails/calendar entries.

---

## 2) Scope
- **In scope:**
  - Ingest judge data (EAF/PAGB booklets, existing spreadsheet, ad-hoc notes).
  - Provide filtering/recommendation (travel distance, specialty, recent usage, difficulty flags).
  - Manage communication timeline (invite, confirmation, reminders, thank-you) with template support.
  - Generate reports (per-competition briefing packs, quarterly committee updates).
  - Integrate with existing club calendar/website logins where feasible.
- **Out of scope:**
  - Complete availability tracking (still requires email confirmation).
  - Automated contact directory for non-judges (guest speakers handled later, though process may be similar).
  - Full email client replacement (manual edits/personalisation still expected).

---

## 3) Requirements
### 3.1 Functional
- [ ] F1 — Import judge roster from EAF/PAGB sources + existing spreadsheet (including manual additions) and normalise contact/profile data.
- [ ] F2 — Provide judge filtering/suggestions based on competition type (PDI/print/panel), travel radius, past usage, and any "avoid" flags.
- [ ] F3 — Maintain communication workflow with templated emails (invite, confirm, info pack, reminder, thank-you) while allowing personal edits.
- [ ] F4 — Automate reminder scheduling (calendar entries, email reminders) for key milestones per judge/competition.
- [ ] F5 — Generate reports: upcoming judge schedule, per-competition packs (bio, requirements), quarterly booking summary.
- [ ] F6 — Support multi-user access (club website integration) with simple permissions.

### 3.2 Non-functional (brief)
- **Performance:** Data set is small; responsiveness matters more than optimisation.
- **Reliability:** Safeguard judge contact data; log communications sent; indicate pending actions.
- **Compatibility:** Optionally integrate with club website (existing login), support Gmail/Outlook email templates; align with UK data protection.
- **Persistence:** Store judge profiles/history locally or on club-managed server with backups.

**Acceptance Criteria (testable)**
- [ ] AC1 — Initial import creates judge directory with fields (name, region, specialty, travel, contact, notes, last booked date).
- [ ] AC2 — For a given competition slot, system suggests at least three viable candidates and shows reasons.
- [ ] AC3 — Communication workflow logs invite sent, confirmation received, reminder scheduled, thank-you sent for a sample judge.
- [ ] AC4 — Calendar entry/email reminder generated using the provided templates for a booked judge.
- [ ] AC5 — Quarterly report summarises upcoming judges, gaps, and recent bookings without duplicates.

---

## 4) Effort & Priority (quick gut check)
- **Effort guess:** Large (data integration + scheduling + reminder logic + website hooks).
- **Priority / sequencing notes:** Essential for smooth competition planning; potential future extension to speaker bookings.

---

## 5) Open Questions / Notes
- Best approach to scrape/parse EAF/PAGB booklets (PDF vs. manual entry).
- How to integrate with club website (API, shared database, authentication).
- Email sending mechanism (SMTP credentials, Outlook/Gmail API, manual copy/paste).
- Data retention policy for judge contact details.

---

## 6) Agent Helper Prompt
"Read this feature request. Ask me concise questions about data import formats, reminder cadence, and website integration before proposing a design. Then highlight missing requirements or risks."

---

## 7) Links (optional)
- **Sources:** EAF judge booklet (PDF), PAGB judge list (if available), existing bookings spreadsheet, email templates in OneNote/Word.
- **Related processes:** Club website calendar, judge preview login workflow.
