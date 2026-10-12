# Chain 2's site is flagged by Google Safe Browsing; Prolific paused its study

From the local Sherlock session, 2026-10-12 ~12:30 PDT, for the driver session and the PI.

## What happened

- **The study:** chain 2's experiment-1 study (`6acd20b6…`, published 11:02) was **paused by
  Prolific** at 196 of 200 places (199 submissions). The dashboard says it was "paused by Prolific
  because of feedback we received from participants … and cannot be restarted".
- **The feedback:** good ratings (ease 4.8, clarity 4.8, reward 4.6). The only written feedback
  is one cancelled participant: "I got a malware warning" (Technical issue).
- **Google Safe Browsing** (transparencyreport.google.com, checked 12:30):

  | site | status |
  |---|---|
  | `auto-psych-2c5da-rsa-c0.web.app` | No unsafe content found |
  | `auto-psych-2c5da-rsa-c1.web.app` | No unsafe content found |
  | **`auto-psych-2c5da-rsa-c2.web.app`** | **"Some pages on this site are unsafe … Try to trick visitors into sharing personal info or downloading software"** (deceptive / phishing) |

- **The submissions look normal:** median 115 s (chain 1: 119 s), 3 returned, 42 held (41 for
  speed). Nothing points to a broken page.
- **A likely cause** (a guess): the page is on a brand-new `*.web.app` subdomain (first used
  today), takes an id from the URL and redirects to Prolific with a completion code. Safe Browsing
  heuristics sometimes flag new Firebase Hosting sites like this.

## What happens now

- **Chain 2's experiment 1 is fine.** Its job (47523801_2) waits until about 14:03, sees the study
  is not recruiting, leaves it as it is, and models the ~196 completers.
- **Chain 2's experiments 2 and 3 will deploy to the same flagged site.** Experiment 2's study
  publishes after experiment 1's inner loop and experiment 2's design, roughly 10–12 h from now,
  and would likely be warned and paused again. Repeated pauses may also affect the account.
- **Chains 0 and 1 are clean for now.** But the same flag could hit any site, as c1 is also new.

## Options (driver / PI)

1. **Request a review** in Google Search Console. This takes the Firebase owner
   (michaelcfrank@gmail.com):
   - add the site as a URL-prefix property and verify it, by an HTML file deployed to that site
     (no DNS for `web.app`);
   - then Security issues → Request review.

   Reviews usually take a day or more, so this may not clear before chain 2's experiment 2.
2. **Move chain 2 to another Hosting site** for experiments 2–3 (a code or config change to the
   site name, logged as a campaign fix). A new site could be flagged in turn.
3. **Hold chain 2 after experiment 1.** I can `scancel` 47523801_2 after its experiment-1 inner
   loop, before experiment 2's collect deploys, and resume it with the same command once the site
   is cleared or moved. Experiment 1's stages are kept.

**Unless told otherwise, I'll do option 3 as a safeguard:** cancel chain 2 once experiment 1's
inner loop has ended (during experiment 2's design, before any deploy), so no new study goes out
on the flagged site.

Separately, it may be worth checking each site's Safe Browsing status before every publish.
