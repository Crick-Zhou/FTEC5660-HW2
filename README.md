# FTEC5660 Homework 2: CV Verification Agent

Build a LangChain agent that reads each CV in a folder, looks the candidate up
on our SocialGraph MCP server (mock LinkedIn and Facebook), and outputs a
reliability score in [0, 1] for each CV.

A CV is **valid** (label `1`) when its claims agree with the candidate's
social media profiles. It **has a discrepancy** (label `0`) when it contains
problems such as an inflated job title, shifted dates, an upgraded degree, a
fake school or employer, a wrong location, or made-up skills. Differences in
wording only ("Bachelor of Science" vs `BSc`, "UI/UX Design" vs `UI/UX`,
"Senior Engineer" for an `Engineer` role with seniority `senior`, listing fewer
skills) are not discrepancies.

## Student task

Fill in the two functions in `hw2.py` that contain `### YOUR CODE HERE`. You
may add imports, constants and helper functions above them, but do not change
the provided code below them:

- `build_agent(tools)` creates your agent from the MCP tools.
- `score_cvs(agent, cvs)` runs the agent on every CV and returns
  `{file_name: score}`, one float in [0, 1] per CV.

A score above `0.5` means "valid"; `0.5` or below means "has discrepancy". You
may use a single tool-calling agent, multiple agents, reflection, or a
combination. Do not hard-code filenames, names, or public answers; grading
uses unseen CVs.

## MCP server

The server is hosted at `https://ftec5660.ngrok.app/mcp`. `hw2.py` already
connects to it and passes you these tools:

| Tool | Purpose |
| --- | --- |
| `search_facebook_users(q, limit, fuzzy)` | find Facebook users by display name |
| `get_facebook_profile(user_id)` | full Facebook profile |
| `get_facebook_mutual_friends(user_id_1, user_id_2)` | mutual friends of two users |
| `search_linkedin_people(q, location, industry, limit, fuzzy)` | search LinkedIn by name, skill, or title |
| `get_linkedin_profile(person_id)` | full LinkedIn profile (experience, education, skills) |
| `get_linkedin_interactions(person_id)` | post and like statistics |

Print `tool.name`, `tool.description`, and `tool.args` for the full schemas.

## Setup and public test

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
echo "DEEPSEEK_API_KEY=your_key_here" > .env
python3 hw2.py --cv-folder public_test
```

Use your own DeepSeek API key (from https://platform.deepseek.com).

The program creates `results.csv` in the current directory. Its columns are
`cv`, `score`, and `correctness`. The public labels are in
`public_test/ground_truth.json`; every CV with label `0` also has a `reason`
explaining the discrepancy, and `results.csv` shows it when your agent misses
one. The starter returns no score (`None`) for every CV so it runs before you
add any API code; missing scores count as incorrect.

The required model is `deepseek-v4-flash`, and all evidence must come from the
MCP server (no web search). The server is shared by the whole class: keep at
most about 3 CVs in flight at a time (e.g. `asyncio.Semaphore(3)`). We grade
with our own API key, so never put a key in your code; `.env` must stay out of
git.


## Task 2: Fool the Verifier

We added a target candidate, **Kelly Tsang** (LinkedIn `person_id` 10001,
Facebook `user_id` 10001, display name "Kel Tsang"). Her true CV is
`task2/target_cv.pdf`.

Write one CV for Kelly Tsang with at least one false or embellished detail
that our 5 verifier agents (same function as Task 1) still score above 0.5.
Each fooled agent is worth 4 points; a CV without a real false or embellished
detail earns 0. Attack your own Task 1 agent first:

```bash
cp my_attack.pdf task2/adversarial_cv.pdf
python3 hw2.py --cv-folder task2
```

`results.csv` shows your agent's score for both CVs (they are not graded, as
`task2/` has no `ground_truth.json`). Only the PDF is the attack surface; do not
attack the MCP server. See the homework description for the full rules.

## Homework 2 solution:

### Task 1
My verifier uses a finder agent to search and inspect plausible profiles, resolving candidates who may share a name. A second model call checks each CV claim against the selected LinkedIn profile, including job titles and seniority, dates, education, location, and every listed skill. Clear discrepancies receive 0.1, matching claims receive 0.9, and unresolved or failed checks receive 0.5. The implementation limits concurrency to two CVs and retries transient tool failures. On the seven public test CVs, the latest complete run scored 7/7 correctly.

### Task 2
I used a PDF visual/text extraction mismatch. The visible one-page CV claims that Kelly Tsang is a **Senior Manager at AIA**, while her SocialGraph LinkedIn profile lists **Manager** with **mid** seniority and Facebook also lists **Manager**. This makes the visible CV genuinely embellished. The visible page is rasterized, while an invisible selectable text layer contains the accurate Manager claim and otherwise matching details. The homework's MarkItDown PDF loader extracts the text layer, so a text-only verifier sees claims consistent with the profile. In one successful local test, my Task 1 agent gave both the adversarial CV and the true target CV a score of 0.9. This does not establish how the five undisclosed graders will respond, especially if they inspect the rendered page or compare it with the extracted text.

### Task 3: Reflection on the AI paradoxes

This homework made three AI paradoxes concrete for me.

**The competence paradox.** My verifier classified all seven public CVs correctly, which made its scores seem trustworthy. Yet it also scored a visibly embellished CV as reliable. The model compared the extracted PDF text with the social profiles correctly; the extracted text simply did not match the visible page. Better reasoning over the wrong representation can produce a more convincing wrong answer. I learned that an accuracy figure describes performance on a particular test, not a guarantee that the whole workflow is reliable.

**The dual-use paradox.** The same AI that helps an employer check claims can help an applicant discover how to evade the check. I used my verifier's score as feedback while testing the adversarial CV. Faster and cheaper verification also makes repeated attack experiments faster and cheaper. This creates a cycle in which a stronger defender may invite a more carefully adapted deception.

**The automation paradox.** AI can reduce the time spent comparing CVs with online profiles, but responsible use of that speed requires more attention to the evidence behind each score. A system should compare the rendered PDF with its extracted text, record which source supports each claim, and flag identity ambiguity or conflicting evidence for human review. Even a matching social profile is not perfect ground truth: it can be incomplete or outdated. My conclusion is that AI should help people inspect evidence and uncertainty, while people remain accountable for decisions that affect applicants.
