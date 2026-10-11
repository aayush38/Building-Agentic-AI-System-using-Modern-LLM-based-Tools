"""Drive the real Streamlit app with Edge and save screenshots for the report."""
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

OUT = Path(__file__).parent / "assets"
URL = "http://localhost:8601"
PROVIDER = sys.argv[1] if len(sys.argv) > 1 else "groq"


def pick(page, label, option):
    page.get_by_label(label, exact=True).first.click()
    page.get_by_role("option", name=option, exact=True).click()


def alerts(page):
    return page.locator("[data-testid=stAlert]").all_inner_texts()


with sync_playwright() as p:
    browser = p.chromium.launch(channel="msedge", headless=True)
    page = browser.new_page(viewport={"width": 1400, "height": 1300}, device_scale_factor=1.5)
    page.goto(URL)
    page.get_by_role("button", name="Run agent").wait_for(timeout=60000)
    pick(page, "Provider", PROVIDER)
    pick(page, "Dataset", "data/churn.csv")
    page.wait_for_timeout(1500)
    page.screenshot(path=OUT / "shot1_input.png")
    print("saved shot1_input")

    page.get_by_role("button", name="Run agent").click()
    page.wait_for_timeout(4000)
    page.screenshot(path=OUT / "shot2_live_trace.png")
    print("saved shot2_live_trace")

    # first turn finished once its user+assistant chat messages are rendered after the rerun
    chat = page.locator("[data-testid=stChatMessage]")
    for _ in range(240):
        if chat.count() >= 2 and page.get_by_text("Tuning and feature importance").count():
            break
        if alerts(page) and any("Rate limit" in a or "failed" in a for a in alerts(page)):
            print("RUN FAILED:", alerts(page)); sys.exit(1)
        page.wait_for_timeout(1000)
    else:
        print("timed out waiting for first run"); sys.exit(1)
    page.set_viewport_size({"width": 1400, "height": 4300})  # tall so nothing is clipped
    page.wait_for_timeout(2500)
    page.screenshot(path=OUT / "shot3_result.png")
    print("saved shot3_result")

    page.get_by_text("Reasoning trace").first.click()
    page.wait_for_timeout(1500)
    page.locator("[data-testid=stChatMessage]").nth(1).screenshot(path=OUT / "shot3b_trace_expanded.png")
    print("saved shot3b_trace_expanded")
    page.get_by_text("Reasoning trace").first.click()  # collapse again

    page.set_viewport_size({"width": 1400, "height": 1300})
    box = page.get_by_placeholder("Ask a follow-up")
    box.fill("Why not random forest? Answer from the results you already have.")
    box.press("Enter")
    for _ in range(180):
        if chat.count() >= 4:
            break
        if alerts(page) and any("Rate limit" in a or "failed" in a for a in alerts(page)):
            print("FOLLOW-UP FAILED:", alerts(page)); break
        page.wait_for_timeout(1000)
    else:
        print("timed out waiting for follow-up")
    print("chat messages:", chat.count())
    page.set_viewport_size({"width": 1400, "height": 5200})
    page.wait_for_timeout(2500)
    page.screenshot(path=OUT / "shot4_followup.png")
    if chat.count() >= 4:
        chat.nth(2).screenshot(path=OUT / "shot4b_followup_question.png")
        chat.nth(3).screenshot(path=OUT / "shot4c_followup_answer.png")
    print("saved shot4")
    browser.close()
