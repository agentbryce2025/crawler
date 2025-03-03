import re
import time
import io
from typing import Any, Dict
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from dotenv import load_dotenv
from duckduckgo_search import DDGS

import requests
import pdfplumber
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
import random
import string
from faker import Faker  # For realistic random data

fake = Faker()

load_dotenv()

# --------------------------------------------------------------------------------
# Browser Class
# --------------------------------------------------------------------------------
class Browser:
    def __init__(self):
        options = uc.ChromeOptions()
        # options.add_argument("--headless")  # Uncomment if headless mode is desired
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        self.driver = uc.Chrome(options=options)
        self.wait_time = 5  # default wait time in seconds

    def go_to_url(self, url: str):
        self.driver.get(url)
        WebDriverWait(self.driver, self.wait_time).until(
            lambda d: d.execute_script("return document.readyState") == "complete"
        )

    def get_page_source(self) -> str:
        time.sleep(1)
        return self.driver.page_source

    def quit(self):
        self.driver.quit()


browser = Browser()

# --------------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------------
@tool
def internet_searcher(query: str) -> str:
    """
    Searches DuckDuckGo for the given query string, returning up to 10 results.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=10))
        if not results:
            return "No results found."
        response = f"Search results for '{query}':\n"
        for i, result in enumerate(results, 1):
            response += (
                f"{i}. Title: {result['title']}\n"
                f"   Summary: {result['body']}\n"
                f"   Link: {result['href']}\n"
            )
        return response
    except Exception as e:
        return f"Error performing search: {str(e)}"


@tool
def web_scraper(url: str) -> str:
    """
    Navigates the Undetected ChromeDriver to the given URL and returns the page source.
    """
    try:
        browser.go_to_url(url)
        content = browser.get_page_source()
        if content and len(content.strip()) > 0:
            return content
        else:
            return "No content could be extracted from the page."
    except Exception as uc_error:
        return f"Error scraping with undetected-chromedriver: {str(uc_error)}"


@tool
def pdf_scraper(url: str) -> str:
    """
    Downloads a PDF from the given URL and extracts its text contents.
    """
    try:
        response = requests.get(url, timeout=15)
        if response.status_code != 200:
            return f"Failed to download PDF, status code: {response.status_code}"
        pdf_text = ""
        with pdfplumber.open(io.BytesIO(response.content)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    pdf_text += page_text + "\n"
        return pdf_text.strip() if pdf_text.strip() else "No text extracted from PDF."
    except Exception as e:
        return f"Error parsing PDF: {str(e)}"


@tool
def go_to_url_tool(url: str) -> str:
    """
    Directs the browser to navigate to the specified URL (no content extraction).
    """
    browser.go_to_url(url)
    return f"Navigated to {url}"


@tool
def get_page_source_tool(arg: str = "") -> str:
    """
    Returns the current page source from the undetected ChromeDriver.
    """
    return browser.get_page_source()


@tool
def shutdown_browser_tool(arg: str = "") -> str:
    """
    Shuts down the browser session (closes the undetected ChromeDriver).
    """
    browser.quit()
    return "Browser has been shut down."


@tool
def fill_every_form_tool(arg: Dict[str, str] = None) -> str:
    """
    Fills every form on the current page with data extracted by the LLM from user input,
    or realistic random text if no data is provided, and dynamically identifies/submits.
    Args:
        arg (Dict[str, str], optional): Dictionary of field names and values extracted by the LLM.
    Returns:
        str: Detailed summary of actions taken.
    """
    driver = browser.driver
    wait = WebDriverWait(driver, browser.wait_time)

    def guess_input_value(input_elem, custom_data=None):
        """Generate input value using LLM-extracted data if provided, otherwise use realistic random data."""
        input_type = (input_elem.get_attribute("type") or "").lower()
        name_id_placeholder = (
            (input_elem.get_attribute("name") or "").lower() +
            (input_elem.get_attribute("id") or "").lower() +
            (input_elem.get_attribute("placeholder") or "").lower()
        ).strip()
        pattern = input_elem.get_attribute("pattern")

        if custom_data:
            field_keys = [key.lower() for key in custom_data.keys()]
            if any(key in name_id_placeholder for key in field_keys):
                for key, value in custom_data.items():
                    if key.lower() in name_id_placeholder:
                        return value

        if pattern:
            if "10" in pattern:
                return ''.join(random.choices(string.digits, k=10))
            elif "[a-zA-Z]" in pattern:
                return fake.word()
        if "email" in name_id_placeholder:
            return fake.email()
        if "phone" in name_id_placeholder or "tel" in input_type:
            return fake.phone_number()
        if input_type == "password":
            return fake.password(length=12)
        if input_type == "date":
            return fake.date()
        if any(x in name_id_placeholder for x in ["name", "user"]):
            return fake.name()
        if any(x in name_id_placeholder for x in ["message", "comment", "description"]):
            return fake.paragraph(nb_sentences=2)
        if "address" in name_id_placeholder:
            return fake.address()
        return fake.text(max_nb_chars=20)

    def find_parent_clickable(element):
        """Find the nearest clickable parent (e.g., button or div)."""
        current = element
        for _ in range(3):
            try:
                if current.is_displayed() and current.is_enabled() and current.tag_name in ["button", "div", "a", "span"]:
                    return current
                current = current.find_element(By.XPATH, "..")
            except:
                break
        return element

    def is_submit_candidate(element, form):
        """Dynamically determine if an element is a submit button based on context and behavior."""
        elem_text = (element.text or "").lower()
        tag = element.tag_name
        if any(k in elem_text for k in ["submit", "send", "save", "confirm", "message"]) or tag in ["button", "input"]:
            try:
                parent_form = element.find_element(By.XPATH, "ancestor::form")
                if parent_form == form:
                    return True
                if element in form.find_elements(By.XPATH, "following-sibling::*") or element in form.find_elements(By.XPATH, "preceding-sibling::*"):
                    return True
            except:
                pass
        return False

    def detect_submission_change(driver):
        """Check if a submission occurred by looking for URL changes, form count changes, or success messages."""
        initial_url = driver.current_url
        initial_source = driver.page_source
        initial_form_count = len(driver.find_elements(By.TAG_NAME, "form"))
        time.sleep(2)
        new_url = driver.current_url
        new_source = driver.page_source
        new_form_count = len(driver.find_elements(By.TAG_NAME, "form"))
        success_indicators = ["thank you", "submitted", "success", "message sent", "your submission"]
        source_changed = new_source != initial_source and any(indicator in new_source.lower() for indicator in success_indicators)
        return new_url != initial_url or new_form_count != initial_form_count or source_changed

    summary = []

    # Initial wait for page load
    time.sleep(2)

    # Navigate to contact form if requested
    try:
        contact_links = WebDriverWait(driver, 5).until(
            EC.presence_of_all_elements_located((By.XPATH, "//*[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'contact')]"))
        )
        if contact_links:
            for link in contact_links:
                if link.is_displayed() and link.is_enabled():
                    driver.execute_script("arguments[0].scrollIntoView(true);", link)
                    driver.execute_script("arguments[0].click();", link)
                    summary.append("[main page] Clicked potential 'contact' link to access form.")
                    WebDriverWait(driver, 5).until(
                        EC.presence_of_element_located((By.TAG_NAME, "form"))
                    )
                    time.sleep(2)
                    break
    except Exception as e:
        summary.append(f"[main page] Error navigating to contact form: {str(e)}")

    # Check main context and iframes
    contexts = [(driver, "main page")]
    iframes = driver.find_elements(By.TAG_NAME, "iframe")
    for iframe in iframes:
        try:
            driver.switch_to.frame(iframe)
            iframe_id = iframe.get_attribute("id") or "unnamed"
            contexts.append((driver, f"iframe '{iframe_id}'"))
            driver.switch_to.default_content()
        except Exception as e:
            summary.append(f"Error accessing iframe: {str(e)}")

    form_count = 0
    submitted_forms = 0

    for context_driver, context_name in contexts:
        if context_driver != driver:
            try:
                driver.switch_to.frame(context_driver)
            except:
                continue

        # Check for success page immediately to avoid redundant processing
        if "thank you" in driver.page_source.lower() or "your submission" in driver.page_source.lower():
            summary.append(f"[{context_name}] Detected 'Thank you' page; submission already successful.")
            submitted_forms += 1
            break

        forms = driver.find_elements(By.TAG_NAME, "form")
        if not forms:
            forms = [driver.find_element(By.TAG_NAME, "body")]

        for form in forms:
            form_count += 1

            # Fill all inputs with LLM-extracted or random data
            all_inputs = form.find_elements(By.XPATH, ".//input | .//textarea | .//select")
            for inp in all_inputs:
                try:
                    itype = (inp.get_attribute("type") or "").lower()
                    if itype == "hidden":
                        continue
                    if not inp.is_enabled():
                        continue
                    if not inp.is_displayed():
                        driver.execute_script("arguments[0].style.display = 'block';", inp)
                        summary.append(f"[{context_name}] Forced visibility of {itype} input.")

                    if itype == "checkbox":
                        if not inp.is_selected():
                            inp.click()
                        summary.append(f"[{context_name}] Checked a checkbox.")
                    elif itype == "radio":
                        radio_name = inp.get_attribute("name")
                        if radio_name and radio_name not in locals().get('visited_radio_groups', set()):
                            if not inp.is_selected():
                                inp.click()
                            locals().setdefault('visited_radio_groups', set()).add(radio_name)
                            summary.append(f"[{context_name}] Selected radio button '{radio_name}'.")
                    elif itype in ["button", "submit", "reset", "file"]:
                        continue
                    else:
                        value = guess_input_value(inp, arg)
                        inp.clear()
                        inp.send_keys(value)
                        summary.append(f"[{context_name}] Filled input ({itype}) with '{value}'.")
                except Exception as e:
                    summary.append(f"[{context_name}] Error filling input ({itype}): {str(e)}")

            # Delay before submission
            time.sleep(2)

            # Dynamically detect and click the submit button
            submitted = False
            potential_buttons = []
            try:
                candidates = form.find_elements(By.XPATH, ".//*[self::button or self::input or self::div or self::span or self::a]")
                for candidate in candidates:
                    if candidate.is_displayed() and candidate.is_enabled() and is_submit_candidate(candidate, form):
                        potential_buttons.append(candidate)

                if not potential_buttons:
                    nearby_candidates = driver.find_elements(By.XPATH, "//*[self::button or self::input or self::div or self::span or self::a]")
                    for candidate in nearby_candidates:
                        if candidate.is_displayed() and candidate.is_enabled() and is_submit_candidate(candidate, form):
                            potential_buttons.append(candidate)

                for btn in potential_buttons:
                    try:
                        clickable = find_parent_clickable(btn)
                        driver.execute_script("arguments[0].scrollIntoView(true);", clickable)
                        driver.execute_script("arguments[0].click();", clickable)
                        summary.append(f"[{context_name}] Attempted submission by clicking candidate: '{btn.text}' (tag: {clickable.tag_name})")
                        if detect_submission_change(driver):
                            summary.append(f"[{context_name}] Submission detected after clicking '{btn.text}'.")
                            submitted = True
                            submitted_forms += 1
                            break
                        time.sleep(2)
                    except Exception as e:
                        summary.append(f"[{context_name}] Error clicking candidate '{btn.text}': {str(e)}")
            except Exception as e:
                summary.append(f"[{context_name}] Error detecting submit button: {str(e)}")

            # Fallback: Enter key
            if not submitted:
                text_inputs = form.find_elements(By.CSS_SELECTOR, "input[type='text'], input[type='email'], textarea")
                if text_inputs:
                    try:
                        text_inputs[0].send_keys(Keys.ENTER)
                        summary.append(f"[{context_name}] Submitted form by sending Enter key.")
                        if detect_submission_change(driver):
                            submitted = True
                            submitted_forms += 1
                        time.sleep(3)
                    except Exception as e:
                        summary.append(f"[{context_name}] Error sending Enter: {str(e)}")

            # Fallback: JavaScript submission with retry
            if not submitted:
                for _ in range(2):
                    try:
                        WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.TAG_NAME, "form")))
                        form = driver.find_elements(By.TAG_NAME, "form")[forms.index(form)]
                        driver.execute_script("""
                            var form = arguments[0];
                            if (form && typeof form.submit === 'function') {
                                var event = new Event('submit', { bubbles: true, cancelable: true });
                                form.dispatchEvent(event);
                                form.submit();
                            } else if (form && form.requestSubmit) {
                                form.requestSubmit();
                            }
                        """, form)
                        summary.append(f"[{context_name}] Submitted form using enhanced JavaScript.")
                        if detect_submission_change(driver):
                            submitted = True
                            submitted_forms += 1
                        time.sleep(2)
                        break
                    except Exception as e:
                        summary.append(f"[{context_name}] JavaScript submission attempt {_+1} failed: {str(e)}")
                        time.sleep(1)

            # Check for success page and stop if found
            if submitted and ("thank you" in driver.page_source.lower() or "your submission" in driver.page_source.lower()):
                summary.append(f"[{context_name}] Confirmed 'Thank you' page after submission.")
                break

            # Last resort: Click any nearby button-like element
            if not submitted:
                try:
                    nearby = driver.find_elements(By.XPATH, "//*[self::button or self::input or self::div or self::span or self::a]")
                    for elem in nearby:
                        if elem.is_displayed() and elem.is_enabled():
                            clickable = find_parent_clickable(elem)
                            driver.execute_script("arguments[0].scrollIntoView(true);", clickable)
                            driver.execute_script("arguments[0].click();", clickable)
                            summary.append(f"[{context_name}] Last resort click on '{elem.text}' (tag: {clickable.tag_name})")
                            if detect_submission_change(driver):
                                submitted = True
                                submitted_forms += 1
                                break
                            time.sleep(3)
                except Exception as e:
                    summary.append(f"[{context_name}] Last resort click failed: {str(e)}")

            if submitted:
                break

        driver.switch_to.default_content()
        if submitted_forms > 0:
            break

    return f"Detected {form_count} form(s) across contexts, submitted {submitted_forms}:\n" + "\n".join(summary)


# --------------------------------------------------------------------------------
# Create the Agent
# --------------------------------------------------------------------------------
def create_agent():
    chat = ChatOpenAI(temperature=0, model="gpt-4o")

    system_prompt = """
You are a helpful assistant with access to these tools:
1) internet_searcher(query: str)
2) web_scraper(url: str)
3) pdf_scraper(url: str)
4) go_to_url_tool(url: str)
5) get_page_source_tool()
6) shutdown_browser_tool()
7) fill_every_form_tool()

Instructions:
- Use 'go_to_url_tool' to navigate to the specified website (e.g., tarifflo.com).
- Use 'fill_every_form_tool' to fill all forms automatically. Extract field-value pairs from the user input (e.g., 'fill out the form with my name as John Doe and email as john@example.com') and pass them as a dictionary via the 'arg' parameter. If no specific data is provided, pass an empty dictionary {} to use random realistic data.
- Dynamically identify the submit button based on context and submit relentlessly until successful or all options are exhausted.
- For requests like "fill out a contact form on [website]", first use go_to_url_tool, then look for a "contact" link or similar to reach the form, and apply fill_every_form_tool.
- Include the full detailed summary from fill_every_form_tool in your response to show exactly what was attempted and the outcome, including any errors.
- If the summary indicates a submission attempt succeeded (e.g., a button was clicked, JavaScript executed, or a 'Thank you' page was detected) and no critical errors prevent submission, report it as a success.
- If successful, summarize the filled fields and the method used to submit, noting that external confirmation (e.g., email) might be needed to verify.
- Stop processing additional forms or actions once a 'Thank you' or similar success page is detected.
- If data is extracted from the input, prioritize it over random data for matching fields based on name, id, or placeholder attributes.
"""

    tools = [
        internet_searcher,
        web_scraper,
        pdf_scraper,
        go_to_url_tool,
        get_page_source_tool,
        shutdown_browser_tool,
        fill_every_form_tool
    ]

    return create_react_agent(
        model=chat,
        tools=tools,
        prompt=system_prompt,
        checkpointer=MemorySaver(),
    )

# --------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    agent = create_agent()
    try:
        user_input = input("Enter a message: ")
        config = {
            "recursion_limit": 50,
            "thread_id": 42
        }
        result = agent.invoke({"messages": [user_input]}, config=config)
        print(result["messages"][-1].content)
    finally:
        browser.quit()