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
        self.wait_time = 10  # increased wait time for better page loading

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
    Particularly effective for login forms if an email is provided.
    Args:
        arg (Dict[str, str], optional): Dictionary of field names and values extracted by the LLM.
                                        For login forms, it's best to include "email" key.
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
            # Look for exact matches first, then partial matches
            field_keys = [key.lower() for key in custom_data.keys()]
            
            # For email fields, prioritize email values regardless of field name
            if "email" in name_id_placeholder or input_type == "email":
                # Look for email value in custom_data
                email_value = None
                for key, value in custom_data.items():
                    if isinstance(value, str) and "@" in value and "." in value:  # Basic email format check
                        email_value = value
                        break
                
                # If we found an email, return it
                if email_value:
                    return email_value
            
            # Special case for customs site: if there's only one input on the page and we have an email
            try:
                driver = browser.driver
                visible_inputs = [inp for inp in driver.find_elements(By.TAG_NAME, "input") 
                                 if inp.is_displayed() and inp.get_attribute("type") not in ["hidden", "submit", "button"]]
                
                if len(visible_inputs) <= 2:  # If there are only a few input fields visible (like in a login form)
                    for key, value in custom_data.items():
                        if isinstance(value, str) and "@" in value and "." in value:  # If we have an email in the data
                            return value
            except:
                pass
            
            # Check for exact match in field names
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
        try:
            initial_url = driver.current_url
            initial_source = driver.page_source
            initial_form_count = len(driver.find_elements(By.TAG_NAME, "form"))
            
            # For login detection, check if there are any visible email/username fields before
            initial_login_fields = len(driver.find_elements(By.XPATH, 
                "//input[contains(@type, 'email') or contains(@name, 'email') or contains(@id, 'email') or contains(@id, 'username') or contains(@name, 'username')]"))
            
            time.sleep(3)  # Increased wait time
            
            new_url = driver.current_url
            new_source = driver.page_source
            new_form_count = len(driver.find_elements(By.TAG_NAME, "form"))
            
            # For login detection, check if visible email/username fields disappeared
            new_login_fields = len(driver.find_elements(By.XPATH, 
                "//input[contains(@type, 'email') or contains(@name, 'email') or contains(@id, 'email') or contains(@id, 'username') or contains(@name, 'username')]"))
            
            success_indicators = ["thank you", "submitted", "success", "message sent", "your submission", "welcome", "dashboard", "account", "profile"]
            source_changed = new_source != initial_source and any(indicator in new_source.lower() for indicator in success_indicators)
            login_success = initial_login_fields > 0 and new_login_fields < initial_login_fields
            
            result = new_url != initial_url or new_form_count != initial_form_count or source_changed or login_success
            
            if result:
                change_reasons = []
                if new_url != initial_url:
                    change_reasons.append(f"URL changed from {initial_url} to {new_url}")
                if new_form_count != initial_form_count:
                    change_reasons.append(f"Form count changed from {initial_form_count} to {new_form_count}")
                if source_changed:
                    change_reasons.append("Success indicator found in page source")
                if login_success:
                    change_reasons.append(f"Login fields reduced from {initial_login_fields} to {new_login_fields}")
                
                summary.append(f"Submission change detected: {', '.join(change_reasons)}")
            
            return result
        except Exception as e:
            summary.append(f"Error in detect_submission_change: {str(e)}")
            return False

    summary = []

    # Initial wait for page load
    time.sleep(2)

    # Navigate to contact/login form if requested
    try:
        # Look for login links or forms first
        login_xpath = "//*[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'login') or contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'sign in')]"
        login_links = WebDriverWait(driver, 5).until(
            EC.presence_of_all_elements_located((By.XPATH, login_xpath))
        )
        
        if login_links:
            for link in login_links:
                if link.is_displayed() and link.is_enabled():
                    driver.execute_script("arguments[0].scrollIntoView(true);", link)
                    driver.execute_script("arguments[0].click();", link)
                    summary.append("[main page] Clicked potential 'login/sign in' link to access form.")
                    # Wait for form elements - look specifically for email or username inputs
                    WebDriverWait(driver, browser.wait_time).until(
                        EC.presence_of_element_located((By.XPATH, "//input[contains(@type, 'email') or contains(@name, 'email') or contains(@id, 'email') or contains(@id, 'username') or contains(@name, 'username')]"))
                    )
                    time.sleep(2)
                    break
        else:
            # Fallback to looking for contact links
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
        summary.append(f"[main page] Error navigating to form: {str(e)}")

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

            # Prioritize email fields first if we have an email in arg
            email_value = None
            if arg:
                for key, value in arg.items():
                    if "@" in value and "." in value:  # Basic email format check
                        email_value = value
                        break
            
            # First pass: identify all email fields using multiple strategies
            email_fields = []
            all_inputs = form.find_elements(By.XPATH, ".//input | .//textarea | .//select")
            
            # Try to find fields that are specifically for email
            for inp in all_inputs:
                try:
                    itype = (inp.get_attribute("type") or "").lower()
                    name = (inp.get_attribute("name") or "").lower()
                    id_attr = (inp.get_attribute("id") or "").lower()
                    
                    # Check multiple attributes for "email" or related terms
                    if (itype == "email" or 
                        "email" in name or 
                        "email" in id_attr or
                        "user" in name or
                        "user" in id_attr):
                        email_fields.append(inp)
                except:
                    pass
            
            # Special case for customsinfo.com: check if there's a field after an "Email:" label
            try:
                # For sites that use table layouts with label in one cell and input in another
                email_label_fields = driver.find_elements(By.XPATH, "//td[contains(text(), 'Email:')]/following-sibling::td/input")
                email_fields.extend(email_label_fields)
                
                # For sites that use label elements
                labeled_fields = driver.find_elements(By.XPATH, "//label[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'email')]/following-sibling::input")
                email_fields.extend(labeled_fields)
                
                # For sites that put the label text right before the input
                text_nodes = driver.find_elements(By.XPATH, "//*[contains(text(), 'Email')]")
                for node in text_nodes:
                    # Find nearby input elements
                    nearby_inputs = node.find_elements(By.XPATH, "./following::input[position() < 3]")
                    email_fields.extend(nearby_inputs)
            except:
                pass
            
            # If we have both email value and fields, fill them first
            if email_value and email_fields:
                for email_field in email_fields:
                    try:
                        if email_field.is_displayed() and email_field.is_enabled():
                            driver.execute_script("arguments[0].scrollIntoView(true);", email_field)
                            email_field.clear()
                            email_field.send_keys(email_value)
                            time.sleep(0.5)  # Short pause after filling email
                            summary.append(f"[{context_name}] Filled email field with '{email_value}'.")
                    except Exception as e:
                        summary.append(f"[{context_name}] Error filling email field: {str(e)}")
            
            # Now fill all inputs with LLM-extracted or random data
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
                    
                    # Skip email fields we already filled
                    if email_value and inp in email_fields:
                        continue

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
- IMPORTANT: When you see an email address anywhere in the user's input, always extract it as a key-value pair with "email" as the key. For example, if the user says "use the email a02324348@usu.edu to sign in", extract {"email": "a02324348@usu.edu"}.
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
    try:
        user_input = input("Enter a message: ")
        print(f"Processing request: {user_input}")
        
        # Extract email directly for testing
        import re
        email_match = re.search(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', user_input)
        email = email_match.group(0) if email_match else None
        print(f"Extracted email: {email}")
        
        # For testing, navigate directly to the customs site
        if "export.customsinfo.com" in user_input:
            print("Navigating to Customs Info site...")
            browser.go_to_url("https://export.customsinfo.com/Default.aspx")
            time.sleep(5)
            
            # Look for login link
            try:
                driver = browser.driver
                login_xpath = "//*[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'login') or contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'sign in')]"
                login_links = driver.find_elements(By.XPATH, login_xpath)
                print(f"Found {len(login_links)} login links")
                
                if login_links:
                    for link in login_links:
                        if link.is_displayed() and link.is_enabled():
                            print(f"Clicking login link: {link.text}")
                            driver.execute_script("arguments[0].scrollIntoView(true);", link)
                            driver.execute_script("arguments[0].click();", link)
                            time.sleep(5)
                            break
                
                # Check for email input fields - we now know customsinfo.com uses an Email: label
                # Try multiple approaches to finding the email field
                
                # 1. Look for fields with email in attributes
                email_fields = driver.find_elements(By.XPATH, "//input[contains(@type, 'email') or contains(@name, 'email') or contains(@id, 'email')]")
                
                # 2. Look for username fields
                username_fields = driver.find_elements(By.XPATH, "//input[contains(@id, 'username') or contains(@name, 'username')]")
                
                # 3. Look for fields preceded by an "Email:" label (specific to customsinfo.com)
                email_label_fields = driver.find_elements(By.XPATH, "//td[contains(text(), 'Email:')]/following-sibling::td/input")
                
                # 4. Look for any text input field as a last resort
                text_fields = driver.find_elements(By.XPATH, "//input[@type='text']")
                
                print(f"Found {len(email_fields)} email fields, {len(username_fields)} username fields, {len(email_label_fields)} email-label fields, and {len(text_fields)} text fields")
                
                # Choose the most likely field to be the email input
                target_field = None
                if email_fields:
                    target_field = email_fields[0]
                    print("Using email field")
                elif email_label_fields:
                    target_field = email_label_fields[0]
                    print("Using field with Email: label")
                elif username_fields:
                    target_field = username_fields[0]
                    print("Using username field")
                elif text_fields:
                    # As a last resort, use the first visible text field
                    for field in text_fields:
                        if field.is_displayed():
                            target_field = field
                            print("Using generic text field")
                            break
                
                # Fill in the email field if found
                if email and target_field:
                    print(f"Filling field: {target_field.get_attribute('name') or target_field.get_attribute('id')} with {email}")
                    driver.execute_script("arguments[0].scrollIntoView(true);", target_field)
                    target_field.clear()
                    target_field.send_keys(email)
                    time.sleep(1)
                    
                    # Try to submit the form - look for various submit mechanisms
                    
                    # 1. Look for Login/Sign in buttons
                    submit_buttons = driver.find_elements(By.XPATH, "//button[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'sign in') or contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'login') or contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'submit')]")
                    
                    # 2. Look for input elements with type="submit" or with "login" value (common pattern)
                    # This specific XPath handles the customsinfo.com case where there's an input with value="Login"
                    submit_inputs = driver.find_elements(By.XPATH, "//input[@type='submit' or contains(translate(@value, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'login')]")
                    
                    # Special case for customsinfo.com
                    if not submit_inputs and "customsinfo.com" in driver.current_url:
                        submit_inputs = driver.find_elements(By.XPATH, "//input[@value='Login']")
                    
                    # 3. For customsinfo.com specifically, look for any element with "Login" text
                    login_elements = driver.find_elements(By.XPATH, "//*[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'login')]")
                    
                    print(f"Found {len(submit_buttons)} submit buttons, {len(submit_inputs)} submit inputs, and {len(login_elements)} login elements")
                    
                    # Try clicking the elements in order of likelihood
                    if submit_inputs:
                        print(f"Clicking submit input: {submit_inputs[0].get_attribute('value')}")
                        driver.execute_script("arguments[0].click();", submit_inputs[0])
                    elif submit_buttons:
                        print(f"Clicking submit button: {submit_buttons[0].text}")
                        driver.execute_script("arguments[0].click();", submit_buttons[0])
                    elif login_elements:
                        # Try to find the most likely login element (one that's clickable and visible)
                        for login_elem in login_elements:
                            if login_elem.is_displayed() and login_elem.is_enabled() and login_elem.text.strip():
                                print(f"Clicking login element: {login_elem.text}")
                                driver.execute_script("arguments[0].click();", login_elem)
                                break
                    else:
                        print("No submit button found, trying Enter key")
                        target_field.send_keys(Keys.ENTER)
                    
                    time.sleep(5)
                    print(f"Current URL after submission: {driver.current_url}")
            except Exception as e:
                print(f"Error during login: {str(e)}")
            
            # Let's just wait to examine the state
            time.sleep(5)
        else:
            # Regular agent execution
            agent = create_agent()
            config = {
                "recursion_limit": 50,
                "thread_id": 42
            }
            result = agent.invoke({"messages": [user_input]}, config=config)
            print(result["messages"][-1].content)
    except Exception as e:
        print(f"Error in main: {str(e)}")
    finally:
        print("Shutting down browser...")
        browser.quit()
        print("Done.")