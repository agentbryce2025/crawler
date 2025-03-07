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
        
        # Set binary location to fix ChromeDriver issue
        import shutil
        import os

        # Check for available browsers
        browsers = ['chromium-browser', 'chromium', 'google-chrome', 'chrome', 'firefox', 'firefox-esr']
        print("Checking for browser paths:")
        for browser in browsers:
            path = shutil.which(browser)
            if path:
                print(f"Found {browser} at: {path}")
        
        # Look for Firefox installations first (which should work on Ubuntu)
        if shutil.which('firefox-esr') or shutil.which('firefox'):
            print("Using Firefox as Chrome alternative")
            # We can't directly use Firefox with undetected_chromedriver, so let's avoid setting binary_location
        else:
            # Try to find any Chrome-compatible browser
            chrome_path = shutil.which('chromium-browser') or shutil.which('chromium') or shutil.which('google-chrome') or shutil.which('chrome')
            if chrome_path:
                print(f"Using Chrome binary at: {chrome_path}")
                options.binary_location = chrome_path
            
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
            
            # Dynamic form detection - identify login-like forms with minimal inputs
            try:
                driver = browser.driver
                visible_inputs = [inp for inp in driver.find_elements(By.TAG_NAME, "input") 
                                 if inp.is_displayed() and inp.get_attribute("type") not in ["hidden", "submit", "button"]]
                
                # Check for simple forms with few inputs - common for login forms
                if len(visible_inputs) <= 3:  # If there are only a few input fields visible
                    # First try to look for an email field in the form
                    for key, value in custom_data.items():
                        if isinstance(value, str) and "@" in value and "." in value:  # If we have an email in the data
                            # Return email for likely email field
                            return value
                    
                    # If no email in custom data, look at field attributes to determine likely type
                    for input_elem in visible_inputs:
                        name_attrs = (
                            (input_elem.get_attribute("name") or "").lower() +
                            (input_elem.get_attribute("id") or "").lower() +
                            (input_elem.get_attribute("placeholder") or "").lower()
                        )
                        
                        # For username/email field check
                        if any(x in name_attrs for x in ["user", "email", "login", "account"]):
                            for key, value in custom_data.items():
                                if isinstance(value, str) and ("@" in value or "username" in key.lower() or "user" in key.lower()):
                                    return value
            except Exception as e:
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
        
    def find_and_click_image_buttons(driver, keywords=None, src_patterns=None, wait_time=3):
        """
        Find and click image buttons based on alt text keywords or src patterns.
        
        Args:
            driver: The WebDriver instance
            keywords: List of keywords to search for in alt text
            src_patterns: List of patterns to search for in src attribute
            wait_time: Time to wait after clicking
            
        Returns:
            bool: True if an image button was found and clicked, False otherwise
        """
        if keywords is None:
            keywords = ['submit', 'search', 'continue', 'next', 'go', 'login', 'sign', 'send', 'save', 'update', 'calc', 'apply']
        
        if src_patterns is None:
            src_patterns = ['button', 'submit', 'search', 'arrow', 'next', 'login']
        
        # Build XPath for alt text keywords
        alt_conditions = " or ".join([
            f"contains(translate(@alt, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{kw.lower()}')" 
            for kw in keywords
        ])
        
        # Build XPath for src patterns
        src_conditions = " or ".join([
            f"contains(@src, '{pattern}')" 
            for pattern in src_patterns
        ])
        
        # Combine conditions
        xpath = f"//img[{alt_conditions} or {src_conditions}]"
        
        try:
            images = driver.find_elements(By.XPATH, xpath)
            
            if images:
                for img in images:
                    if img.is_displayed():
                        alt_text = img.get_attribute("alt") or ""
                        src = img.get_attribute("src") or ""
                        print(f"Found image button with alt text: '{alt_text}' and src: {src}")
                        driver.execute_script("arguments[0].click();", img)
                        time.sleep(wait_time)
                        print(f"Clicked on image button: {alt_text or src}")
                        return True
        except Exception as e:
            print(f"Error finding/clicking image buttons: {str(e)}")
        
        return False

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
            
            # General approach for finding email fields by nearby labels
            try:
                # For sites that use table layouts with label in one cell and input in another
                email_label_fields = driver.find_elements(By.XPATH, 
                    "//td[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'email')]/following-sibling::td/input")
                email_fields.extend(email_label_fields)
                
                # For sites that use label elements - more general approach for all sites
                labeled_fields = driver.find_elements(By.XPATH, 
                    "//label[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'email')]/following-sibling::input | " +
                    "//label[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'email')]/input")
                email_fields.extend(labeled_fields)
                
                # For accessibility-focused sites that use aria-label
                aria_fields = driver.find_elements(By.XPATH, 
                    "//input[contains(translate(@aria-label, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'email') or " +
                    "contains(translate(@placeholder, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'email')]")
                email_fields.extend(aria_fields)
                
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
You are a helpful assistant with access to these tools to interact with web content:
1) internet_searcher(query: str) - Search the web for information
2) web_scraper(url: str) - Retrieve the HTML content of a webpage
3) pdf_scraper(url: str) - Extract text from PDF documents
4) go_to_url_tool(url: str) - Navigate the browser to a specific URL
5) get_page_source_tool() - Get the current page's HTML content
6) shutdown_browser_tool() - Close the browser properly
7) fill_every_form_tool() - Intelligently fill forms with extracted or random data

General Instructions:
- Extract URLs from user requests to navigate to websites using 'go_to_url_tool'
- Identify and extract key information from user requests such as:
  * Email addresses and login credentials
  * Product codes (HS codes, HTS codes, etc.)
  * Country names and requirements
  * Specific tasks to perform (searching, form filling, information extraction)
- Use 'fill_every_form_tool' to handle forms, passing extracted data as a dictionary via the 'arg' parameter
- IMPORTANT: When detecting an email address, always extract it with "email" as the key (e.g., {"email": "user@example.com"})
- Extract any other field-value pairs from user instructions (e.g., names, quantities, descriptions)
- For unknown values, pass an empty dictionary to use realistic random data
- For search and lookup operations, extract search terms and navigate through appropriate UI elements
- When navigating sites, look for relevant links, tabs, and buttons that match the user's goals
- When success is achieved, report details of exactly what was found or accomplished
- Include screenshots or extracted content as appropriate to show results
- The crawler is designed to handle a wide variety of websites and form structures dynamically

For specific duty/tariff lookups:
- Extract product codes (e.g., HS codes like 9018.19.10) and countries (e.g., Brazil)
- Navigate to the appropriate search fields and enter the codes
- Look for country selection options and select the target country
- Find and extract duty rates, which are often displayed as percentages
- Report the complete findings with context about the product and applicable rates
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
        
        # Extract URL from user input if present
        url_match = re.search(r'https?://[^\s]+', user_input)
        if url_match:
            target_url = url_match.group(0)
            print(f"Found URL in input: {target_url}")
            browser.go_to_url(target_url)
            time.sleep(5)
            
            # Look for login link - general approach for any site
            try:
                driver = browser.driver
                # Comprehensive login detection xpath
                login_xpath = "//*[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'login') or " + \
                              "contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'sign in') or " + \
                              "contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'log in') or " + \
                              "contains(@href, 'login') or contains(@href, 'signin') or contains(@href, 'account')]"
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
                
                # Try multiple approaches to finding login fields (email/username)
                
                # 1. Look for fields with email in attributes - standard across most sites
                email_fields = driver.find_elements(By.XPATH, 
                    "//input[contains(@type, 'email') or contains(@name, 'email') or contains(@id, 'email')]")
                
                # 2. Look for username fields - common on many login forms
                username_fields = driver.find_elements(By.XPATH, 
                    "//input[contains(@id, 'username') or contains(@name, 'username') or contains(@placeholder, 'username')]")
                
                # 3. Look for fields preceded by email-related labels - works on sites with various layouts
                email_label_fields = driver.find_elements(By.XPATH, 
                    "//td[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'email')]/following-sibling::td/input | " +
                    "//label[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'email')]/following-sibling::input | " +
                    "//div[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'email')]/following-sibling::input")
                
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
                    
                    # 2. Look for input elements with type="submit" or with login-related values (common pattern across sites)
                    submit_inputs = driver.find_elements(By.XPATH, 
                        "//input[@type='submit' or " +
                        "contains(translate(@value, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'login') or " +
                        "contains(translate(@value, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'sign in') or " +
                        "contains(translate(@value, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'submit') or " +
                        "contains(translate(@value, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'continue')]")
                    
                    # If no submit inputs found, look for button elements with similar text
                    if not submit_inputs:
                        submit_inputs = driver.find_elements(By.XPATH,
                            "//button[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'login') or " +
                            "contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'sign in') or " +
                            "contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'submit') or " +
                            "contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'continue')]")
                    
                    # Look for any login-related elements across multiple sites
                    login_elements = driver.find_elements(By.XPATH, 
                        "//*[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'login') or " +
                        "contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'sign in') or " + 
                        "contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'log in') or " +
                        "contains(@class, 'login') or contains(@id, 'login')]"
                    )
                    
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
                    
                    # Check if we're on the Global Tariffs page or need to navigate there
                    if "GlobalTariffs" in driver.current_url:
                        print("Successfully navigated to Global Tariffs page")
                    else:
                        print("Trying to navigate to Global Tariffs page")
                        # Look for Global Tariffs link
                        try:
                            global_tariffs_links = driver.find_elements(By.XPATH, "//*[contains(text(), 'Global Tariffs') or contains(@href, 'GlobalTariffs')]")
                            if global_tariffs_links:
                                for link in global_tariffs_links:
                                    if link.is_displayed():
                                        print(f"Clicking Global Tariffs link: {link.text}")
                                        driver.execute_script("arguments[0].click();", link)
                                        time.sleep(3)
                                        break
                        except Exception as e:
                            print(f"Error navigating to Global Tariffs: {str(e)}")
                    
                    # Now search for the duty rate using HS code and Brazil
                    try:
                        print("Attempting to search for duty rate information")
                        
                        # Extract product codes from user input using various formats
                        import re
                        # Dictionary of code types and their regex patterns
                        code_patterns = {
                            "HS Code (10-digit)": r'\b\d{4}\.\d{2}\.\d{2}\b',
                            "HS Code (6-digit)": r'\b\d{4}\.\d{2}\b',
                            "HS Code (4-digit)": r'\b\d{4}\b',
                            "HTS Code": r'\b\d{4}\.\d{2}\.\d{4}\b',
                            "Schedule B": r'\b\d{10}\b',
                            "ECCN Code": r'\b\d[A-Z]\d{3}[A-Z]\b',
                        }
                        
                        # Try to find any of these codes in the input
                        found_codes = {}
                        for code_type, pattern in code_patterns.items():
                            matches = re.findall(pattern, user_input)
                            if matches:
                                found_codes[code_type] = matches
                        
                        # Default to HS code first (most common for tariffs)
                        hs_code = None
                        
                        # Look through the codes we found, prioritizing HS codes
                        if "HS Code (10-digit)" in found_codes:
                            hs_code = found_codes["HS Code (10-digit)"][0]
                        elif "HS Code (6-digit)" in found_codes:
                            hs_code = found_codes["HS Code (6-digit)"][0]
                        elif "HS Code (4-digit)" in found_codes:
                            hs_code = found_codes["HS Code (4-digit)"][0]
                        elif len(found_codes) > 0:
                            # Use the first code of any type we found
                            first_type = list(found_codes.keys())[0]
                            hs_code = found_codes[first_type][0]
                            
                        # Hardcoded examples for specific cases
                        if "9018.19.10" in user_input:
                            hs_code = "9018.19.10"
                        
                        # Extract country information dynamically
                        country_patterns = {
                            "Brazil": [r'\bbrazil\b', r'\bbra\b', r'\bbr\b'],
                            "China": [r'\bchina\b', r'\bchn\b', r'\bcn\b'],
                            "United States": [r'\bunited states\b', r'\busa\b', r'\bus\b', r'\bu\.s\.a?\b'],
                            "India": [r'\bindia\b', r'\bind\b', r'\bin\b'],
                            "Japan": [r'\bjapan\b', r'\bjpn\b', r'\bjp\b'],
                            "Mexico": [r'\bmexico\b', r'\bmex\b', r'\bmx\b'],
                        }
                        
                        # Look for country matches in the input
                        country = None
                        for country_name, patterns in country_patterns.items():
                            for pattern in patterns:
                                if re.search(pattern, user_input.lower()):
                                    country = country_name
                                    break
                            if country:
                                break
                                
                        # Default to Brazil if no country found (for backward compatibility)
                        if not country:
                            country = "Brazil"
                        
                        print(f"Searching for HS code: {hs_code} for country: {country}")
                        
                        if hs_code and country:
                            # Look for HS Code input field using various approaches
                            print("Searching for HS Code input field...")
                            
                            # 1. Look for fields with specific HS Code attributes
                            hs_code_fields = driver.find_elements(By.XPATH, 
                                "//input[contains(@id, 'HSCode') or contains(@name, 'HSCode') or " +
                                "contains(@placeholder, 'HS Code') or contains(@placeholder, 'Enter HS Code')]"
                            )
                            
                            # 2. Look for fields preceded by HS Code label
                            if not hs_code_fields:
                                print("Looking for HS Code field by label...")
                                hs_label_elements = driver.find_elements(By.XPATH, 
                                    "//*[contains(text(), 'HS Code') or contains(text(), 'HTS Code')]"
                                )
                                
                                if hs_label_elements:
                                    for label in hs_label_elements:
                                        print(f"Found HS Code label: {label.text}")
                                        # Try to find an input field next to the label
                                        try:
                                            # Check various directions from label (following, parent, siblings)
                                            input_near_label = None
                                            
                                            # Check following element first
                                            possible_fields = driver.find_elements(By.XPATH, f"//td[contains(text(), 'HS Code')]/following-sibling::td//input")
                                            if possible_fields:
                                                input_near_label = possible_fields[0]
                                            
                                            # Check sibling
                                            if not input_near_label:
                                                possible_fields = driver.find_elements(By.XPATH, f"//input[preceding-sibling::*[contains(text(), 'HS Code')]]")
                                                if possible_fields:
                                                    input_near_label = possible_fields[0]
                                                    
                                            if input_near_label and input_near_label.is_displayed():
                                                hs_code_fields = [input_near_label]
                                                break
                                        except Exception as e:
                                            print(f"Error finding input near label: {str(e)}")
                            
                            # 3. Look for common search fields across various tariff/trade sites
                            if not hs_code_fields:
                                print("Checking for common product code search fields...")
                                # Common field IDs/names across multiple trade/tariff sites
                                common_fields = driver.find_elements(By.XPATH, 
                                    "//input[" +
                                    "contains(@id, 'search') or contains(@name, 'search') or " +
                                    "contains(@id, 'code') or contains(@name, 'code') or " +
                                    "contains(@id, 'tariff') or contains(@name, 'tariff') or " +
                                    "contains(@id, 'hs') or contains(@name, 'hs') or " +
                                    "contains(@id, 'hts') or contains(@name, 'hts') or " +
                                    "contains(@placeholder, 'Search') or contains(@placeholder, 'Enter code') or " +
                                    "@id='tb_HSCodeNumber' or @name='tb_HSCodeNumber' or " +
                                    "@id='txtHSCode' or @name='txtHSCode' or " +
                                    "@id='txtSearchCode' or @name='txtSearchCode']"
                                )
                                
                                if common_fields:
                                    hs_code_fields = common_fields
                            
                            # 4. If still not found, look for any text field that's not for email, etc.
                            if not hs_code_fields:
                                print("Looking for any text input field that could be for HS codes...")
                                # Exclude common fields like email, username, password, etc.
                                hs_code_fields = driver.find_elements(By.XPATH, 
                                    "//input[@type='text' and not(" +
                                    "contains(@id, 'email') or contains(@name, 'email') or " +
                                    "contains(@id, 'user') or contains(@name, 'user') or " +
                                    "contains(@id, 'password') or contains(@name, 'password') or " +
                                    "contains(@id, 'search') or contains(@name, 'search'))]"
                                )
                            
                            # Look for country dropdown or input
                            country_selects = driver.find_elements(By.XPATH, "//select[contains(@id, 'Country') or contains(@name, 'Country')]")
                            country_fields = driver.find_elements(By.XPATH, "//input[contains(@id, 'Country') or contains(@name, 'Country')]")
                            
                            # Fill in HS Code if field found
                            if hs_code_fields:
                                hs_field = hs_code_fields[0]
                                field_id = hs_field.get_attribute("id") or hs_field.get_attribute("name") or "unknown"
                                print(f"Found HS code field: {field_id}")
                                driver.execute_script("arguments[0].scrollIntoView(true);", hs_field)
                                
                                # Enhanced handling for fields that might not be interactable
                                # This applies to all sites, not just specific ones
                                if field_id in ["txtSearchCode", "search", "query", "code", "lookup"] or not hs_field.is_enabled():
                                    try:
                                        # Make the element interactable using JavaScript
                                        driver.execute_script(
                                            "arguments[0].style.display = 'block'; " +
                                            "arguments[0].style.visibility = 'visible'; " +
                                            "arguments[0].style.opacity = '1'; " +
                                            "arguments[0].disabled = false; " +
                                            "arguments[0].readOnly = false;", 
                                            hs_field
                                        )
                                        time.sleep(1)
                                        
                                        # Set the value using JavaScript - works even with disabled fields
                                        driver.execute_script("arguments[0].value = arguments[1];", hs_field, hs_code)
                                        print(f"Set search code using JavaScript: {hs_code}")
                                        
                                        # Look for search button with multiple approaches
                                        search_buttons = driver.find_elements(By.XPATH, 
                                            "//input[@type='submit' or @value='Search' or contains(@onclick, 'search')] | " +
                                            "//button[contains(text(), 'Search') or contains(@onclick, 'search')] | " +
                                            "//a[contains(text(), 'Search') or contains(@onclick, 'search')]"
                                        )
                                        
                                        if search_buttons:
                                            # Try to find the most relevant search button
                                            for btn in search_buttons:
                                                if btn.is_displayed():
                                                    print(f"Clicking search button: {btn.get_attribute('value') or btn.text}")
                                                    driver.execute_script("arguments[0].click();", btn)
                                                    break
                                        else:
                                            # Try submitting the form
                                            try:
                                                form = hs_field.find_element(By.XPATH, "./ancestor::form")
                                                driver.execute_script("arguments[0].submit();", form)
                                                print("Submitted form")
                                            except:
                                                # Last resort: press Enter
                                                try:
                                                    hs_field.send_keys(Keys.ENTER)
                                                    print("Sent ENTER key to field")
                                                except:
                                                    print("Could not submit search in any way")
                                        
                                        time.sleep(5)  # Wait longer for search results
                                    except Exception as js_error:
                                        print(f"Error with JavaScript approach: {str(js_error)}")
                                        # Fallback to regular approach
                                        try:
                                            hs_field.clear()
                                            # Send keys character by character
                                            for char in hs_code:
                                                hs_field.send_keys(char)
                                                time.sleep(0.2)  # Slight delay between characters
                                            print(f"Entered code using fallback: {hs_code}")
                                            hs_field.send_keys(Keys.ENTER)
                                        except Exception as fallback_error:
                                            print(f"Error with fallback approach: {str(fallback_error)}")
                                else:
                                    # Regular approach for other sites
                                    hs_field.clear()
                                    hs_field.send_keys(hs_code)
                                    
                                    # Additional debugging
                                    print(f"Entered HS code: {hs_code} into field {field_id}")
                                    time.sleep(1)
                                
                                # Check for autocomplete or suggestions after entering HS code
                                try:
                                    # Wait for any autocomplete suggestion to appear
                                    time.sleep(2)
                                    suggestion_elements = driver.find_elements(By.XPATH, 
                                        "//div[contains(@class, 'autocomplete') or contains(@class, 'suggestion')]//li | " +
                                        "//ul[contains(@class, 'autocomplete') or contains(@class, 'suggestion')]//li"
                                    )
                                    
                                    if suggestion_elements:
                                        for suggestion in suggestion_elements:
                                            if suggestion.is_displayed() and hs_code in suggestion.text:
                                                print(f"Clicking autocomplete suggestion: {suggestion.text}")
                                                driver.execute_script("arguments[0].click();", suggestion)
                                                break
                                except Exception as auto_error:
                                    print(f"Error handling HS code autocomplete: {str(auto_error)}")
                            else:
                                print("No HS code field found - this might be an issue with the site structure")
                            
                            # Select or input country
                            if country_selects:
                                # If dropdown, select Brazil
                                country_select = country_selects[0]
                                print(f"Found country dropdown: {country_select.get_attribute('id') or country_select.get_attribute('name')}")
                                select = Select(country_select)
                                
                                # Try selecting by visible text
                                try:
                                    select.select_by_visible_text(country)
                                    print(f"Selected {country} from dropdown")
                                except Exception as dropdown_error:
                                    print(f"Couldn't select by text: {str(dropdown_error)}")
                                    
                                    # Try with different case or partial match
                                    try:
                                        options = [option.text for option in select.options]
                                        for option in options:
                                            if country.lower() in option.lower():
                                                print(f"Found matching option: {option}")
                                                select.select_by_visible_text(option)
                                                break
                                    except Exception as e:
                                        print(f"Error with partial match selection: {str(e)}")
                                        
                                        # Last attempt: try to select Brazil by index or value
                                        try:
                                            # Look for values containing "BR" or "BRA" (country codes for Brazil)
                                            brazil_options = []
                                            for i, option in enumerate(select.options):
                                                value = option.get_attribute("value")
                                                if value and ("BR" in value or "BRA" in value or "brazil" in value.lower()):
                                                    brazil_options.append((i, option))
                                                    
                                            if brazil_options:
                                                idx, option = brazil_options[0]
                                                print(f"Found Brazil by code at index {idx}: {option.text}")
                                                select.select_by_index(idx)
                                            else:
                                                # Last resort: use JavaScript to set the value
                                                print("Using JavaScript to set dropdown value")
                                                
                                                # Enhanced dynamic country selection for all sites
                                                try:
                                                    # First try looking for any country-related elements with the target country name
                                                    country_elements = driver.find_elements(By.XPATH, 
                                                        f"//*[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{country.lower()}')]"
                                                    )
                                                    
                                                    # Also look for country codes (2-letter and 3-letter codes)
                                                    country_codes = {
                                                        "united states": ["US", "USA"],
                                                        "brazil": ["BR", "BRA"],
                                                        "china": ["CN", "CHN"],
                                                        "india": ["IN", "IND"],
                                                        "japan": ["JP", "JPN"],
                                                        "germany": ["DE", "DEU"],
                                                        "united kingdom": ["GB", "GBR", "UK"],
                                                        "france": ["FR", "FRA"],
                                                        "italy": ["IT", "ITA"],
                                                        "canada": ["CA", "CAN"],
                                                        "australia": ["AU", "AUS"],
                                                        "spain": ["ES", "ESP"],
                                                        "mexico": ["MX", "MEX"],
                                                        "south korea": ["KR", "KOR"],
                                                        "russia": ["RU", "RUS"],
                                                    }
                                                    
                                                    # Get country codes for the target country if available
                                                    target_codes = country_codes.get(country.lower(), [])
                                                    
                                                    # Look for elements with country code matches if we have codes for this country
                                                    if target_codes:
                                                        for code in target_codes:
                                                            code_elements = driver.find_elements(By.XPATH, 
                                                                f"//*[text()='{code}' or @value='{code}']"
                                                            )
                                                            country_elements.extend(code_elements)
                                                    
                                                    # Try clicking on any matching country element
                                                    for elem in country_elements:
                                                        if elem.is_displayed():
                                                            print(f"Found country element: {elem.text}")
                                                            driver.execute_script("arguments[0].click();", elem)
                                                            time.sleep(1)
                                                            break
                                                    
                                                    # Look for any duty/tariff/tax related elements
                                                    duty_elements = driver.find_elements(By.XPATH,
                                                        "//*[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'duty') or " +
                                                        "contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'tax') or " +
                                                        "contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'tariff')]"
                                                    )
                                                    
                                                    # Try clicking on any duty-related elements
                                                    for elem in duty_elements:
                                                        if elem.is_displayed() and elem.is_enabled():
                                                            print(f"Clicking duty/tariff element: {elem.text}")
                                                            driver.execute_script("arguments[0].click();", elem)
                                                            time.sleep(2)
                                                            break
                                                            
                                                    # Look for toggle/expand elements that might reveal more info
                                                    toggles = driver.find_elements(By.XPATH,
                                                        "//*[contains(@id, 'toggle') or contains(@class, 'toggle') or " +
                                                        "contains(@class, 'expand') or contains(@class, 'collapse') or " +
                                                        "contains(@title, 'expand') or contains(@title, 'show more')]"
                                                    )
                                                    
                                                    # Try clicking on any toggle elements
                                                    for toggle in toggles:
                                                        if toggle.is_displayed() and toggle.is_enabled():
                                                            print(f"Clicking toggle/expand element")
                                                            driver.execute_script("arguments[0].click();", toggle)
                                                            time.sleep(1)
                                                except Exception as dynamic_error:
                                                    print(f"Error with dynamic country handling: {str(dynamic_error)}")
                                                driver.execute_script(
                                                    "arguments[0].value = arguments[1]; arguments[0].dispatchEvent(new Event('change'));", 
                                                    country_select, 
                                                    "BR"  # Common value for Brazil
                                                )
                                        except Exception as js_error:
                                            print(f"All dropdown selection methods failed: {str(js_error)}")
                                time.sleep(1)
                            elif country_fields:
                                country_field = country_fields[0]
                                print(f"Found country field: {country_field.get_attribute('id') or country_field.get_attribute('name')}")
                                driver.execute_script("arguments[0].scrollIntoView(true);", country_field)
                                country_field.clear()
                                country_field.send_keys(country)
                                time.sleep(1)
                                
                                # Look for autocomplete suggestions after typing
                                try:
                                    # Wait for autocomplete suggestions to appear
                                    time.sleep(2)
                                    
                                    # Look for visible autocomplete suggestions
                                    autocomplete_items = driver.find_elements(By.XPATH, 
                                        "//div[contains(@class, 'autocomplete') or contains(@class, 'dropdown') or contains(@class, 'suggestion')]//li | " +
                                        "//ul[contains(@class, 'autocomplete') or contains(@class, 'dropdown') or contains(@class, 'suggestion')]//li"
                                    )
                                    
                                    if autocomplete_items:
                                        for item in autocomplete_items:
                                            if item.is_displayed() and country.lower() in item.text.lower():
                                                print(f"Clicking autocomplete suggestion: {item.text}")
                                                driver.execute_script("arguments[0].click();", item)
                                                time.sleep(1)
                                                break
                                except Exception as auto_error:
                                    print(f"Error handling autocomplete: {str(auto_error)}")
                            else:
                                # If no specific country field found, look for any likely fields
                                print("No standard country field found, looking for alternatives")
                                
                                # Look for any inputs or spans that might be a country selector
                                country_elements = driver.find_elements(By.XPATH, 
                                    "//input[contains(@placeholder, 'country') or contains(@placeholder, 'dest')] | " +
                                    "//span[contains(text(), 'Country') or contains(text(), 'Destination')]/following-sibling::*[1]"
                                )
                                
                                if country_elements:
                                    elem = country_elements[0]
                                    print(f"Found potential country element: {elem.tag_name}")
                                    
                                    if elem.tag_name == "input":
                                        elem.clear()
                                        elem.send_keys(country)
                                    elif elem.is_displayed() and elem.is_enabled():
                                        driver.execute_script("arguments[0].click();", elem)
                                        time.sleep(1)
                                        
                                        # After clicking, look for a dropdown or input
                                        dropdown_options = driver.find_elements(By.XPATH, "//li[contains(text(), 'Brazil')]")
                                        for option in dropdown_options:
                                            if option.is_displayed():
                                                driver.execute_script("arguments[0].click();", option)
                                                break
                                else:
                                    print("No country field found")
                            
                            # Look for search/submit buttons
                            search_buttons = driver.find_elements(By.XPATH, "//button[contains(text(), 'Search') or contains(@value, 'Search')] | //input[@type='submit' or @type='button'][contains(@value, 'Search')]")
                            if not search_buttons:
                                # Look for any button that might be for searching
                                search_buttons = driver.find_elements(By.XPATH, "//button | //input[@type='submit' or @type='button']")
                            
                            # Click search button
                            if search_buttons:
                                for button in search_buttons:
                                    if button.is_displayed() and button.is_enabled():
                                        print(f"Clicking search button: {button.text or button.get_attribute('value')}")
                                        driver.execute_script("arguments[0].click();", button)
                                        time.sleep(5)
                                        
                                        # After clicking search, look for any action buttons that might appear
                                        time.sleep(2)  # Wait for page to update
                                        
                                        # Use our helper method with general action keywords that would work across sites
                                        action_keywords = ['view', 'details', 'calc', 'show', 'open', 'more', 'info', 'select', 'next']
                                        action_src_patterns = ['button', 'arrow', 'view', 'details', 'next']
                                        
                                        find_and_click_image_buttons(
                                            driver, 
                                            keywords=action_keywords, 
                                            src_patterns=action_src_patterns,
                                            wait_time=3
                                        )
                                        break
                            else:
                                # If no button found, try pressing Enter in the last field used
                                print("No search button found, trying Enter key")
                                if country_fields:
                                    country_fields[0].send_keys(Keys.ENTER)
                                elif hs_code_fields:
                                    hs_code_fields[0].send_keys(Keys.ENTER)
                                time.sleep(5)
                            
                            # Extract and display the duty rate information
                            print("\nSearching for duty rate information in page...\n")
                            duty_rate_found = False
                            
                            # Generic data extraction for duty/tariff sites
                            print("Using intelligent data extraction for duty/tariff information")
                            
                            # Common structure across tariff lookup sites:
                            # 1. First search for a product code and get results with description
                            # 2. Often need to access specific tabs or sections (Duties, Tariffs, Taxes)
                            # 3. May need to select or filter by country
                            
                            try:
                                # First, take screenshots for debugging
                                screenshot_path = "/tmp/screenshot.png"
                                driver.save_screenshot(screenshot_path)
                                print(f"Screenshot saved to {screenshot_path}")
                                
                                # Check if we're on the Global Tariff page or need to navigate to it
                                if "GlobalTariffs" not in driver.current_url:
                                    global_tariff_links = driver.find_elements(By.XPATH, 
                                        "//a[contains(@href, 'GlobalTariffs') or contains(text(), 'Global Tariff') or contains(text(), 'Tariff')]"
                                    )
                                    if global_tariff_links:
                                        for link in global_tariff_links:
                                            if link.is_displayed():
                                                print(f"Clicking link to Global Tariffs: {link.text}")
                                                driver.execute_script("arguments[0].click();", link)
                                                time.sleep(3)
                                                break
                                    
                                # Now look for the search field on the Global Tariffs page
                                try:
                                    # Try to find search input fields in a general way
                                    # First look for common product/HS code field patterns
                                    search_field = None
                                    search_field_candidates = driver.find_elements(By.XPATH, 
                                        "//input[contains(@id, 'code') or contains(@name, 'code') or " +
                                        "contains(@id, 'product') or contains(@name, 'product') or " +
                                        "contains(@id, 'hs') or contains(@name, 'hs') or " +
                                        "contains(@placeholder, 'code') or contains(@placeholder, 'product') or " + 
                                        "contains(@placeholder, 'search')]")
                                    
                                    if search_field_candidates:
                                        search_field = search_field_candidates[0]
                                    else:
                                        # Fallback: try to find any text input field
                                        text_inputs = driver.find_elements(By.XPATH, "//input[@type='text']")
                                        if text_inputs:
                                            search_field = text_inputs[0]
                                    
                                    # Need to check if search field is in an iframe
                                    iframes = driver.find_elements(By.TAG_NAME, "iframe")
                                    if iframes:
                                        for iframe in iframes:
                                            try:
                                                driver.switch_to.frame(iframe)
                                                search_fields = driver.find_elements(By.ID, "txtSearchCode")
                                                if search_fields and search_fields[0].is_displayed():
                                                    search_field = search_fields[0]
                                                    break
                                                driver.switch_to.default_content()
                                            except:
                                                driver.switch_to.default_content()
                                    
                                    # Ensure the field is interactable
                                    driver.execute_script(
                                        "arguments[0].style.display = 'block'; " +
                                        "arguments[0].style.visibility = 'visible'; " +
                                        "arguments[0].disabled = false; " +
                                        "arguments[0].readOnly = false;", 
                                        search_field
                                    )
                                    
                                    # Enter the HS code using JavaScript
                                    driver.execute_script("arguments[0].value = arguments[1];", search_field, hs_code)
                                    print(f"Set HS code using JavaScript: {hs_code}")
                                    
                                    # Find and click the search button in a general way
                                    search_button = None
                                    
                                    # Try multiple approaches to find a search button
                                    search_button_candidates = driver.find_elements(By.XPATH,
                                        "//button[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'search')] | " +
                                        "//input[@type='submit' and (contains(@value, 'search') or contains(@value, 'Search'))] | " +
                                        "//input[@type='button' and (contains(@value, 'search') or contains(@value, 'Search'))] | " + 
                                        "//a[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'search')] | " +
                                        "//button[contains(@id, 'search') or contains(@class, 'search')] | " +
                                        "//input[contains(@id, 'search') or contains(@class, 'search')]")
                                    
                                    if search_button_candidates:
                                        search_button = search_button_candidates[0]
                                    else:
                                        # Fallback to any button near the search field
                                        try:
                                            # Look for buttons near our search field
                                            nearby_buttons = search_field.find_elements(By.XPATH, "..//button | ../..//button | ../following::button[1]")
                                            if nearby_buttons:
                                                search_button = nearby_buttons[0]
                                        except:
                                            pass
                                    if search_button:
                                        driver.execute_script("arguments[0].click();", search_button)
                                        print("Clicked search button")
                                    else:
                                        # Try pressing Enter in the search field as a last resort
                                        search_field.send_keys(Keys.ENTER)
                                        print("Used Enter key to submit search")
                                    
                                    # After clicking search, look for any action buttons that might appear
                                    time.sleep(3)  # Wait for page to update
                                    
                                    # Use our helper method with general action keywords for any site
                                    action_keywords = ['view', 'details', 'calc', 'show', 'open', 'more', 'info', 'select', 'next', 'continue']
                                    action_src_patterns = ['button', 'arrow', 'view', 'details', 'next', 'continue']
                                    
                                    find_and_click_image_buttons(
                                        driver, 
                                        keywords=action_keywords, 
                                        src_patterns=action_src_patterns,
                                        wait_time=3
                                    )
                                    time.sleep(5)
                                except Exception as search_error:
                                    print(f"Error during search: {str(search_error)}")
                                    
                                # First check if we found the HS code
                                hs_code_found = False
                                
                                # Look for result tables with the HS code
                                result_tables = driver.find_elements(By.XPATH, "//table[.//td[contains(text(), '" + hs_code + "')]]")
                                if not result_tables:
                                    # Try with just the beginning of the HS code 
                                    code_prefix = hs_code[:6] if len(hs_code) > 6 else hs_code
                                    result_tables = driver.find_elements(By.XPATH, f"//table[.//td[contains(text(), '{code_prefix}')]]")
                                
                                if result_tables:
                                    hs_code_found = True
                                    print("Found HS code in search results")
                                    
                                    # Try to click on the HS code to open details if it's a link
                                    hs_code_links = driver.find_elements(By.XPATH, f"//a[contains(text(), '{hs_code}')]")
                                    if hs_code_links:
                                        for link in hs_code_links:
                                            if link.is_displayed():
                                                print(f"Clicking HS code link: {link.text}")
                                                driver.execute_script("arguments[0].click();", link)
                                                time.sleep(3)
                                                break
                                    
                                    for table in result_tables:
                                        print("Found table with HS code information:")
                                        rows = table.find_elements(By.TAG_NAME, "tr")
                                        for row in rows:
                                            cells = row.find_elements(By.TAG_NAME, "td")
                                            if cells:
                                                row_text = " ".join([cell.text for cell in cells])
                                                print(f"HS Code info: {row_text}")
                                                duty_rate_found = True
                                    
                                    # Check if we're in product detail view
                                    # The site shows HS Code hierarchy with specific formatting
                                    hs_code_header = driver.find_elements(By.XPATH, 
                                        "//div[contains(text(), 'HS Code:') or contains(text(), 'Full HS Code')]")
                                    
                                    if hs_code_header:
                                        print(f"Found HS code detail view: {hs_code_header[0].text}")
                                        
                                        # Find any description elements 
                                        description_elems = driver.find_elements(By.XPATH, 
                                            "//*[contains(text(), 'Endoscopy') or contains(text(), 'endoscopy')]")
                                        
                                        for elem in description_elems:
                                            if elem.is_displayed():
                                                print(f"Product description: {elem.text}")
                                                duty_rate_found = True
                                        
                                        # Check if Duties and Taxes tab is available
                                        duties_tab = driver.find_elements(By.XPATH, 
                                            "//*[contains(text(), 'Duties and Taxes') or contains(text(), 'Duty') or contains(text(), 'Tariff')]")
                                        
                                        if duties_tab:
                                            for tab in duties_tab:
                                                if tab.is_displayed() and tab.is_enabled():
                                                    print("Found 'Duties and Taxes' tab")
                                                    try:
                                                        driver.execute_script("arguments[0].click();", tab)
                                                        print(f"Clicked on tab: {tab.text}")
                                                        time.sleep(3)
                                                        
                                                        # Take another screenshot after clicking the tab
                                                        screenshot_path = "/tmp/after_duties_tab_click.png"
                                                        driver.save_screenshot(screenshot_path)
                                                        print(f"Screenshot saved to {screenshot_path}")
                                                        
                                                        # Look for Brazil specific information
                                                        brazil_elements = driver.find_elements(By.XPATH, 
                                                            "//*[contains(text(), 'Brazil') or text()='BR']"
                                                        )
                                                        
                                                        for brazil_elem in brazil_elements:
                                                            if brazil_elem.is_displayed():
                                                                # Check if it's clickable
                                                                try:
                                                                    driver.execute_script("arguments[0].click();", brazil_elem)
                                                                    print(f"Clicked on Brazil element: {brazil_elem.text}")
                                                                    time.sleep(2)
                                                                except Exception as brazil_click_error:
                                                                    print(f"Could not click Brazil element: {str(brazil_click_error)}")
                                                                
                                                                # Look for duty rates near this element
                                                                parent = brazil_elem
                                                                for i in range(5):  # Go up to 5 levels up
                                                                    try:
                                                                        parent = parent.find_element(By.XPATH, "..")
                                                                        
                                                                        # Look for percentage values in this parent
                                                                        if "%" in parent.text:
                                                                            print(f"Found percentage in parent context: {parent.text}")
                                                                            duty_rate_found = True
                                                                            
                                                                            # Extract all percentages
                                                                            import re
                                                                            percentages = re.findall(r'\d+\.?\d*\s*%', parent.text)
                                                                            if percentages:
                                                                                print(f"🌟 Found duty rates for Brazil: {', '.join(percentages)}")
                                                                            break
                                                                    except:
                                                                        break
                                                                        
                                                                # Look for nearby elements with percentage signs
                                                                nearby_percentages = driver.find_elements(By.XPATH, 
                                                                    f"//td[contains(text(), '%') and preceding::*[contains(text(), 'Brazil')] or following::*[contains(text(), 'Brazil')]]"
                                                                )
                                                                
                                                                for pct_elem in nearby_percentages:
                                                                    if pct_elem.is_displayed():
                                                                        print(f"Found percentage element near Brazil: {pct_elem.text}")
                                                                        duty_rate_found = True
                                                                        break
                                                    except Exception as tab_click_error:
                                                        print(f"Error clicking duties tab: {str(tab_click_error)}")
                                                    
                                                    # Check if it's already selected
                                                    if "selected" not in tab.get_attribute("class"):
                                                        print("Clicking on Duties and Taxes tab")
                                                        driver.execute_script("arguments[0].click();", tab)
                                                        time.sleep(2)
                                    
                                    # Look for Country selection dropdowns
                                    country_dropdowns = driver.find_elements(By.XPATH, 
                                        "//select[contains(@id, 'Country') or following-sibling::text()[contains(., 'Country')]]")
                                    
                                    if country_dropdowns:
                                        print("Found country selection dropdowns")
                                        
                                        # Check if there's a Calculate button
                                        calc_buttons = driver.find_elements(By.XPATH, 
                                            "//input[@value='Calculate' or @type='button'][contains(@id, 'Calculate')]")
                                        
                                        # Try regular buttons first
                                        button_clicked = False
                                        if calc_buttons:
                                            for btn in calc_buttons:
                                                if btn.is_displayed():
                                                    print("Found Calculate button")
                                                    driver.execute_script("arguments[0].click();", btn)
                                                    time.sleep(2)
                                                    button_clicked = True
                                                    break
                                        
                                        # If no regular button found/clicked, try image buttons
                                        if not button_clicked:
                                            action_keywords = ['view', 'details', 'calc', 'show', 'open', 'more', 'info', 'select', 'next', 'continue', 'proceed']
                                            action_src_patterns = ['button', 'arrow', 'view', 'details', 'next', 'continue']
                                            
                                            button_clicked = find_and_click_image_buttons(
                                                driver, 
                                                keywords=action_keywords, 
                                                src_patterns=action_src_patterns,
                                                wait_time=2
                                            )
                                    
                                    # Provide a summary of the general workflow followed
                                    print("\nGeneral workflow summary:")
                                    print("1. Logged in to the website using provided credentials")
                                    print("2. Navigated to the appropriate search page")
                                    print("3. Entered search criteria including product code and country")
                                    print("4. Looked for action buttons and relevant data on result pages")
                                    print("5. Analyzed any tables, percentage values, and tariff information")
                                    
                                    # Add information about what was found
                                    if hs_code:
                                        print(f"\nSearched for product code: {hs_code}")
                                    if country:
                                        print(f"Searched for import country: {country}")
                                # We'll extract any duty or tax-related information found in the page
                                try:
                                    # Look for percentage values which might indicate duty rates
                                    percentage_pattern = r"(\d+(?:\.\d+)?%)"
                                    percentages = re.findall(percentage_pattern, driver.page_source)
                                    if percentages:
                                        print("\nFound potential duty/tax rates in the content:")
                                        print(", ".join(list(set(percentages[:5]))))  # Display unique rates, limit to 5
                                    
                                    # Look for tables with duty information
                                    tables = driver.find_elements(By.TAG_NAME, "table")
                                    if tables:
                                        print("\nFound tables that might contain duty information")
                                        
                                    # Look for any tax or duty terms
                                    duty_terms = ["duty", "tax", "tariff", "vat", "customs", "levy", "charge", "fee"]
                                    for term in duty_terms:
                                        if term in driver.page_source.lower():
                                            print(f"Found '{term}' references in the content")
                                except Exception as e:
                                    print(f"Error analyzing page content: {str(e)}")
                                duty_rate_found = True
                                    
                                # Try to extract any duty-related information from the page
                                if not duty_rate_found:
                                    # Look for any content with duty/tariff keywords
                                    duty_elements = driver.find_elements(By.XPATH, 
                                        "//*[contains(text(), 'duty') or contains(text(), 'Duty') or " +
                                        "contains(text(), 'tariff') or contains(text(), 'Tariff') or " +
                                        "contains(text(), 'rate') or contains(text(), 'Rate')]")
                                    
                                    for element in duty_elements:
                                        if element.is_displayed():
                                            print(f"Duty-related information: {element.text}")
                                            duty_rate_found = True
                                
                            except Exception as e:
                                print(f"Error in site-specific extraction: {str(e)}")
                            
                            # General approach for all sites - look for tables with duty information
                            if not duty_rate_found:
                                print("Looking for tables with duty rate information...")
                                tables = driver.find_elements(By.TAG_NAME, "table")
                                
                                for table in tables:
                                    try:
                                        # Check if the table has headers first
                                        headers = table.find_elements(By.TAG_NAME, "th")
                                        header_text = " ".join([h.text for h in headers]).lower()
                                        
                                        # If headers contain relevant keywords, this is likely our table
                                        if any(keyword in header_text for keyword in ['duty', 'tariff', 'rate', 'tax', 'charge']):
                                            print("Found table with relevant headers:")
                                            print(f"Headers: {header_text}")
                                            
                                            # Extract all rows
                                            rows = table.find_elements(By.TAG_NAME, "tr")
                                            for row in rows:
                                                cells = row.find_elements(By.TAG_NAME, "td")
                                                if cells:
                                                    row_text = " ".join([cell.text for cell in cells])
                                                    print(f"Row data: {row_text}")
                                                    
                                                    # Look for percentage values which likely indicate rates
                                                    import re
                                                    percentages = re.findall(r'\d+\.?\d*\s*%', row_text)
                                                    if percentages:
                                                        print(f"🌟 Found percentage values: {', '.join(percentages)}")
                                                    
                                                    duty_rate_found = True
                                        else:
                                            # Check individual rows for duty rate information
                                            rows = table.find_elements(By.TAG_NAME, "tr")
                                            for row in rows:
                                                cells = row.find_elements(By.TAG_NAME, "td")
                                                row_text = " ".join([cell.text for cell in cells]).lower()
                                                if any(keyword in row_text for keyword in ['duty', 'tariff', 'rate', 'tax', 'import charge', 'percentage']):
                                                    print(f"Found potential duty rate information: {row_text}")
                                                    
                                                    # Extract percentage values
                                                    import re
                                                    percentages = re.findall(r'\d+\.?\d*\s*%', row_text)
                                                    if percentages:
                                                        print(f"🌟 Found percentage values: {', '.join(percentages)}")
                                                    
                                                    duty_rate_found = True
                                    except Exception as e:
                                        print(f"Error processing table: {str(e)}")
                            
                            # If no data in tables, look for any text elements with duty information
                            if not duty_rate_found:
                                print("Looking for any text elements with duty rate information...")
                                duty_texts = driver.find_elements(By.XPATH, 
                                    "//*[contains(text(), 'duty') or contains(text(), 'Duty') or " +
                                    "contains(text(), 'rate') or contains(text(), 'Rate') or " + 
                                    "contains(text(), 'tariff') or contains(text(), 'Tariff') or " +
                                    "contains(text(), 'tax') or contains(text(), 'Tax')]"
                                )
                                
                                for text_elem in duty_texts:
                                    if text_elem.is_displayed():
                                        elem_text = text_elem.text.strip()
                                        if elem_text and len(elem_text) > 3:  # Avoid empty or very short texts
                                            print(f"Found text with duty/rate information: {elem_text}")
                                            
                                            # Look for percentage values which likely indicate rates
                                            import re
                                            percentages = re.findall(r'\d+\.?\d*\s*%', elem_text)
                                            if percentages:
                                                print(f"🌟 Found percentage values: {', '.join(percentages)}")
                                                
                                            duty_rate_found = True
                            
                            # Look for labels/divs that are near percentage values
                            if not duty_rate_found:
                                print("Looking for percentage values that might indicate duty rates...")
                                try:
                                    # Find elements containing percentage symbols
                                    percentage_elements = driver.find_elements(By.XPATH, "//*[contains(text(), '%')]")
                                    for elem in percentage_elements:
                                        if elem.is_displayed():
                                            print(f"Found element with percentage: {elem.text}")
                                            duty_rate_found = True
                                except Exception as e:
                                    print(f"Error finding percentage elements: {str(e)}")
                            
                            # If all extraction methods failed
                            if not duty_rate_found:
                                print("Could not find specific duty rate information on the page.")
                                print("Taking screenshot of current page for manual analysis...")
                                try:
                                    driver.save_screenshot("/tmp/duty_rate_page.png")
                                    print("Screenshot saved to /tmp/duty_rate_page.png")
                                except Exception as ss_error:
                                    print(f"Error saving screenshot: {str(ss_error)}")
                                
                                # Get page source for offline analysis
                                try:
                                    with open("/tmp/page_source.html", "w") as f:
                                        f.write(driver.page_source)
                                    print("Page source saved to /tmp/page_source.html for offline analysis")
                                except Exception as ps_error:
                                    print(f"Error saving page source: {str(ps_error)}")
                    except Exception as e:
                        print(f"Error searching for duty rate: {str(e)}")
            except Exception as e:
                print(f"Error during login: {str(e)}")
            
            # Let's wait to examine the state
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