# Dynamic Web Crawler

A powerful, dynamic web crawler that can intelligently interact with any website structure, fill forms, extract data, and navigate complex interfaces.

## Features

- **Intelligent site navigation**: Automatically identifies login forms, search fields, and interactive elements
- **Automatic form filling**: Extracts data from user instructions or uses realistic random data
- **Product code recognition**: Identifies HS codes, HTS codes, and other product classification systems
- **Dynamic country selection**: Handles various country selection interfaces and formats
- **Duty rate extraction**: Finds and extracts duty rates and tariff information from various sources
- **Browser automation**: Uses undetected_chromedriver to avoid detection
- **Advanced search handling**: Deals with interactable and non-interactable search fields
- **Natural language instruction processing**: Extracts key information from user instructions

## Usage

```python
python main.py
```

When prompted, enter a natural language instruction such as:

```
Go to https://export.customsinfo.com/Default.aspx. Go sign into the customs info site 
(the one the government kind of endorses but it's owned by descartes). 
Use the email a02324348@usu.edu to sign in. Then find the duty rate for 9018.19.10 to Brazil.
```

The crawler will:
1. Extract the URL, email, product code, and country
2. Navigate to the specified website
3. Find and fill the login form with the email
4. Search for the specified product code
5. Select the target country
6. Extract and report duty rate information

## Implementation Details

The crawler works in several stages:

1. **Instruction parsing**: Extracts URLs, emails, product codes, and countries
2. **Site navigation**: Opens the website and identifies login forms or relevant links
3. **Form interaction**: Fills forms with extracted or generated data
4. **Search execution**: Finds and interacts with search functionality
5. **Data extraction**: Locates and extracts the requested information
6. **Result reporting**: Provides detailed output of findings

## Troubleshooting

- If Chrome/Chromium fails to start, the crawler will attempt to detect available browsers
- For sites with complex JavaScript, the crawler uses multiple fallback mechanisms
- Interactable elements are made interactable using JavaScript when needed

## Requirements

- Python 3.8+
- Packages in requirements.txt
- Chrome/Chromium browser

## Environment Variables

Place in a `.env` file:

```
OPENAI_API_KEY='your-openai-api-key'
```

## Contributions

Contributions are welcome! Please submit pull requests with improvements.

## License

MIT License