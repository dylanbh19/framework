import requests
from bs4 import BeautifulSoup

SEARCH_TERMS = {
    "fabric": "Microsoft Fabric whitepaper",
    "apim": "Azure API Management whitepaper",
    "aks": "Azure Kubernetes Service whitepaper",
    "mobile": "Azure Mobile App Development whitepaper",
    "dbs": "Azure Database whitepaper"
}

BASE_URL = "https://www.bing.com/search?q=site%3Amicrosoft.com+{}"

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

def fetch_whitepapers(query):
    search_url = BASE_URL.format(query.replace(" ", "+"))
    response = requests.get(search_url, headers=HEADERS)
    soup = BeautifulSoup(response.text, "html.parser")
    
    results = []
    for item in soup.select("li.b_algo h2 a"):
        title = item.text.strip()
        url = item.get("href")
        if "whitepaper" in title.lower() or "white paper" in title.lower():
            results.append((title, url))
    return results

def main():
    all_results = {}
    for key, query in SEARCH_TERMS.items():
        print(f"\n🔍 Searching for: {query}")
        results = fetch_whitepapers(query)
        if results:
            all_results[key] = results
            for title, link in results:
                print(f"- {title}\n  {link}")
        else:
            print("No whitepapers found.")
    return all_results

if __name__ == "__main__":
    main()