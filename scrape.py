import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

BASE_URL = "https://learn.microsoft.com"
TOPIC_PATHS = {
    "fabric": "/en-us/fabric/",
    "apim": "/en-us/azure/api-management/",
    "aks": "/en-us/azure/aks/",
    "mobile": "/en-us/azure/developer/mobile-apps/",
    "dbs": "/en-us/azure/architecture/data-guide/technology-choices/data-store-overview"
}

visited_urls = set()

def crawl_topic(topic_name, start_path, max_pages=20):
    full_start_url = urljoin(BASE_URL, start_path)
    to_visit = [full_start_url]
    results = []

    while to_visit and len(results) < max_pages:
        url = to_visit.pop(0)
        if url in visited_urls or not url.startswith(BASE_URL):
            continue
        visited_urls.add(url)

        try:
            res = requests.get(url, headers=HEADERS, timeout=10)
            soup = BeautifulSoup(res.text, "html.parser")
            title = soup.title.string.strip() if soup.title else "No title"
            results.append((title, url))
            print(f"[{topic_name.upper()}] Collected: {title}")

            for a in soup.find_all("a", href=True):
                href = a['href']
                if href.startswith("/en-us") and not any(x in href for x in ["/blog", "/support", "/legal"]):
                    next_url = urljoin(BASE_URL, href)
                    if next_url not in visited_urls:
                        to_visit.append(next_url)

            time.sleep(0.5)  # Be respectful
        except Exception as e:
            print(f"Error crawling {url}: {e}")
            continue

    return results

def main():
    all_results = {}
    for topic, path in TOPIC_PATHS.items():
        print(f"\n🚀 Crawling: {topic}")
        pages = crawl_topic(topic, path)
        all_results[topic] = pages

    # Print summary
    for topic, links in all_results.items():
        print(f"\n📚 {topic.upper()} - {len(links)} pages found:")
        for title, link in links:
            print(f"- {title}\n  {link}")
    return all_results

if __name__ == "__main__":
    main()