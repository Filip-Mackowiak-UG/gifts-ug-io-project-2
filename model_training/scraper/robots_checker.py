import requests


def can_scrape(site_url):
    robots_url = f"{site_url}/robots.txt"
    response = requests.get(robots_url)

    if response.status_code == 200:
        print(response.text)
        return True
    else:
        print(f"Failed to retrieve robots.txt from {site_url}")
        return False


site_url = 'https://www.ebay.com/'
can_scrape(site_url)
