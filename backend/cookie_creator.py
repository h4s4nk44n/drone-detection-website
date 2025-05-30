# cookie_cleaner.py
import browser_cookie3

with open("cookies.txt", "w", encoding="utf-8") as f:
    cj = browser_cookie3.firefox(domain_name="youtube.com")
    for cookie in cj:
        expires = int(cookie.expires) if cookie.expires else 0
        f.write(f"{cookie.domain}\t{'TRUE' if cookie.domain.startswith('.') else 'FALSE'}\t")
        f.write(f"{cookie.path}\t{'TRUE' if cookie.secure else 'FALSE'}\t")
        f.write(f"{expires}\t{cookie.name}\t{cookie.value}\n")
