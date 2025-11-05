# NIDCR Dental Corpus

**Namespace**: dental  
**Last robots.txt check**: 2025-11-03

## Crawl Policy
- Host: nidcr.nih.gov
- Respect robots.txt allowances below (mirrored for quick reference)
- Rate limit ~1 req/sec/domain; back off on 429/5xx
- User-Agent: AskMyDocsBot/0.1 (contact: you@example.com)

### Allowed Scope
1. `/health-info` (topics index) → follow depth-1 to `/health-info/<topic-slug>` pages
2. `/health-info/publications` → follow depth-1 to first-party PDF links
3. Include directly linked official PDFs from optional Oral Health in America report set when present.

### Notes
- U.S. federal site; content generally public domain, but always cite canonical URLs and “Last Reviewed” dates.
- Skip disallowed paths such as search, user, admin, taxonomy, forms, and `/now-leaving/` interstitials.

### robots.txt Snapshot
```
User-agent: *
# CSS, JS, Images
Allow: /core/*.css$
Allow: /core/*.css?
Allow: /core/*.js$
Allow: /core/*.js?
Allow: /core/*.gif
Allow: /core/*.jpg
Allow: /core/*.jpeg
Allow: /core/*.png
Allow: /core/*.svg
Allow: /profiles/*.css$
Allow: /profiles/*.css?
Allow: /profiles/*.js$
Allow: /profiles/*.js?
Allow: /profiles/*.gif
Allow: /profiles/*.jpg
Allow: /profiles/*.jpeg
Allow: /profiles/*.png
Allow: /profiles/*.svg
# Directories
Disallow: /core/
Disallow: /profiles/
# Files
Disallow: /README.txt
Disallow: /web.config
# Paths (clean URLs)
Disallow: /admin/
Disallow: /comment/reply/
Disallow: /filter/tips
Disallow: /node/add/
Disallow: /search/
Disallow: /user/register/
Disallow: /user/password/
Disallow: /user/login/
Disallow: /user/logout/
# Paths (no clean URLs)
Disallow: /index.php/admin/
Disallow: /index.php/comment/reply/
Disallow: /index.php/filter/tips
Disallow: /index.php/node/add/
Disallow: /index.php/search/
Disallow: /index.php/user/password/
Disallow: /index.php/user/register/
Disallow: /index.php/user/login/
Disallow: /index.php/user/logout/
Disallow: /now-leaving/
Disallow: /taxonomy/term/
Disallow: /quizquestion/
Disallow: /form/
```
