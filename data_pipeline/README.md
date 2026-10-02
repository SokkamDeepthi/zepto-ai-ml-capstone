# Module 1 — Data Pipeline

## Overview

This module builds an end-to-end data pipeline using the Books to Scrape website.

The pipeline performs:

1. Web scraping using `requests` and `BeautifulSoup`
2. Data cleaning and type conversion using `pandas`
3. GBP to INR price conversion
4. SQLite database creation
5. SQL querying
6. Pandas `pd.read_sql_query()` verification
7. Pandas `pd.merge()` verification of the SQL JOIN

---

## Dataset

The pipeline scrapes the first 5 pages of the Books to Scrape website.

- Total books scraped: 100
- Minimum required books: 60
- Multiple book categories are included
- Categories are extracted from the individual book detail pages

### Fields collected

- `title`
- `price_gbp`
- `star_rating`
- `availability`
- `category`

### Cleaned fields

- `price_gbp` — float
- `price_inr` — float
- `rating` — integer from 1 to 5
- `in_stock` — boolean
- `category` — text

---

## Project Files

```text
data_pipeline/
│
├── scrape_books.py
├── books_cleaned.csv
├── database.py
├── books.db
├── queries.py
├── query_outputs.txt
└── README.md