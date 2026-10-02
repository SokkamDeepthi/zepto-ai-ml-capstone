import requests
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path
import re


BASE_DIR = Path(__file__).parent
OUTPUT_FILE = BASE_DIR / "books_cleaned.csv"

BASE_URL = "https://books.toscrape.com/catalogue/page-{}.html"

GBP_TO_INR = 105.50

RATING_MAP = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}


# ---------------------------------------------------------
# Get category from book detail page
# ---------------------------------------------------------

def get_category(book_url):

    response = requests.get(
        book_url,
        headers=HEADERS,
        timeout=15
    )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.content,
        "html.parser"
    )

    breadcrumb = soup.select("ul.breadcrumb li")

    if len(breadcrumb) >= 3:
        return breadcrumb[2].get_text(
            strip=True
        )

    return "Unknown"


# ---------------------------------------------------------
# Extract numeric price safely
# ---------------------------------------------------------

def extract_price(price_text):

    if not price_text:
        return None

    # Find number such as 51.77 from:
    # £51.77
    # Â£51.77
    # GBP 51.77
    # etc.

    match = re.search(
        r"\d+(?:\.\d+)?",
        price_text
    )

    if match:
        return float(match.group())

    return None


# ---------------------------------------------------------
# Scrape one page
# ---------------------------------------------------------

def scrape_page(page_number):

    url = BASE_URL.format(page_number)

    print(
        f"Scraping page {page_number}: {url}"
    )

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=15
    )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.content,
        "html.parser"
    )

    books = soup.select(
        "article.product_pod"
    )

    page_data = []

    for book in books:

        # -------------------------------------------------
        # Title
        # -------------------------------------------------

        title_tag = book.select_one(
            "h3 a"
        )

        title = (
            title_tag.get("title", "").strip()
            if title_tag
            else ""
        )


        # -------------------------------------------------
        # Price
        # -------------------------------------------------

        price_tag = book.select_one(
            ".price_color"
        )

        price_text = (
            price_tag.get_text(
                strip=True
            )
            if price_tag
            else ""
        )

        price_gbp = extract_price(
            price_text
        )


        # -------------------------------------------------
        # Rating
        # -------------------------------------------------

        rating_tag = book.select_one(
            ".star-rating"
        )

        if rating_tag:

            classes = rating_tag.get(
                "class",
                []
            )

            star_rating = (
                classes[-1]
                if classes
                else "Unknown"
            )

            rating = RATING_MAP.get(
                star_rating
            )

        else:

            star_rating = "Unknown"
            rating = None


        # -------------------------------------------------
        # Availability
        # -------------------------------------------------

        availability_tag = book.select_one(
            ".availability"
        )

        availability = (
            availability_tag.get_text(
                " ",
                strip=True
            )
            if availability_tag
            else ""
        )

        in_stock = (
            "In stock" in availability
        )


        # -------------------------------------------------
        # Detail page URL
        # -------------------------------------------------

        if title_tag:

            detail_link = title_tag.get(
                "href"
            )

            detail_url = (
                "https://books.toscrape.com/catalogue/"
                + detail_link
            )

            category = get_category(
                detail_url
            )

        else:

            category = "Unknown"


        # -------------------------------------------------
        # Store record
        # -------------------------------------------------

        page_data.append({

            "title": title,

            "price_gbp": price_gbp,

            "rating": rating,

            "in_stock": in_stock,

            "category": category,

            "star_rating": star_rating,

            "availability": availability

        })

    return page_data


# =========================================================
# MAIN
# =========================================================

print("=" * 60)
print("ZEPTO BOOKS SCRAPER")
print("=" * 60)


all_books = []


# First 5 pages = 100 books

for page_number in range(1, 6):

    page_books = scrape_page(
        page_number
    )

    all_books.extend(
        page_books
    )

    print(
        f"Books collected so far: "
        f"{len(all_books)}"
    )


# ---------------------------------------------------------
# Create DataFrame
# ---------------------------------------------------------

df = pd.DataFrame(
    all_books
)


print("\nRaw dataset preview:")
print(
    df.head()
)


# ---------------------------------------------------------
# Price cleaning
# ---------------------------------------------------------

df["price_gbp"] = pd.to_numeric(
    df["price_gbp"],
    errors="coerce"
)


# ---------------------------------------------------------
# Rating cleaning
# ---------------------------------------------------------

df["rating"] = pd.to_numeric(
    df["rating"],
    errors="coerce"
)


# ---------------------------------------------------------
# Check missing prices
# ---------------------------------------------------------

missing_prices = (
    df["price_gbp"]
    .isna()
    .sum()
)


print(
    f"\nMissing price values: "
    f"{missing_prices}"
)


if missing_prices > 0:

    median_price = (
        df["price_gbp"]
        .median()
    )

    if pd.isna(median_price):

        raise ValueError(
            "All price values are invalid. "
            "Price extraction failed."
        )

    df["price_gbp"] = (
        df["price_gbp"]
        .fillna(median_price)
    )

    print(
        f"Median price used: "
        f"{median_price:.2f}"
    )


# ---------------------------------------------------------
# Check missing ratings
# ---------------------------------------------------------

missing_ratings = (
    df["rating"]
    .isna()
    .sum()
)


print(
    f"Missing rating values: "
    f"{missing_ratings}"
)


if missing_ratings > 0:

    median_rating = round(
        df["rating"].median()
    )

    df["rating"] = (
        df["rating"]
        .fillna(median_rating)
    )

    print(
        f"Median rating used: "
        f"{median_rating}"
    )


# ---------------------------------------------------------
# Convert final types
# ---------------------------------------------------------

df["price_gbp"] = (
    df["price_gbp"]
    .astype(float)
)

df["rating"] = (
    df["rating"]
    .astype(int)
)

df["in_stock"] = (
    df["in_stock"]
    .astype(bool)
)


# ---------------------------------------------------------
# GBP → INR
# ---------------------------------------------------------

df["price_inr"] = (
    df["price_gbp"]
    * GBP_TO_INR
)


# ---------------------------------------------------------
# Column order
# ---------------------------------------------------------

df = df[
    [
        "title",
        "price_gbp",
        "price_inr",
        "rating",
        "in_stock",
        "category",
        "star_rating",
        "availability"
    ]
]


# =========================================================
# VALIDATION
# =========================================================

print("\n" + "=" * 60)
print("DATASET VALIDATION")
print("=" * 60)


print(
    f"\nTotal books: {len(df)}"
)

print(
    f"Unique categories: "
    f"{df['category'].nunique()}"
)

print(
    f"Missing prices: "
    f"{df['price_gbp'].isna().sum()}"
)

print(
    f"Missing ratings: "
    f"{df['rating'].isna().sum()}"
)

print(
    f"Missing categories: "
    f"{df['category'].isna().sum()}"
)


# ---------------------------------------------------------
# Required validations
# ---------------------------------------------------------

if len(df) < 60:

    raise ValueError(
        "Dataset must contain at least 60 books."
    )


if df["category"].nunique() < 3:

    raise ValueError(
        "Dataset must contain at least 3 categories."
    )


if df["price_gbp"].isna().any():

    raise ValueError(
        "price_gbp still contains missing values."
    )


if df["rating"].isna().any():

    raise ValueError(
        "rating still contains missing values."
    )


# ---------------------------------------------------------
# Save CSV
# ---------------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nSaved file:")
print(OUTPUT_FILE)


# ---------------------------------------------------------
# Final information
# ---------------------------------------------------------

print("\nFinal data types:")
print(df.dtypes)


print("\nFirst 5 rows:")
print(df.head())


print("\n" + "=" * 60)
print("SCRAPING + CLEANING COMPLETED")
print("=" * 60)