import sqlite3
from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).parent
CSV_FILE = BASE_DIR / "books_cleaned.csv"
DB_FILE = BASE_DIR / "books.db"


print("=" * 60)
print("ZEPTO DATA PIPELINE - DATABASE LOADING")
print("=" * 60)


# ---------------------------------------------------------
# 1. Load cleaned CSV
# ---------------------------------------------------------

df = pd.read_csv(CSV_FILE)

print(f"\nLoaded rows: {len(df)}")
print(f"Loaded columns: {list(df.columns)}")


# ---------------------------------------------------------
# 2. Make sure numeric columns are valid
# ---------------------------------------------------------

df["price_gbp"] = pd.to_numeric(df["price_gbp"], errors="coerce")
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
df["price_inr"] = pd.to_numeric(df["price_inr"], errors="coerce")


# ---------------------------------------------------------
# 3. Handle missing numeric values
# ---------------------------------------------------------

if df["price_gbp"].isna().any():
    median_price = df["price_gbp"].median()
    df["price_gbp"] = df["price_gbp"].fillna(median_price)
    print(f"\nMissing price_gbp values filled with median: {median_price:.2f}")


if df["rating"].isna().any():
    median_rating = round(df["rating"].median())
    df["rating"] = df["rating"].fillna(median_rating)
    print(f"Missing rating values filled with median: {median_rating}")


# Recalculate INR from cleaned GBP price
df["price_inr"] = df["price_gbp"] * 105.50


# Convert types explicitly
df["price_gbp"] = df["price_gbp"].astype(float)
df["price_inr"] = df["price_inr"].astype(float)
df["rating"] = df["rating"].astype(int)
df["in_stock"] = df["in_stock"].astype(bool)


# ---------------------------------------------------------
# 4. Connect to SQLite
# ---------------------------------------------------------

connection = sqlite3.connect(DB_FILE)
cursor = connection.cursor()

cursor.execute("PRAGMA foreign_keys = ON")


# ---------------------------------------------------------
# 5. Recreate tables
# ---------------------------------------------------------

cursor.execute("DROP TABLE IF EXISTS books")
cursor.execute("DROP TABLE IF EXISTS categories")


cursor.execute("""
    CREATE TABLE categories (
        category_id INTEGER PRIMARY KEY AUTOINCREMENT,
        category_name TEXT UNIQUE NOT NULL
    )
""")


cursor.execute("""
    CREATE TABLE books (
        book_id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        price_gbp REAL NOT NULL,
        price_inr REAL NOT NULL,
        rating INTEGER NOT NULL,
        in_stock INTEGER NOT NULL,
        category_id INTEGER NOT NULL,
        FOREIGN KEY (category_id)
            REFERENCES categories(category_id)
    )
""")


# ---------------------------------------------------------
# 6. Insert categories
# ---------------------------------------------------------

categories = sorted(
    df["category"]
    .dropna()
    .unique()
)


for category in categories:

    cursor.execute(
        """
        INSERT INTO categories (category_name)
        VALUES (?)
        """,
        (category,)
    )


# Create category → ID mapping

cursor.execute(
    "SELECT category_id, category_name FROM categories"
)

category_rows = cursor.fetchall()

category_mapping = {
    category_name: category_id
    for category_id, category_name in category_rows
}


# ---------------------------------------------------------
# 7. Insert books
# ---------------------------------------------------------

for _, row in df.iterrows():

    category_id = category_mapping[row["category"]]

    cursor.execute(
        """
        INSERT INTO books (
            title,
            price_gbp,
            price_inr,
            rating,
            in_stock,
            category_id
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            row["title"],
            float(row["price_gbp"]),
            float(row["price_inr"]),
            int(row["rating"]),
            int(row["in_stock"]),
            category_id
        )
    )


# ---------------------------------------------------------
# 8. Save database
# ---------------------------------------------------------

connection.commit()


# ---------------------------------------------------------
# 9. Verify database
# ---------------------------------------------------------

category_count = cursor.execute(
    "SELECT COUNT(*) FROM categories"
).fetchone()[0]


book_count = cursor.execute(
    "SELECT COUNT(*) FROM books"
).fetchone()[0]


print("\nDatabase created successfully.")
print(f"Categories inserted: {category_count}")
print(f"Books inserted: {book_count}")


print("\nDatabase file:")
print(DB_FILE)


# ---------------------------------------------------------
# 10. Check foreign key
# ---------------------------------------------------------

foreign_keys = cursor.execute(
    "PRAGMA foreign_key_list(books)"
).fetchall()


print("\nForeign key definition:")

for fk in foreign_keys:
    print(fk)


# ---------------------------------------------------------
# 11. JOIN preview
# ---------------------------------------------------------

preview = pd.read_sql_query(
    """
    SELECT
        b.book_id,
        b.title,
        b.price_gbp,
        b.price_inr,
        b.rating,
        b.in_stock,
        c.category_name
    FROM books b
    JOIN categories c
        ON b.category_id = c.category_id
    LIMIT 5
    """,
    connection
)


print("\nJOIN preview:")
print(preview)


# ---------------------------------------------------------
# 12. Final checks
# ---------------------------------------------------------

print("\nFinal database checks:")

print(
    "Books with NULL price:",
    cursor.execute(
        "SELECT COUNT(*) FROM books WHERE price_gbp IS NULL"
    ).fetchone()[0]
)

print(
    "Books with NULL rating:",
    cursor.execute(
        "SELECT COUNT(*) FROM books WHERE rating IS NULL"
    ).fetchone()[0]
)

print(
    "Books with NULL category:",
    cursor.execute(
        "SELECT COUNT(*) FROM books WHERE category_id IS NULL"
    ).fetchone()[0]
)


connection.close()


print("\n" + "=" * 60)
print("DATABASE CREATION COMPLETED")
print("=" * 60)