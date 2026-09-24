import sqlite3
import pandas as pd

# Load cleaned data
df = pd.read_csv("data_pipeline/books_cleaned.csv")

# Connect to SQLite database
conn = sqlite3.connect("data_pipeline/books.db")
cursor = conn.cursor()

# Enable foreign keys
cursor.execute("PRAGMA foreign_keys = ON")

# Create categories table
cursor.execute("""
CREATE TABLE IF NOT EXISTS categories (
    category_id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT UNIQUE NOT NULL
)
""")

# Create books table
cursor.execute("""
CREATE TABLE IF NOT EXISTS books (
    book_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    price_gbp REAL,
    price_inr REAL,
    rating INTEGER,
    in_stock INTEGER,
    category_id INTEGER,
    FOREIGN KEY (category_id) REFERENCES categories(category_id)
)
""")

# Clear old data if script is run again
cursor.execute("DELETE FROM books")
cursor.execute("DELETE FROM categories")

# Insert categories
for category in df["category"].dropna().unique():
    cursor.execute(
        "INSERT INTO categories (category_name) VALUES (?)",
        (category,)
    )

# Insert books
for _, row in df.iterrows():

    cursor.execute(
        "SELECT category_id FROM categories WHERE category_name = ?",
        (row["category"],)
    )

    result = cursor.fetchone()

    if result is None:
        continue

    category_id = result[0]

    cursor.execute("""
        INSERT INTO books
        (title, price_gbp, price_inr, rating, in_stock, category_id)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        row["title"],
        row["price_gbp"],
        row["price_inr"],
        row["rating"],
        int(row["in_stock"]),
        category_id
    ))

# Save database
conn.commit()

# Verify records
cursor.execute("SELECT COUNT(*) FROM books")
book_count = cursor.fetchone()[0]

cursor.execute("SELECT COUNT(*) FROM categories")
category_count = cursor.fetchone()[0]

print("Database created successfully!")
print("Books inserted:", book_count)
print("Categories inserted:", category_count)

# Show sample JOIN
print("\nSample data from JOIN:")

cursor.execute("""
SELECT
    books.title,
    books.price_gbp,
    books.price_inr,
    books.rating,
    books.in_stock,
    categories.category_name
FROM books
JOIN categories
ON books.category_id = categories.category_id
LIMIT 5
""")

for row in cursor.fetchall():
    print(row)

conn.close()