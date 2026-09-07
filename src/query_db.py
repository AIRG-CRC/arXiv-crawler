import sqlite3
conn = sqlite3.connect("data/manifest.db")
total, with_doi = conn.execute(
    "SELECT COUNT(*), COUNT(doi) FROM papers"
).fetchone()
print(f"{with_doi:,} / {total:,} papers have a DOI ({with_doi/total:.1%})")