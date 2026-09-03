CREATE TABLE IF NOT EXISTS papers (
    id               TEXT PRIMARY KEY,
    title            TEXT NOT NULL,
    authors          TEXT[],
    date_released    DATE,
    date_updated     DATE,
    doi              TEXT,
    categories       TEXT[],
    primary_category TEXT,
    source_url       TEXT,
    n_pages          INTEGER,
    n_tables         INTEGER,
    n_chars          INTEGER,
    md_path          TEXT,
    tables_path      TEXT,
    raw              JSONB
);

CREATE INDEX IF NOT EXISTS idx_papers_categories ON papers USING GIN (categories);
CREATE INDEX IF NOT EXISTS idx_papers_primary    ON papers (primary_category);
CREATE INDEX IF NOT EXISTS idx_papers_released   ON papers (date_released);
CREATE INDEX IF NOT EXISTS idx_papers_doi        ON papers (doi) WHERE doi IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_papers_title_fts
    ON papers USING GIN (to_tsvector('english', coalesce(title, '')));
CREATE INDEX IF NOT EXISTS idx_papers_has_tables ON papers (n_tables) WHERE n_tables > 0;

-- Example queries
--
--   -- papers per category (the EDA notebook's headline chart, in SQL)
--   SELECT primary_category, count(*) FROM papers GROUP BY 1 ORDER BY 2 DESC;
--
--   -- everything cross-listed to cs.LG, newest first
--   SELECT id, title FROM papers WHERE categories @> ARRAY['cs.LG']
--   ORDER BY date_released DESC LIMIT 50;
--
--   -- title search
--   SELECT id, title FROM papers
--   WHERE to_tsvector('english', title) @@ plainto_tsquery('english', 'graph neural network');
--
--   -- the tables files worth feeding to a retrieval index
--   SELECT id, n_tables, tables_path FROM papers
--   WHERE n_tables > 0 ORDER BY n_tables DESC;
