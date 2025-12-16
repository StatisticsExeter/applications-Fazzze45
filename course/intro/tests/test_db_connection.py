import os
import psycopg2
import pytest
from course.utils import load_db_config

pytest.skip("Skipping database connection test — no local PostgreSQL", allow_module_level=True)


@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true",
    reason="Skipping database test in GitHub Actions"
)
def test_can_read_hello_world():
    config = load_db_config()
    conn = psycopg2.connect(**config)
    cur = conn.cursor()
    cur.execute("SELECT message FROM hello_world LIMIT 1;")
    result = cur.fetchone()
    assert result[0] == "Hello, World!"
    cur.close()
    conn.close()
