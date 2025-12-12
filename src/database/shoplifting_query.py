import json
import logging
import os
from psycopg2.pool import SimpleConnectionPool
from dotenv import load_dotenv

logger = logging.getLogger("detection")
logger.setLevel(logging.INFO)

load_dotenv()

# ---------------------------------------------------------
# 1️⃣ CREATE CONNECTION POOL
# ---------------------------------------------------------
def setup_database():
    try:
        pool = SimpleConnectionPool(
            1, 20,   # min=1 max=20 connections
            host=os.environ["DB_HOST"],
            dbname=os.environ["DB_NAME"],
            user=os.environ["DB_USER"],
            password=os.environ["DB_PASSWORD"],
            port=int(os.environ.get("DB_PORT", 5432))
        )
        logger.info("✅ PostgreSQL Connection Pool Created")
        return pool
    except Exception as e:
        logger.error(f"❌ Failed to create pool: {e}")
        raise


# Global pool
pool = setup_database()

# ---------------------------------------------------------
# 2️⃣ INSERT SHOPLIFTING FRAME USING POOL
# ---------------------------------------------------------
def insert_shoplifting_frame(data: dict, s3_url: str):
    """
    Insert a shoplifting frame into the shop_lifting table
    using PostgreSQL connection pool.
    """
    conn = pool.getconn()

    insert_query = """
        INSERT INTO shop_lifting (
            cam_id,
            org_id,
            user_id,
            frame_id,
            timestamp,
            persons,
            alerts,
            message,
            s3_url,
            status
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s)
        RETURNING id;
    """

    try:
        with conn.cursor() as cursor:
            cursor.execute(
                insert_query,
                (
                    data['cam_id'],
                    data['org_id'],
                    data['user_id'],
                    data['frame_id'],
                    data['timestamp'],
                    data['persons'],               # integer array
                    json.dumps(data['alerts']),    # JSONB
                    data['message'],
                    s3_url,
                    data['status']
                )
            )

            inserted_id = cursor.fetchone()[0]
            conn.commit()

            logger.info(
                f"✅ Shoplifting frame stored: id={inserted_id}, frame_id={data['frame_id']}"
            )
            return inserted_id

    except Exception as e:
        conn.rollback()
        logger.error(f"❌ Failed to insert shoplifting frame: {e}")
        return None

    finally:
        pool.putconn(conn)
