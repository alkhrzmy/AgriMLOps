import argparse
import sqlite3
from pathlib import Path


def batch_validate_pending(database_path: Path, validator_note: str) -> int:
    if not database_path.exists():
        print(f"Database not found: {database_path}")
        return 1

    # Find all pending items and their latest feedback suggested_label
    query = """
        SELECT
            al.id AS item_id,
            al.prediction_id,
            al.predicted_label,
            al.confidence,
            al.reason,
            fl.suggested_label
        FROM active_learning_queue al
        LEFT JOIN (
            SELECT prediction_id, suggested_label
            FROM feedback_logs
            WHERE suggested_label IS NOT NULL
              AND TRIM(suggested_label) != ''
            ORDER BY created_at DESC
        ) fl ON fl.prediction_id = al.prediction_id
        WHERE al.status = 'pending'
        GROUP BY al.id
    """

    with sqlite3.connect(database_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query).fetchall()

        if not rows:
            print("No pending active learning items found.")
            return 0

        print(f"Found {len(rows)} pending items to validate:\n")
        for row in rows:
            print(f"  {row['item_id']} | {row['predicted_label']} | confidence={row['confidence']:.4f} | reason={row['reason']}")
            if row['suggested_label']:
                print(f"    -> will validate as: {row['suggested_label']}")
            else:
                print(f"    -> WARNING: no suggested_label found, skipping")

        # Validate each item
        validated_count = 0
        skipped_count = 0
        for row in rows:
            suggested = row["suggested_label"]
            if not suggested or not str(suggested).strip():
                skipped_count += 1
                continue

            conn.execute(
                """
                UPDATE active_learning_queue
                SET status = 'validated',
                    validated_label = ?,
                    validator_note = ?,
                    validated_at = datetime('now')
                WHERE id = ?
                """,
                (suggested, validator_note, row["item_id"]),
            )
            validated_count += 1

        conn.commit()

    print(f"\nValidation complete:")
    print(f"  - validated: {validated_count}")
    print(f"  - skipped (no suggested_label): {skipped_count}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch validate all pending active learning items using their feedback suggested_label.")
    parser.add_argument("--database", default="data/agrimlops.db")
    parser.add_argument("--validator-note", default="batch validated from controlled feedback simulation")
    args = parser.parse_args()
    return batch_validate_pending(Path(args.database), args.validator_note)


if __name__ == "__main__":
    raise SystemExit(main())
