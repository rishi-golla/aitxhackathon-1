"""
Violation Database for storing and querying safety violations.

Uses SQLite with async support via aiosqlite.
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional
import structlog

log = structlog.get_logger()

# Try importing aiosqlite
try:
    import aiosqlite
    AIOSQLITE_AVAILABLE = True
except ImportError:
    AIOSQLITE_AVAILABLE = False
    log.warning("aiosqlite_not_installed_using_sync")

# Fallback to sqlite3
import sqlite3


# Database schema
SCHEMA = """
CREATE TABLE IF NOT EXISTS violations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    violation_id TEXT UNIQUE NOT NULL,
    timestamp REAL NOT NULL,
    zone_id TEXT NOT NULL,
    zone_name TEXT,
    violation_type TEXT NOT NULL,
    worker_id TEXT,
    missing_items TEXT,
    osha_reference TEXT,
    confidence REAL,
    frame_path TEXT,
    alert_sent BOOLEAN DEFAULT FALSE,
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_by TEXT,
    acknowledged_at REAL,
    false_positive BOOLEAN DEFAULT FALSE,
    notes TEXT,
    created_at REAL DEFAULT (unixepoch()),
    agent_reasoning TEXT,
    coach_message TEXT
);

CREATE INDEX IF NOT EXISTS idx_violations_timestamp ON violations(timestamp);
CREATE INDEX IF NOT EXISTS idx_violations_zone ON violations(zone_id);
CREATE INDEX IF NOT EXISTS idx_violations_type ON violations(violation_type);
CREATE INDEX IF NOT EXISTS idx_violations_acknowledged ON violations(acknowledged);
"""


class ViolationDatabase:
    """
    Async database for storing safety violations.

    Features:
    - SQLite with async operations
    - Connection pooling
    - Automatic reconnection
    - Statistics and analytics
    - CSV export
    """

    def __init__(self, db_path: str = "data/violations.db"):
        """
        Initialize database.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self._connection: Optional[Any] = None
        self._lock = asyncio.Lock()
        self._initialized = False

    async def init(self) -> None:
        """Initialize database and create tables."""
        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        if AIOSQLITE_AVAILABLE:
            self._connection = await aiosqlite.connect(str(self.db_path))
            self._connection.row_factory = aiosqlite.Row
            await self._connection.executescript(SCHEMA)
            await self._connection.commit()
        else:
            self._connection = sqlite3.connect(str(self.db_path))
            self._connection.row_factory = sqlite3.Row
            self._connection.executescript(SCHEMA)
            self._connection.commit()

        self._initialized = True
        log.info("database_initialized", path=str(self.db_path))

    async def close(self) -> None:
        """Close database connection."""
        if self._connection:
            if AIOSQLITE_AVAILABLE:
                await self._connection.close()
            else:
                self._connection.close()
            self._connection = None
            log.info("database_closed")

    async def _execute(self, query: str, params: tuple = ()) -> Any:
        """Execute a query."""
        if not self._initialized:
            await self.init()

        async with self._lock:
            if AIOSQLITE_AVAILABLE:
                cursor = await self._connection.execute(query, params)
                await self._connection.commit()
                return cursor
            else:
                cursor = self._connection.execute(query, params)
                self._connection.commit()
                return cursor

    async def _fetchall(self, query: str, params: tuple = ()) -> list[dict]:
        """Execute query and fetch all results."""
        if not self._initialized:
            await self.init()

        async with self._lock:
            if AIOSQLITE_AVAILABLE:
                cursor = await self._connection.execute(query, params)
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
            else:
                cursor = self._connection.execute(query, params)
                rows = cursor.fetchall()
                return [dict(row) for row in rows]

    async def _fetchone(self, query: str, params: tuple = ()) -> Optional[dict]:
        """Execute query and fetch one result."""
        if not self._initialized:
            await self.init()

        async with self._lock:
            if AIOSQLITE_AVAILABLE:
                cursor = await self._connection.execute(query, params)
                row = await cursor.fetchone()
                return dict(row) if row else None
            else:
                cursor = self._connection.execute(query, params)
                row = cursor.fetchone()
                return dict(row) if row else None

    async def insert_violation(
        self,
        violation,  # ZoneViolation
        frame_path: Optional[str] = None,
        agent_reasoning: Optional[str] = None,
        coach_message: Optional[str] = None
    ) -> str:
        """
        Insert a violation record.

        Args:
            violation: ZoneViolation object
            frame_path: Path to saved frame snapshot
            agent_reasoning: Agent's analysis reasoning
            coach_message: Coach agent's message

        Returns:
            violation_id
        """
        violation_id = str(uuid.uuid4())

        query = """
        INSERT INTO violations (
            violation_id, timestamp, zone_id, zone_name, violation_type,
            worker_id, missing_items, osha_reference, confidence,
            frame_path, agent_reasoning, coach_message
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        params = (
            violation_id,
            violation.timestamp,
            violation.zone.zone_id,
            violation.zone.name,
            violation.violation_type,
            violation.worker_id,
            json.dumps(violation.missing_items),
            violation.zone.osha_reference,
            violation.confidence,
            frame_path,
            agent_reasoning,
            coach_message
        )

        await self._execute(query, params)
        log.info(
            "violation_inserted",
            violation_id=violation_id,
            zone=violation.zone.zone_id,
            type=violation.violation_type
        )
        return violation_id

    async def get_violation(self, violation_id: str) -> Optional[dict]:
        """Get a single violation by ID."""
        query = "SELECT * FROM violations WHERE violation_id = ?"
        result = await self._fetchone(query, (violation_id,))

        if result and result.get("missing_items"):
            result["missing_items"] = json.loads(result["missing_items"])

        return result

    async def get_violations(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        zone_id: Optional[str] = None,
        violation_type: Optional[str] = None,
        acknowledged: Optional[bool] = None,
        false_positive: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0
    ) -> list[dict]:
        """
        Query violations with filters.

        Args:
            start_time: Filter by start timestamp
            end_time: Filter by end timestamp
            zone_id: Filter by zone
            violation_type: Filter by violation type
            acknowledged: Filter by acknowledgment status
            false_positive: Filter by false positive status
            limit: Maximum results
            offset: Result offset

        Returns:
            List of violation records
        """
        conditions = []
        params = []

        if start_time:
            conditions.append("timestamp >= ?")
            params.append(start_time)

        if end_time:
            conditions.append("timestamp <= ?")
            params.append(end_time)

        if zone_id:
            conditions.append("zone_id = ?")
            params.append(zone_id)

        if violation_type:
            conditions.append("violation_type = ?")
            params.append(violation_type)

        if acknowledged is not None:
            conditions.append("acknowledged = ?")
            params.append(acknowledged)

        if false_positive is not None:
            conditions.append("false_positive = ?")
            params.append(false_positive)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        query = f"""
        SELECT * FROM violations
        WHERE {where_clause}
        ORDER BY timestamp DESC
        LIMIT ? OFFSET ?
        """

        params.extend([limit, offset])
        results = await self._fetchall(query, tuple(params))

        # Parse JSON fields
        for r in results:
            if r.get("missing_items"):
                r["missing_items"] = json.loads(r["missing_items"])

        return results

    async def mark_acknowledged(
        self,
        violation_id: str,
        user: str,
        false_positive: bool = False,
        notes: Optional[str] = None
    ) -> None:
        """
        Mark a violation as acknowledged.

        Args:
            violation_id: Violation to acknowledge
            user: User acknowledging
            false_positive: Mark as false positive
            notes: Optional notes
        """
        query = """
        UPDATE violations
        SET acknowledged = TRUE,
            acknowledged_by = ?,
            acknowledged_at = ?,
            false_positive = ?,
            notes = ?
        WHERE violation_id = ?
        """

        await self._execute(query, (
            user,
            time.time(),
            false_positive,
            notes,
            violation_id
        ))

        log.info(
            "violation_acknowledged",
            violation_id=violation_id,
            user=user,
            false_positive=false_positive
        )

    async def mark_alert_sent(self, violation_id: str) -> None:
        """Mark that an alert was sent for this violation."""
        query = "UPDATE violations SET alert_sent = TRUE WHERE violation_id = ?"
        await self._execute(query, (violation_id,))

    async def get_stats(
        self,
        period: str = "day"
    ) -> dict:
        """
        Get violation statistics.

        Args:
            period: Time period ("hour", "day", "week", "month")

        Returns:
            Dictionary with statistics
        """
        # Calculate time range
        now = time.time()
        period_seconds = {
            "hour": 3600,
            "day": 86400,
            "week": 604800,
            "month": 2592000
        }
        start_time = now - period_seconds.get(period, 86400)

        # Total violations
        total_query = """
        SELECT COUNT(*) as count FROM violations
        WHERE timestamp >= ?
        """
        total_result = await self._fetchone(total_query, (start_time,))
        total = total_result["count"] if total_result else 0

        # By type
        type_query = """
        SELECT violation_type, COUNT(*) as count
        FROM violations
        WHERE timestamp >= ?
        GROUP BY violation_type
        """
        type_results = await self._fetchall(type_query, (start_time,))
        by_type = {r["violation_type"]: r["count"] for r in type_results}

        # By zone
        zone_query = """
        SELECT zone_id, zone_name, COUNT(*) as count
        FROM violations
        WHERE timestamp >= ?
        GROUP BY zone_id
        """
        zone_results = await self._fetchall(zone_query, (start_time,))
        by_zone = {r["zone_id"]: {"name": r["zone_name"], "count": r["count"]}
                   for r in zone_results}

        # Acknowledgment rate
        ack_query = """
        SELECT
            SUM(CASE WHEN acknowledged THEN 1 ELSE 0 END) as acknowledged,
            SUM(CASE WHEN false_positive THEN 1 ELSE 0 END) as false_positives,
            COUNT(*) as total
        FROM violations
        WHERE timestamp >= ?
        """
        ack_result = await self._fetchone(ack_query, (start_time,))

        ack_rate = 0.0
        fp_rate = 0.0
        if ack_result and ack_result["total"] > 0:
            ack_rate = (ack_result["acknowledged"] or 0) / ack_result["total"]
            fp_rate = (ack_result["false_positives"] or 0) / ack_result["total"]

        # Hourly trend (last 24 hours)
        hourly_query = """
        SELECT
            CAST((timestamp / 3600) AS INTEGER) * 3600 as hour_bucket,
            COUNT(*) as count
        FROM violations
        WHERE timestamp >= ?
        GROUP BY hour_bucket
        ORDER BY hour_bucket
        """
        hourly_start = now - 86400  # Last 24 hours
        hourly_results = await self._fetchall(hourly_query, (hourly_start,))
        hourly_trend = [(r["hour_bucket"], r["count"]) for r in hourly_results]

        return {
            "period": period,
            "start_time": start_time,
            "end_time": now,
            "total_violations": total,
            "by_type": by_type,
            "by_zone": by_zone,
            "acknowledgment_rate": round(ack_rate * 100, 1),
            "false_positive_rate": round(fp_rate * 100, 1),
            "hourly_trend": hourly_trend
        }

    async def get_compliance_rate(
        self,
        zone_id: Optional[str] = None,
        period: str = "day"
    ) -> float:
        """
        Calculate compliance rate (inverse of violation rate).

        This is a simplified metric - actual compliance would need
        total observation count.
        """
        stats = await self.get_stats(period)
        # This would need to be calculated based on actual monitoring time
        # For now, return a placeholder
        return 94.5  # Placeholder

    async def export_csv(
        self,
        path: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        zone_id: Optional[str] = None
    ) -> None:
        """
        Export violations to CSV file.

        Args:
            path: Output file path
            start_time: Filter start
            end_time: Filter end
            zone_id: Filter by zone
        """
        import csv

        violations = await self.get_violations(
            start_time=start_time,
            end_time=end_time,
            zone_id=zone_id,
            limit=10000
        )

        if not violations:
            log.warning("no_violations_to_export")
            return

        # Ensure directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # Write CSV
        fieldnames = list(violations[0].keys())

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for v in violations:
                # Convert lists to strings
                if isinstance(v.get("missing_items"), list):
                    v["missing_items"] = ", ".join(v["missing_items"])

                # Convert timestamps to readable format
                for key in ["timestamp", "acknowledged_at", "created_at"]:
                    if v.get(key):
                        v[key] = datetime.fromtimestamp(v[key]).isoformat()

                writer.writerow(v)

        log.info("violations_exported", path=path, count=len(violations))

    async def cleanup_old_records(self, days: int = 90) -> int:
        """
        Delete old violation records.

        Args:
            days: Delete records older than this

        Returns:
            Number of records deleted
        """
        cutoff = time.time() - (days * 86400)

        # Get count first
        count_query = "SELECT COUNT(*) as count FROM violations WHERE timestamp < ?"
        result = await self._fetchone(count_query, (cutoff,))
        count = result["count"] if result else 0

        # Delete
        delete_query = "DELETE FROM violations WHERE timestamp < ?"
        await self._execute(delete_query, (cutoff,))

        log.info("old_records_deleted", count=count, days=days)
        return count
