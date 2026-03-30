#!/usr/bin/env bash
set -euo pipefail

# Prefer the dump baked into the devcontainer image; fall back to the workspace bind mount.
dump_path="${DATABASE_DUMP_PATH:-/opt/devcontainer-database/database_dump.dump}"
if [[ ! -f "${dump_path}" ]]; then
    dump_path="/workspace/data/database_dump.dump"
fi
if [[ ! -f "${dump_path}" ]]; then
    echo "error: no database dump found (expected /opt/devcontainer-database/database_dump.dump or /workspace/data/database_dump.dump)" >&2
    exit 1
fi
if [[ ! -s "${dump_path}" ]]; then
    echo "error: database dump is empty. Run: cargo make database-dump" >&2
    exit 1
fi

: "${DATABASE_URL:?DATABASE_URL is required}"

echo "Restoring from ${dump_path} (pg_restore custom format)"
pg_restore --dbname="${DATABASE_URL}" --no-owner --no-acl --exit-on-error --verbose "${dump_path}"
echo "Restore finished."
