#!/usr/bin/env bash
set -euo pipefail

cd /workspace
mkdir -p data

: "${DATABASE_URL:?DATABASE_URL is required}"

echo "Writing data/database_dump.dump (pg_dump custom binary format, -Fc)"
pg_dump "${DATABASE_URL}" --no-owner --no-acl --format=custom --file=data/database_dump.dump
echo "Done. Rebuild the devcontainer image to bake this file into /opt/devcontainer-database/database_dump.dump"
