#!/usr/bin/env bash
set -euo pipefail

cd /workspace
npm install

cd /workspace/crates/database
sqlx migrate run
