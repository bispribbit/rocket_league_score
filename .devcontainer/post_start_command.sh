#!/usr/bin/env bash
set -euo pipefail

cd /workspace/crates/is_this_a_smurf
npm install

cd /workspace/crates/database
sqlx migrate run
