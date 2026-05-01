#!/usr/bin/env bash
# Helper script to sync code to Explorer and retrieve results.

REMOTE_HOST="explorer"
REMOTE_DIR="/scratch/yirga.t/sail_vs_spark_benchmark"

push_code() {
    echo "Preparing remote directories on $REMOTE_HOST..."
    ssh $REMOTE_HOST "mkdir -p $REMOTE_DIR/logs"
    
    echo "Pushing code to $REMOTE_HOST..."
    # --filter=':- .gitignore' tells rsync to respect your .gitignore patterns
    # We also explicitly exclude .git to avoid transferring history
    rsync -avz --exclude '.git' --exclude '.gitignore' \
          --filter=':- .gitignore' \
          ./ $REMOTE_HOST:$REMOTE_DIR/
}

pull_results() {
    echo "Retrieving results and logs from $REMOTE_HOST..."
    mkdir -p results_explorer logs_explorer
    rsync -avz $REMOTE_HOST:$REMOTE_DIR/results/ results_explorer/
    rsync -avz $REMOTE_HOST:$REMOTE_DIR/logs/ logs_explorer/
}

usage() {
    echo "Usage: $0 {push|pull}"
    exit 1
}

case "${1:-}" in
    push) push_code ;;
    pull) pull_results ;;
    *) usage ;;
esac
