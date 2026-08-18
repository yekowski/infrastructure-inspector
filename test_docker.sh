#!/usr/bin/env bash
set -e

IMAGE_NAME="vision-pipeline-test"
CONTAINER_NAME="vision-test-container"

echo "=================================================="
echo "Starting Containerized Pipeline Verification"
echo "=================================================="

# Define cleanup function to ensure machine state remains clean on exit
cleanup() {
  echo "=================================================="
  echo "Cleaning Up Local Container and Image Artifacts..."
  echo "=================================================="
  
  if docker ps -a --format '{{.Names}}' | grep -Eq "^${CONTAINER_NAME}\$"; then
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1
  fi
  
  if docker images --format '{{.Repository}}' | grep -Eq "^${IMAGE_NAME}\$"; then
    docker rmi -f "$IMAGE_NAME" >/dev/null 2>&1
  fi
}

# Trap exits to guarantee cleanup runs even on unexpected script termination
trap cleanup EXIT

echo " -> Building Docker image: ${IMAGE_NAME}..."
docker build -t "$IMAGE_NAME" .

echo " -> Running test suite inside container: ${CONTAINER_NAME}..."
# Run the test suite by overriding the image entrypoint
docker run --name "$CONTAINER_NAME" \
  --entrypoint "python3" \
  "$IMAGE_NAME" \
  run_evals.py

TEST_STATUS=$?

if [ $TEST_STATUS -eq 0 ]; then
  echo "=================================================="
  echo "✔ SUCCESS: All containerized tests passed successfully!"
  echo "=================================================="
else
  echo "=================================================="
  echo "✘ FAILURE: Containerized test suite failed (exit code: ${TEST_STATUS})."
  echo "=================================================="
  exit $TEST_STATUS
fi
