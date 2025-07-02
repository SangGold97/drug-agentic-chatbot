#!/bin/bash

# Script variables
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
DOCKER_COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.yml"
VOLUMES_DIR="${SCRIPT_DIR}/volumes"

# Check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        echo "Docker is not running. Please start Docker first."
        exit 1
    fi
    echo "Docker is running."
}

# Start Milvus and PostgreSQL databases
start() {
    echo "Starting Milvus and PostgreSQL databases..."
    check_docker
    docker-compose -f "${DOCKER_COMPOSE_FILE}" up -d
    echo "Databases started."
}

# Stop Milvus and PostgreSQL databases
stop() {
    echo "Stopping Milvus and PostgreSQL databases..."
    docker-compose -f "${DOCKER_COMPOSE_FILE}" down
    echo "Databases stopped."
}

# Check status of running containers
status() {
    echo "Checking status of databases..."
    docker-compose -f "${DOCKER_COMPOSE_FILE}" ps -a
    echo
    echo "Health status:"
    docker ps --format "table {{.Names}}\t{{.Status}}" --filter "name=milvus"
    docker ps --format "table {{.Names}}\t{{.Status}}" --filter "name=postgres"
}

# Restart databases
restart() {
    echo "Restarting databases..."
    stop
    sleep 2
    start
    status
}

# Main script logic
case "$1" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    status)
        status
        ;;
    restart)
        restart
        ;;
    *)
        echo "Usage: $0 {start|stop|status|restart}"
        exit 1
        ;;
esac
