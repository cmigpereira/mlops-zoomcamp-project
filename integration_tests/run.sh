#! /usr/bin/env bash

if [[ -z "${GITHUB_ACTIONS}" ]]; then
  cd "$(dirname "$0")"
fi

docker-compose -f docker-compose-integration-test.yml up --build -d

sleep 5

python tests.py

ERROR_CODE=$?

if [ ${ERROR_CODE} != 0 ]; then
    docker-compose -f docker-compose-integration-test.yml logs
    docker-compose -f docker-compose-integration-test.yml down
    exit ${ERROR_CODE}
fi

docker-compose -f docker-compose-integration-test.yml down