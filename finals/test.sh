TEAM_NAME="2b3y"
TEAM_TRACK="advanced"
if [[ $TEAM_NAME =~ ^[0-9] ]]; then
    # Starts with a number
    repo_name=repo-til-25-${TEAM_NAME}
else
    # Starts with a letter
    repo_name=${TEAM_NAME}-repo-til-25
fi
ar_ref=asia-southeast1-docker.pkg.dev/til-ai-2025/${repo_name}/${TEAM_NAME}-server:finals
# Builds finals repo and pushes to Artifact Registry
docker build -t $ar_ref finals && docker push $ar_ref
# Run all containers from docker-compose.yml
REPO_NAME=$repo_name docker compose up --build --force-recreate --abort-on-container-exit