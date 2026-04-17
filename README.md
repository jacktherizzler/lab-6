# Lab 6: Jenkins Pipeline Equivalent to GitHub Actions

This repository recreates the Lab 4 CI/CD automation in Jenkins.

## Repository Contents

- `scripts/train.py`: trains the wine quality model and writes artifacts to `app/artifacts/`
- `app/main.py`: FastAPI inference service
- `Dockerfile`: container image for inference
- `requirements.txt`: Python dependencies
- `Jenkinsfile`: Jenkins pipeline definition

## Required Jenkins Credentials

Create these in `Manage Jenkins -> Credentials -> System -> Global credentials`:

1. `dockerhub-creds`
   Type: Username with password
   Purpose: Docker Hub login for image build and push
2. `git-creds`
   Type: Username with password
   Purpose: GitHub repository access from Jenkins SCM configuration
3. `best-accuracy`
   Type: Secret text
   Purpose: baseline comparison value

Suggested initial value for `best-accuracy`: `-999999`

## Jenkins Pipeline Job Setup

1. Create a new Pipeline job named `2022BCD0002`
2. Select `Pipeline script from SCM`
3. Choose `Git`
4. Repository URL: `https://github.com/jacktherizzler/lab-6.git`
5. Credentials: `git-creds`
6. Script Path: `Jenkinsfile`

## Pipeline Stages

The `Jenkinsfile` runs these stages in order:

1. `Checkout`
2. `Setup Python Virtual Environment`
3. `Train Model`
4. `Read Accuracy`
5. `Compare Accuracy`
6. `Build Docker Image`
7. `Push Docker Image`

Artifacts are always archived from:

- `app/artifacts/**`

## Metric Notes

This lab continues the regression workflow from earlier labs, so `metrics.json`
stores:

- `mse`
- `r2_score`
- `accuracy`

The `accuracy` field is set to the model's `r2_score` so Jenkins can compare a
higher-is-better value using the `best-accuracy` credential.
