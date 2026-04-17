pipeline {
    agent any

    environment {
        IMAGE_NAME = 'jacktherizzzzler/wine_predict_2022bcd0002'
        CURRENT_ACCURACY = ''
        SHOULD_DEPLOY = 'false'
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Setup Python Virtual Environment') {
            steps {
                sh '''
                    python3 -m venv .venv
                    . .venv/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements.txt
                '''
            }
        }

        stage('Train Model') {
            steps {
                sh '''
                    . .venv/bin/activate
                    python scripts/train.py
                    test -f app/artifacts/trained_model.pkl
                    test -f app/artifacts/metrics.json
                '''
            }
        }

        stage('Read Accuracy') {
            steps {
                script {
                    env.CURRENT_ACCURACY = sh(
                        script: "jq -r '.accuracy' app/artifacts/metrics.json",
                        returnStdout: true
                    ).trim()
                    echo "Current accuracy: ${env.CURRENT_ACCURACY}"
                }
            }
        }

        stage('Compare Accuracy') {
            steps {
                withCredentials([string(credentialsId: 'best-accuracy', variable: 'BEST_ACCURACY')]) {
                    script {
                        def decision = sh(
                            script: """
                                python3 - <<'PY'
                                current = float('${env.CURRENT_ACCURACY}')
                                baseline = float('${BEST_ACCURACY}')
                                print('true' if current > baseline else 'false')
                                PY
                            """,
                            returnStdout: true
                        ).trim()
                        env.SHOULD_DEPLOY = decision
                        echo "Baseline accuracy: ${BEST_ACCURACY}"
                        echo "Deploy decision: ${env.SHOULD_DEPLOY}"
                    }
                }
            }
        }

        stage('Build Docker Image') {
            when {
                expression { env.SHOULD_DEPLOY == 'true' }
            }
            steps {
                withCredentials([usernamePassword(credentialsId: 'dockerhub-creds', usernameVariable: 'DOCKER_USER', passwordVariable: 'DOCKER_PASS')]) {
                    sh '''
                        echo "$DOCKER_PASS" | docker login -u "$DOCKER_USER" --password-stdin
                        docker build -t "$IMAGE_NAME:${BUILD_NUMBER}" -t "$IMAGE_NAME:latest" .
                    '''
                }
            }
        }

        stage('Push Docker Image') {
            when {
                expression { env.SHOULD_DEPLOY == 'true' }
            }
            steps {
                withCredentials([usernamePassword(credentialsId: 'dockerhub-creds', usernameVariable: 'DOCKER_USER', passwordVariable: 'DOCKER_PASS')]) {
                    sh '''
                        echo "$DOCKER_PASS" | docker login -u "$DOCKER_USER" --password-stdin
                        docker push "$IMAGE_NAME:${BUILD_NUMBER}"
                        docker push "$IMAGE_NAME:latest"
                    '''
                }
            }
        }
    }

    post {
        always {
            archiveArtifacts artifacts: 'app/artifacts/**', fingerprint: true
        }
    }
}
