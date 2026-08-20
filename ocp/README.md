# OCP 배포 가이드 (Bastion 기준)

## 전체 흐름

```
[Bastion]
  1. git clone
  2. .env 없이 Secret YAML 수정
  3. docker build (앱 + 인제스트)
  4. OCP internal registry에 push
  5. oc apply (순서대로)
  6. Job으로 PDF 인제스트 (최초 1회)
```

## 사전 확인

```bash
# bastion에서 아래 도구가 모두 있어야 함
docker --version   # 또는 podman --version
oc version
oc whoami          # 로그인 상태 확인
```

---

## Step 1. 레포 클론

```bash
git clone https://github.com/papooo-dev/instana-chatbot.git
cd instana-chatbot
```

> **중요**: `.gitignore`에 `app.py`와 `uv.lock`은 제외되어 있지 않으므로
> clone 후 모든 파일이 정상적으로 존재해야 합니다.
> 아래 명령으로 확인:
> ```bash
> ls app.py uv.lock core/ utils/ data/instana-logo.png
> ```

---

## Step 2. PDF 문서 준비

`data/*.pdf`는 `.gitignore`로 제외되어 있어 clone 후 없습니다. 직접 다운로드해야 합니다.

```bash
curl -o data/instana-observability-1.0.312-documentation.pdf \
  "https://www.ibm.com/docs/en/SSE1JP5_1.0.312/pdf/instana-observability-1.0.312-documentation.pdf"
```

> 인터넷이 막혀 있다면, 로컬에서 scp로 복사:
> ```bash
> scp instana-observability-1.0.312-documentation.pdf \
>   itzuser@<BASTION-IP>:~/instana-chatbot/data/
> ```

---

## Step 3. Secret 값 설정

```bash
cp ocp/1-secret.yaml.example ocp/1-secret.yaml  # 없으면 직접 편집
vi ocp/1-secret.yaml
```

아래 항목을 실제 값으로 채웁니다:

```yaml
stringData:
  WATSONX_APIKEY: "실제_API_키"
  WATSONX_PROJECT_ID: "실제_프로젝트_ID"
  TRACELOOP_BASE_URL: "https://실제-instana-백엔드:4317"
  QR_TEXT: "https://실제-설문-URL"
```

> `ocp/1-secret.yaml`은 `.gitignore`에 등록되어 있어 커밋되지 않습니다.

---

## Step 4. 네임스페이스 생성 및 OCP 레지스트리 로그인

```bash
oc new-project askstan

# OCP internal registry 외부 접근 주소 확인
oc get route default-route -n openshift-image-registry \
  --template='{{ .spec.host }}'
# 예시 출력: default-route-openshift-image-registry.apps.cluster.example.com
```

```bash
# 위에서 확인한 주소로 로그인
REGISTRY=$(oc get route default-route -n openshift-image-registry \
  --template='{{ .spec.host }}')

docker login -u $(oc whoami) -p $(oc whoami -t) ${REGISTRY}
```

---

## Step 5. 이미지 빌드 & 푸시

```bash
REGISTRY=$(oc get route default-route -n openshift-image-registry \
  --template='{{ .spec.host }}')

# 앱 이미지
docker build -t ${REGISTRY}/askstan/askstan:latest .
docker push ${REGISTRY}/askstan/askstan:latest

# 인제스트 이미지
docker build -f Dockerfile.ingest \
  -t ${REGISTRY}/askstan/askstan-ingest:latest .
docker push ${REGISTRY}/askstan/askstan-ingest:latest
```

> **인터넷이 차단된 bastion이라면:**
> `Dockerfile`과 `Dockerfile.ingest`에서 `ghcr.io` 라인을 주석 처리하고
> `pip install uv` 라인을 활성화하세요.
> ```bash
> # Dockerfile / Dockerfile.ingest 상단 수정
> # COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv  ← 주석
> RUN pip install uv --no-cache-dir                                  ← 활성화
> ```

---

## Step 6. 매니페스트의 이미지 주소 업데이트

빌드한 이미지 주소를 매니페스트에 반영합니다:

```bash
REGISTRY=$(oc get route default-route -n openshift-image-registry \
  --template='{{ .spec.host }}')

# 앱 배포 YAML 이미지 주소 교체
sed -i "s|image-registry.openshift-image-registry.svc:5000/askstan/askstan:latest|${REGISTRY}/askstan/askstan:latest|g" \
  ocp/4-app-deployment.yaml

# 인제스트 Job YAML 이미지 주소 교체
sed -i "s|image-registry.openshift-image-registry.svc:5000/askstan/askstan-ingest:latest|${REGISTRY}/askstan/askstan-ingest:latest|g" \
  ocp/5-ingest-job.yaml
```

---

## Step 7. OCP 배포 (순서 중요)

```bash
# 1. Secret
oc apply -f ocp/1-secret.yaml

# 2. Milvus PVC (스토리지)
oc apply -f ocp/2-milvus-pvc.yaml

# 3. Milvus 서비스 기동
oc apply -f ocp/3-milvus-deployment.yaml

# Milvus 완전히 뜰 때까지 대기 (약 2~3분)
oc rollout status deployment/milvus-etcd -n askstan
oc rollout status deployment/milvus-minio -n askstan
oc rollout status deployment/milvus-standalone -n askstan

# 4. 앱 배포
oc apply -f ocp/4-app-deployment.yaml

# 5. PDF 인제스트 Job (최초 1회)
oc apply -f ocp/5-ingest-job.yaml
```

---

## Step 8. 배포 확인

```bash
# Pod 상태
oc get pods -n askstan

# 인제스트 Job 로그 (완료까지 수 분 소요)
oc logs -f job/milvus-ingest -n askstan

# 앱 접속 URL 확인
oc get route askstan -n askstan
```

출력된 HOST 주소로 브라우저 접속 (HTTPS 자동 적용)

---

## 운영

### 앱 로그 확인
```bash
oc logs -f deployment/askstan -n askstan
```

### 이미지 재빌드 & 재배포
```bash
docker build -t ${REGISTRY}/askstan/askstan:latest . && \
docker push ${REGISTRY}/askstan/askstan:latest && \
oc rollout restart deployment/askstan -n askstan
```

### 인제스트 Job 재실행 (문서 업데이트 시)
```bash
oc delete job milvus-ingest -n askstan
oc apply -f ocp/5-ingest-job.yaml
```

---

## 파일 구조

```
instana-chatbot/
├── Dockerfile                         # 앱 이미지
├── Dockerfile.ingest                  # PDF 인제스트 Job 이미지
├── .dockerignore                      # 빌드 컨텍스트 제외 목록
├── .streamlit/config.toml             # Streamlit headless 설정
└── ocp/
    ├── README.md                      # 이 파일
    ├── 1-secret.yaml                  # ⚠️ 실제 값 채울 것 (gitignore됨)
    ├── 2-milvus-pvc.yaml              # Milvus용 PVC 3개
    ├── 3-milvus-deployment.yaml       # etcd + MinIO + Milvus standalone
    ├── 4-app-deployment.yaml          # AskStan Pod + Service + Route
    └── 5-ingest-job.yaml              # PDF → Milvus 벡터화 Job (최초 1회)
```
