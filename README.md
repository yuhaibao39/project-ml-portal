# Multi-Application Next.js Portal

一个统一的 Next.js 门户，托管两个独立的应用，分别使用不同的后端技术栈（Python FastAPI 和 Java Spring Boot），均集成同一个 ML 回归模型（加利福尼亚房价预测）。

---

## 项目架构

```
d:\work-project-plan\
├── portal/                  # Next.js 16 门户 (App Router + Tailwind CSS v4)
│   ├── src/app/
│   │   ├── layout.tsx       # 根布局
│   │   ├── page.tsx         # 首页
│   │   └── *.tsx            # 各页面组件
│   └── Dockerfile           # 多阶段构建
│
├── backend-python/          # App 1: Property Value Estimator
│   ├── app/
│   │   ├── main.py          # FastAPI 入口 (port 8000)
│   │   ├── routes/          # 路由 (health, predict)
│   │   ├── models/          # Pydantic 数据模型
│   │   └── services/        # ML 模型服务
│   ├── requirements.txt
│   └── Dockerfile
│
├── backend-java/            # App 2: Property Market Analysis
│   ├── pom.xml              # Maven, Spring Boot 3.4.4, Java 21
│   ├── src/main/java/.../
│   │   ├── config/          # CORS, Cache, RestTemplate 配置
│   │   └── PropertyMarketAnalysisApplication.java
│   ├── src/main/resources/application.yml
│   └── Dockerfile
│
└── ml-service/              # ML 模型微服务 (Flask, port 8001)
    ├── model.py             # 模型训练 & 预测 (RandomForest)
    ├── requirements.txt
    └── Dockerfile
```

### 端口分配

| 服务 | 端口 |
|------|------|
| Next.js Portal | `3000` |
| Python FastAPI (Estimator) | `8000` |
| ML Model Service | `8001` |
| Java Spring Boot (Market Analysis) | `8080` |

---

## 环境要求

| 组件 | 版本要求 | 验证命令 |
|------|---------|---------|
| Node.js | >= 20 | `node --version` |
| npm | >= 10 | `npm --version` |
| Python | >= 3.12 | `python --version` |
| Java / JDK | >= 21 | `java --version` |
| Maven | >= 3.9 | `mvn --version` |
| Docker (可选) | >= 24 | `docker --version` |

---

## 安装步骤

### 方式一：本地开发（无 Docker）

#### 1. ML 模型服务

```bash
cd d:\work-project-plan\ml-service
pip install -r requirements.txt
python -c "from model import load_model; load_model(); print('Model trained OK')"
```

验证训练成功（输出 `Model trained OK`）。

#### 2. Python FastAPI 后端

```bash
cd d:\work-project-plan\backend-python
pip install -r requirements.txt
```

#### 3. Java Spring Boot 后端

```bash
cd d:\work-project-plan\backend-java
mvn clean package -DskipTests
```

#### 4. Next.js 门户

```bash
cd d:\work-project-plan\portal
npm install
```

### 方式二：Docker 一键部署

```bash
cd d:\work-project-plan
docker compose up --build
```

首次构建会下载镜像和依赖，耗时约 5-10 分钟。之后可加上 `-d` 参数后台运行：

```bash
docker compose up -d
```

---

## 启动指南

### 本地开发（分别启动各服务）

需要按依赖顺序启动：

#### Step 1: 启动 ML 模型服务 (端口 8001)

```bash
cd d:\work-project-plan\ml-service

# Flask 开发模式
python -c "
from model import load_model
load_model()
from flask import Flask
app = Flask(__name__)
# 手动启动模型后再启动 Flask

# 生产模式推荐
gunicorn --bind 0.0.0.0:8001 model:app
"
```

或写成一行：

```bash
cd d:\work-project-plan\ml-service && pip install -r requirements.txt && gunicorn --bind 0.0.0.0:8001 model:app
```

预期输出：
```
[INFO] Starting gunicorn ...
[INFO] Listening at: http://0.0.0.0:8001
[INFO] Model loaded ...
```

#### Step 2: 启动 Python FastAPI 后端 (端口 8000)

```bash
cd d:\work-project-plan\backend-python
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

预期输出：
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Model loaded (R²=0.XXXX)
```

打开浏览器访问 http://localhost:8000/docs 查看 Swagger API 文档。

#### Step 3: 启动 Java Spring Boot 后端 (端口 8080)

```bash
cd d:\work-project-plan\backend-java
mvn spring-boot:run
```

预期输出：
```
Tomcat started on port 8080
Started PropertyMarketAnalysisApplication in X seconds
```

#### Step 4: 启动 Next.js 门户 (端口 3000)

```bash
cd d:\work-project-plan\portal
npm run dev
```

预期输出：
```
▲ Next.js 16.2.9
- Local: http://localhost:3000
```

---

## 测试指南

### API 健康检查

启动后，验证各服务是否正常工作：

```bash
# Python FastAPI 健康检查
curl http://localhost:8000/health
# 预期: {"status":"healthy","model_loaded":true,"model_version":"1.0.0","uptime_seconds":X.XX}

# ML 模型信息
curl http://localhost:8000/api/v1/model/info
# 预期: {"model_type":"RandomForestRegressor","features":["MedInc",...],"r2_score":0.XXXX,...}

# Java Spring Boot 健康检查
curl http://localhost:8080/api/v2/health
# 预期: {"status":"UP"}

# ML Service (Flask) 健康检查
curl http://localhost:8001/health
# 预期: {"status":"healthy"}
```

### 功能测试

#### App 1: Property Value Estimator (Python)

**单一预测：**

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "MedInc": 3.5,
    "HouseAge": 25.0,
    "AveRooms": 6.0,
    "AveBedrms": 1.5,
    "Population": 1200.0,
    "AveOccup": 3.0,
    "Latitude": 34.5,
    "Longitude": -118.5
  }'
```

预期输出：
```json
{
  "predicted_value": 2.3584,
  "confidence_interval": [1.8675, 2.8493],
  "features_used": {
    "MedInc": 0.5123,
    "Latitude": 0.1078,
    "Longitude": 0.0982,
    ...
  }
}
```

**批量预测：**

```bash
curl -X POST http://localhost:8000/api/v1/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"properties": [
    {"MedInc": 3.5, "HouseAge": 25.0, "AveRooms": 6.0, "AveBedrms": 1.5, "Population": 1200.0, "AveOccup": 3.0, "Latitude": 34.5, "Longitude": -118.5},
    {"MedInc": 8.0, "HouseAge": 10.0, "AveRooms": 8.0, "AveBedrms": 1.2, "Population": 800.0, "AveOccup": 2.5, "Latitude": 37.5, "Longitude": -122.0}
  ]}'
```

**获取预测历史：**

```bash
curl http://localhost:8000/api/v1/predict/history
```

#### App 2: Property Market Analysis (Java)

**获取属性列表：**

```bash
curl "http://localhost:8080/api/v2/properties?region=LA&sortBy=price&sortOrder=desc"
```

**市场统计：**

```bash
curl "http://localhost:8080/api/v2/statistics?minPrice=200000&maxPrice=1000000"
```

**趋势数据：**

```bash
curl http://localhost:8080/api/v2/trends
```

**区域数据：**

```bash
curl http://localhost:8080/api/v2/regions
```

**比较属性：**

```bash
curl -X POST http://localhost:8080/api/v2/compare \
  -H "Content-Type: application/json" \
  -d '{
    "properties": [...],
    "includeForecast": true
  }'
```

**What-If 分析：**

```bash
curl -X POST http://localhost:8080/api/v2/what-if \
  -H "Content-Type: application/json" \
  -d '{
    "baseFeatures": {...},
    "whatIfChanges": {"MedInc": 5.0}
  }'
```

**导出数据：**

```bash
curl -o properties.csv "http://localhost:8080/api/v2/export?format=csv"
curl -o report.pdf "http://localhost:8080/api/v2/export?format=pdf"
```

**清除缓存：**

```bash
curl -X DELETE http://localhost:8080/api/v2/cache
```

### 浏览器验证

启动所有服务后，打开：

| URL | 功能 |
|-----|------|
| http://localhost:3000 | 门户首页 |
| http://localhost:3000/estimator | 房产估价工具 |
| http://localhost:3000/market-analysis | 市场分析仪表板 |
| http://localhost:8000/docs | Python API Swagger 文档 |
| http://localhost:8000/redoc | Python API ReDoc 文档 |
| http://localhost:8080/actuator/health | Java API Actuator 健康检查 |

---

## Docker 部署

### 使用 Docker Compose（推荐）

```bash
# 构建并启动所有服务
cd d:\work-project-plan
docker compose up --build

# 后台启动
docker compose up -d

# 查看日志
docker compose logs -f

# 停止所有服务
docker compose down

# 停止并删除数据卷
docker compose down -v
```

### 单独构建各镜像

```bash
# ML 服务
docker build -t ml-service d:\work-project-plan\ml-service
docker run -d -p 8001:8001 --name ml-service ml-service

# Python 后端
docker build -t backend-python d:\work-project-plan\backend-python
docker run -d -p 8000:8000 --name backend-python --link ml-service backend-python

# Java 后端（首次需下载 Maven 依赖，较慢）
docker build -t backend-java d:\work-project-plan\backend-java
docker run -d -p 8080:8080 --name backend-java --link ml-service backend-java

# Next.js 门户
docker build -t portal d:\work-project-plan\portal
docker run -d -p 3000:3000 --name portal --link backend-python --link backend-java portal
```

---

## 开发指南

### 常见问题排查

| 问题 | 原因 | 解决 |
|------|------|------|
| Portal 无法连接 Python API | Python 未启动 | 先启动 `uvicorn app.main:app` |
| Python API 启动慢 | 首次需训练 ML 模型 | 等待约 30 秒让 RandomForest 训练完成 |
| Java 构建失败 | Maven 依赖未下载 | 运行 `mvn dependency:resolve` |
| Portal build 报错 | 代码类型错误 | `npm run lint` 检查 |
| Docker 启动顺序问题 | 服务间依赖 | 使用 `docker compose up` 自动处理 |

### 各服务 API 文档

- **Python FastAPI**: http://localhost:8000/docs (Swagger) / http://localhost:8000/redoc (ReDoc)
- **Java Spring Boot**: 查看 `MarketController.java` 中的端点定义
- **ML Service**: 查看 `model.py` 中的路由定义 (`/predict`, `/predict/batch`, `/health`, `/info`)

### 代码结构变更

如需要扩展功能：

1. **添加新 API 端点**: 在对应后端的 controller/route 文件中添加
2. **添加新页面**: 在 `portal/src/app/` 下创建新路由目录
3. **修改 ML 模型**: 编辑 `ml-service/model.py` 或 `backend-python/app/services/model_service.py`
4. **共享 UI 组件**: 添加到 `portal/src/components/ui/`

---

## 技术栈总结

| 层 | 技术 | 版本 |
|------|------|------|
| 门户框架 | Next.js (App Router) | 16.2.9 |
| UI 框架 | React | 19.2.4 |
| 样式 | Tailwind CSS | v4 |
| 状态管理 | Zustand | 5.0 |
| 图表 | Recharts | 3.9 |
| Python API | FastAPI | 0.115 |
| Java API | Spring Boot | 3.4.4 |
| Java 版本 | JDK | 21 / 25 |
| ML 模型 | scikit-learn (RandomForest) | 1.6.1 |
| ML API | Flask | 3.1 |
