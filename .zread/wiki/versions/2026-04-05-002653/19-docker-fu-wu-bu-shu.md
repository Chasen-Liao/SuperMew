本文档详细介绍 SuperMew 项目的 Docker 服务部署方案，涵盖 Milvus 向量库集群、PostgreSQL 会话存储以及相关依赖组件的启动、配置与运维操作。

## 服务架构概览

SuperMew 采用微服务架构部署核心依赖组件，所有服务通过 Docker Compose 统一管理。以下架构图展示了各服务间的依赖关系与通信方式：

```mermaid
graph TB
    subgraph "Docker Compose Network"
        subgraph "Milvus Cluster"
            E[etcd<br/>:2379]
            M[MinIO<br/>:9000]
            S[Milvus Standalone<br/>:19530 :9091]
            A[Attu UI<br/>:8080]
        end
        
        P[PostgreSQL<br/>:5432]
    end
    
    subgraph "Host Machine"
        B[Backend App<br/>:8000]
        F[Frontend<br/>Static Files]
        E2[External LLM API]
    end
    
    S --> E
    S --> M
    A --> S
    
    B --> S
    B --> P
    B --> E2
    F --> B
```

## 服务组件详解

### 1. Milvus 向量库集群

Milvus 是本项目的核心向量检索引擎，采用单机部署模式，包含以下四个服务组件：

| 组件 | 镜像版本 | 端口 | 用途 | 健康检查 |
|------|----------|------|------|----------|
| **etcd** | quay.io/coreos/etcd:v3.5.18 | 2379 | 元数据存储 | `etcdctl endpoint health` |
| **MinIO** | minio/minio:RELEASE.2024-05-28T17-19-04Z | 9000/9001 | 对象存储 | `curl http://localhost:9000/minio/health/live` |
| **standalone** | milvusdb/milvus:v2.5.14 | 19530/9091 | 向量数据库核心 | `curl http://localhost:9091/healthz` |
| **attu** | zilliz/attu:v2.5.11 | 8080 | Web 可视化界面 | — |

**etcd 配置参数说明**：
- `ETCD_AUTO_COMPACTION_MODE=revision`：按修订版本自动压缩
- `ETCD_AUTO_COMPACTION_RETENTION=1000`：保留最近 1000 个修订版本
- `ETCD_QUOTA_BACKEND_BYTES=4294967296`：后端存储限额 4GB
- `ETCD_SNAPSHOT_COUNT=50000`：每 50000 条操作执行一次快照

Sources: [docker-compose.yml](docker-compose.yml#L21-L36)

### 2. PostgreSQL 会话存储

PostgreSQL 用于存储 LangGraph Checkpointer 的会话状态，实现跨请求的对话上下文持久化：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| 数据库名 | `supermew` | Checkpointer 状态存储 |
| 用户名 | `supermew` | 连接认证 |
| 密码 | `supermew123` | 连接认证（**生产环境需修改**） |
| 端口 | `5433` | 映射到宿主机端口 |
| 数据卷 | `./volumes/postgres` | 持久化存储路径 |

Sources: [docker-compose.yml](docker-compose.yml#L4-L19)

## 快速启动

### 前置条件

```bash
# 检查 Docker 和 Docker Compose 版本
docker --version          # >= 20.10
docker compose version    # >= 2.0
```

### 启动所有服务

```bash
# 在项目根目录执行
docker compose up -d

# 或指定数据卷目录
DOCKER_VOLUME_DIRECTORY=/data/supermew docker compose up -d
```

### 验证服务状态

```bash
# 查看所有容器状态
docker compose ps

# 预期输出
NAME                    STATUS          PORTS
milvus-attu             running         0.0.0.0:8080->3000/tcp
milvus-etcd             running         2379/tcp
milvus-minio            running         0.0.0.0:9000-9001->9000-9001/tcp
milvus-standalone       running         0.0.0.0:9091->9091/tcp, 0.0.0.0:19530->19530/tcp
supermew-postgres       running         0.0.0.0:5433->5432/tcp
```

Sources: [README.md](README.md#L68-L81)

### 查看服务日志

```bash
# 查看 Milvus 核心服务日志
docker compose logs -f standalone

# 查看 etcd 日志（排查连接问题）
docker compose logs -f etcd

# 查看 PostgreSQL 日志
docker compose logs -f postgres
```

## 服务连接配置

### 环境变量配置

在项目根目录创建 `.env` 文件，根据实际部署环境调整以下配置：

```env
# ===== PostgreSQL 连接配置 =====
POSTGRES_HOST=127.0.0.1
POSTGRES_PORT=5433
POSTGRES_USER=supermew
POSTGRES_PASSWORD=supermew123
POSTGRES_DB=supermew

# ===== Milvus 连接配置 =====
MILVUS_HOST=127.0.0.1
MILVUS_PORT=19530
MILVUS_COLLECTION=embeddings_collection

# ===== 记忆向量库配置 =====
MEMORY_COLLECTION_NAME=user_memory
```

Sources: [.env.example](.env.example#L61-L65), [backend/config.py](backend/config.py#L45-L50)

### 服务端口映射

```mermaid
graph LR
    subgraph "Docker Network"
        M1[standalone<br/>19530]
        M2[attu<br/>3000]
        M3[minio<br/>9000]
        M4[postgres<br/>5432]
    end
    
    subgraph "Host Ports"
        H1["19530 → standalone:19530"]
        H2["8080 → attu:3000"]
        H3["9000 → minio:9000"]
        H4["9001 → minio:9001"]
        H5["5433 → postgres:5432"]
        H6["9091 → standalone:9091"]
    end
```

| 服务 | 容器端口 | 宿主机端口 | 访问地址 |
|------|----------|------------|----------|
| Milvus | 19530 | 19530 | `http://127.0.0.1:19530` |
| Milvus 健康检查 | 9091 | 9091 | `http://127.0.0.1:9091/healthz` |
| Attu 管理界面 | 3000 | 8080 | `http://127.0.0.1:8080` |
| MinIO API | 9000 | 9000 | `http://127.0.0.1:9000` |
| MinIO Console | 9001 | 9001 | `http://127.0.0.1:9001` |
| PostgreSQL | 5432 | 5433 | `postgresql://supermew:supermew123@127.0.0.1:5433/supermew` |

## 运维管理

### 服务重启

```bash
# 重启单个服务
docker compose restart standalone

# 重启所有服务
docker compose restart
```

### 服务停止

```bash
# 停止所有服务（保留数据卷）
docker compose stop

# 停止并删除容器（保留数据卷）
docker compose down

# 完全清理（包括数据卷）
docker compose down -v
```

### 数据卷管理

数据卷默认存储在 `./volumes/` 目录下，可通过环境变量自定义：

```bash
# 自定义数据卷目录
export DOCKER_VOLUME_DIRECTORY=/mnt/nas/supermew
docker compose up -d
```

目录结构：
```
volumes/
├── postgres/     # PostgreSQL 数据
├── etcd/         # Milvus 元数据
├── minio/        # Milvus 对象存储
└── milvus/       # Milvus 主数据
```

### 清理与重置

```bash
# 清理 Milvus 索引数据（保留数据卷定义）
docker compose down
rm -rf volumes/milvus/* volumes/etcd/* volumes/minio/*
docker compose up -d

# 完全重置（包括 PostgreSQL 数据）
docker compose down -v
docker compose up -d
```

## 常见问题排查

### 1. Milvus 连接失败

```bash
# 检查 Milvus 健康状态
curl http://127.0.0.1:9091/healthz
# 预期返回: {"status":"ok"}

# 检查 etcd 连接
docker compose exec etcd etcdctl endpoint health
# 预期返回: http://etcd:2379 is healthy

# 检查 MinIO 连接
curl http://127.0.0.1:9000/minio/health/live
```

### 2. PostgreSQL 连接问题

```bash
# 测试 PostgreSQL 连接
docker compose exec postgres pg_isready -U supermew

# 从容器内连接测试
docker compose exec -it postgres psql -U supermew -d supermew
```

### 3. 服务启动顺序问题

Docker Compose 会根据 `depends_on` 自动处理启动顺序。如需手动调整：

```bash
# 按依赖顺序启动
docker compose up -d etcd
docker compose up -d minio
docker compose up -d postgres
docker compose up -d standalone
docker compose up -d attu
```

## 生产环境部署建议

### 安全加固

1. **修改默认密码**：更新 PostgreSQL 和 MinIO 的默认凭据
2. **限制端口暴露**：仅暴露必要端口，或使用 Nginx 反向代理
3. **启用 TLS**：为所有服务配置 SSL/TLS 证书

### 性能优化

| 优化项 | 配置建议 |
|--------|----------|
| Milvus 内存 | 推荐 >= 16GB RAM |
| MinIO 存储 | 使用 SSD 以提升检索性能 |
| PostgreSQL | 配置 `shared_buffers` 和 `work_mem` |

### 监控告警

```bash
# 使用 Attu 监控 Milvus 状态
# 访问 http://127.0.0.1:8080 查看集合统计

# 设置健康检查告警（示例）
curl -f http://127.0.0.1:9091/healthz || alert "Milvus unhealthy"
```

## 下一步

完成 Docker 服务部署后，建议继续阅读：

- [API 接口文档](20-api-jie-kou-wen-dang) — 了解后端服务提供的 REST API
- [系统架构总览](5-xi-tong-jia-gou-zong-lan) — 深入理解各组件间的交互机制