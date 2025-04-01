# 贝叶斯优化应用部署指南

## 系统要求
- Docker Engine 20.10+
- docker-compose 1.29+

## 部署步骤

1. 导入Docker镜像
```bash
docker load -i bayesian-optimizer.tar
```

2. 启动服务
```bash
docker-compose up -d
```

3. 访问应用
打开浏览器访问: http://localhost:8501

## 数据准备
- 将数据文件放入项目目录下的`data/`文件夹
- 支持的文件格式请参考原始文档

## 常见问题
Q: 端口冲突怎么办？
A: 修改docker-compose.yml中的ports配置，例如改为"8502:8501"

Q: 如何查看日志？
```bash
docker-compose logs -f
```

## 联系方式
如有问题请联系项目维护人员
