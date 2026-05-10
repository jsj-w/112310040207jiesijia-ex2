[README.md](https://github.com/user-attachments/files/27567922/README.md)
# CNN手写数字识别

基于卷积神经网络（CNN）的 MNIST 手写数字识别完整实验项目，包含模型训练、Web部署和交互式手写识别系统。

## 学生信息

- **姓名**：揭思佳
- **学号**：112310040207

## 项目结构

```
project/
├── 实验一.py              # 模型训练与超参数调优
├── 实验二.py              # 模型封装与Web部署
├── 实验三.py              # 交互式手写识别系统
├── 实验一输出/            # 训练输出目录
│   ├── mnist_cnn_final.pth    # 训练好的模型权重
│   ├── loss曲线.png           # Loss曲线图
│   ├── 对比实验结果.csv       # 对比实验结果
│   └── submission.csv         # Kaggle提交文件
├── requirements.txt       # 依赖列表
└── CNN手写数字识别实验模板.md  # 实验报告模板
```

## 环境要求

- Python 3.8+
- PyTorch
- torchvision
- matplotlib
- Gradio
- Pillow

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 实验一：模型训练与超参数调优

```bash
python 实验一.py
```

运行四组对比实验并训练最终模型。

### 实验二：Web部署

```bash
python 实验二.py
```

启动Gradio Web应用，上传手写数字图片进行预测。

### 实验三：交互式手写识别

```bash
python 实验三.py
```

启动带有手写画板的交互式识别系统。

## 实验内容

| 阶段 | 内容 | 要求 |
|------|------|------|
| 实验一 | 模型训练与超参数调优 | 必做 |
| 实验二 | 模型封装与Web部署 | 必做 |
| 实验三 | 交互式手写识别系统 | 选做（加分） |

## 模型性能

最终模型在测试集上达到 **0.9886** 的准确率。
