# 基于DeepSpeedChat进行实验

## 背景

DeepSpeedChat提供了SFT、RLHF等LLM预训练的完整流程，且代码简洁，非常适合用于开展LLM的研究工作。本文档将介绍如何使用DeepSpeedChat在
翻译团队的机器资源上进行实验。


## 数据格式准备

数据需要准备成jsonline的格式，每一条样本需要包含的字段如下：

- input: 输入的文本，可以是单个句子，也可以是多个句子组成的list
- target: （可选）希望LLM生成的文本
- prompt：（可选）提示模板

对于不同的组合，训练数据会按照如下方式组合：

```python
# item是一条样本
if "prompt" in item:
    sample = item['prompt'].format(*item['input'])
else:
    sample = item['input'][0]

if "target" in item:
    sample = sample + item['target']
```

## 启动训练

- 本地训练：参见`examples/train_local.sh`
- PAI Studio训练：参见`examples/train_pai.sh`

## TODOList

- [ ] PAI Studio上的log写入问题（OSS通过SmartCache看起来不能增量写入，待排查）
- [ ] 继续训练的相关代码建设
- [ ] 训练时评估的相关代码建设