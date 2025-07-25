# import time
# from transformers import pipeline

# # 选择一个支持多语言且较快的模型，mDeBERTa-v3-base-mnli-xnli 是不错选择
# classifier = pipeline(
#     "zero-shot-classification",
#     model="./multilingual-MiniLMv2-L6-mnli-xnli",
#     device=-1  # CPU模式，-1 表示CPU，0是GPU
# )

# text = "对。"
# candidate_labels = ["提问", "陈述", "命令", "无效"]

# begin_time = time.time()
# result = classifier(text, candidate_labels)
# print(f"{(time.time()-begin_time) * 1000}ms")
# print(result)


from transformers import pipeline

classifier = pipeline(
    "zero-shot-classification",
    model="./multilingual-MiniLMv2-L6-mnli-xnli",
    device=-1  # CPU 模式
)

def is_valid_sentence(text, threshold=0.6):
    labels = ["有效", "无效"]
    result = classifier(text, labels)
    # result 结构示例:
    # {'sequence': '...', 'labels': ['有效', '无效'], 'scores': [0.75, 0.25]}
    
    # 找“有效”的概率
    scores = dict(zip(result['labels'], result['scores']))
    valid_score = scores.get("有效", 0.0)
    
    # 如果有效概率高于阈值，则认为有效
    return valid_score >= threshold, valid_score

# 测试示例
texts = [
    "恩来释放连接数量"
]

labels = ["提问", "非提问", "无意义"]
result = classifier("恩来释放连接数量距离", labels)
print(result)


for text in texts:
    valid, score = is_valid_sentence(text)
    print(f"文本：{text}，有效：{valid}，概率：{score:.3f}")
