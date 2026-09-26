# DecisionOps 基础知识手册

> 目标：在正式开发前，先建立一套足够可靠的理论与工程认知，避免把“模型自己给出的 confidence”误当成真实可信度。
>
> 项目暂定名：**DecisionOps / Decision Reliability Gateway**
>
> 核心问题：**什么时候可以相信一个 AI decision，并让软件自动执行？什么时候必须升级到更强模型、验证器或人工？**

---

## 1. 先理解我们到底在做什么

传统 LLM 应用通常是：

```text
Input
  ↓
LLM
  ↓
Generated text / JSON
  ↓
Program parses result
```

System-One / Decision Model 更接近：

```text
State + Question + Choices
          ↓
     Decision Model
          ↓
Typed probability distribution
          ↓
Program decision
```

例如：

```json
{
  "refund": 0.87,
  "reject": 0.08,
  "review": 0.05
}
```

这类模型的价值不在于“会说话”，而在于快速回答：

- 选哪个工具？
- 走哪个模型？
- 是否 retry？
- 是否需要人工？
- 风险等级是多少？
- 某个判断为真的概率是多少？

TypeSafe Jev 把这类模型称为 **System One Models**；Laya 是当前值得研究的开源 System-One 风格模型之一。

---

## 2. 最重要的概念：Probability 不等于 Reliability

如果模型输出：

```text
refund = 0.90
```

不能直接理解成：

> “这个退款判断有 90% 概率正确。”

模型的 softmax / self-reported probability 可能严重过度自信。

真正希望达到的状态是：

```text
所有预测 confidence ≈ 0.9 的样本
长期真实正确率也 ≈ 90%
```

这叫 **Calibration（概率校准）**。

### 2.1 Calibration 的定义

理想情况下：

```text
P(correct | confidence = 0.8) ≈ 0.8
P(correct | confidence = 0.9) ≈ 0.9
```

如果模型经常：

```text
confidence = 0.9
actual accuracy = 0.65
```

说明模型是 **over-confident**。

### 2.2 为什么这是项目的第一理论支柱

Guo et al. 在 ICML 2017 的经典论文 *On Calibration of Modern Neural Networks* 中系统说明：现代神经网络虽然准确率高，但往往校准不好；Temperature Scaling 是一个简单且常用的 post-hoc calibration 方法。

这意味着我们的系统不能只做：

```text
if confidence > 0.9:
    auto_execute()
```

而要做：

```text
raw confidence
      ↓
calibration model
      ↓
calibrated reliability
      ↓
policy
```

---

## 3. 常见 Calibration 方法

### 3.1 Temperature Scaling

模型原始 logits：

```text
z = [3.8, 1.2, 0.4]
```

普通 softmax：

```text
softmax(z)
```

Temperature Scaling：

```text
softmax(z / T)
```

其中 `T` 在 calibration dataset 上拟合。

通常：

- `T > 1`：降低过度自信
- `T < 1`：提高置信度

优点：

- 简单
- 稳定
- 参数少
- 非常适合 MVP

限制：

- 通常只调整整体 sharpness
- 对复杂 domain shift 不一定够

### 3.2 Platt Scaling

使用逻辑回归把 raw score 映射成概率，经典二分类方法。

### 3.3 Isotonic Regression

非参数单调映射：

```text
raw score → empirical probability
```

优点是灵活；缺点是小数据集容易过拟合。

### 3.4 Beta Calibration / Vector Scaling

适用于更复杂的概率校准场景，可作为后续扩展。

---

## 4. 怎么衡量 Calibration 好不好

### 4.1 Reliability Diagram

例如模型输出：

| Confidence Bin | Avg Confidence | Actual Accuracy |
|---|---:|---:|
| 0.5-0.6 | 0.55 | 0.52 |
| 0.6-0.7 | 0.65 | 0.61 |
| 0.8-0.9 | 0.85 | 0.73 |
| 0.9-1.0 | 0.94 | 0.78 |

如果完美校准：

```text
confidence ≈ accuracy
```

### 4.2 ECE - Expected Calibration Error

把概率分桶后，计算预测 confidence 与真实 accuracy 的加权差异。

直观理解：

```text
ECE 越低越好
```

但 ECE 不是万能指标，对 binning 比较敏感，所以不能单独使用。

### 4.3 Brier Score

二分类时：

```text
Brier = mean((p - y)^2)
```

多分类也有对应版本。

它同时考虑：

- 概率是否准确
- 判断是否正确

对 Decision Model 很重要。

### 4.4 NLL - Negative Log Likelihood

对“过度自信但预测错误”的情况惩罚非常大。

DecisionOps 第一版建议至少支持：

```text
Accuracy
ECE
Brier Score
NLL
Reliability Diagram
```

---

## 5. 第二理论支柱：Selective Prediction

我们的目标并不是：

> AI 必须回答所有请求。

而是：

> AI 只自动处理自己足够可靠的那部分请求。

这叫 **Selective Prediction / Reject Option**。

流程：

```text
Prediction
    ↓
Reliability score
    ↓
threshold
   /   \
accept  abstain
```

`abstain` 可以意味着：

- 换更强模型
- 再调用 verifier
- 请求更多 context
- 人工处理

SelectiveNet（ICML 2019）就是该研究方向的重要代表工作。

---

## 6. Risk 与 Coverage

### Coverage

系统自动接受的样本比例：

```text
1000 requests
400 auto accepted
coverage = 40%
```

### Selective Risk

被自动接受的这部分请求中的错误率。

例如：

```text
400 auto accepted
4 errors
risk = 1%
```

产品真正应该让用户配置的是：

```yaml
max_error_rate: 0.01
```

而不是：

```yaml
confidence_threshold: 0.9
```

因为不同模型的 0.9 含义完全不同。

### Risk-Coverage Curve

通常：

```text
threshold ↑
coverage ↓
risk ↓
```

系统的价值是告诉用户：

```text
Target Risk 1%
→ Coverage 38%

Target Risk 5%
→ Coverage 72%
```

用户再决定自动化程度。

---

## 7. 第三理论支柱：Conformal Risk Control

Calibration 解决：

> “0.9 是否大体对应 90%？”

Conformal Risk Control 更进一步解决：

> “如何根据有限 calibration data，在指定风险目标下选择阈值？”

抽象目标：

```text
E[L(new_sample, λ_hat)] <= α
```

其中：

- `L`：风险函数
- `λ`：threshold / policy parameter
- `α`：用户允许的目标风险

例如：

```yaml
max_auto_error_rate: 0.01
confidence_level: 0.95
```

系统使用 calibration set 算出：

```text
threshold = 0.963
```

生产环境：

```text
reliability >= 0.963
→ AUTO

otherwise
→ ESCALATE
```

这比人工拍一个 `0.9` threshold 强得多。

---

## 8. 2026 年 LLM Routing 的直接证据

2026 ACL Student Research Workshop 的论文：

**Conformal LLM Routing with Distribution-Free Safety Guarantees**

核心思想与我们的项目非常接近：

```text
query
 ↓
gating score
 ↓
conformal calibration
 ↓
safe threshold
 ↓
cheap model / expensive model
```

其目标不是简单优化 cost，而是约束：

> 被路由给 cheap model 的请求中，cheap model 错而 expensive model 对的 violation rate。

这说明：

**“根据可靠性决定模型升级 / fallback”已经有正式研究依据。**

---

## 9. System-One 与 DecisionOps 的关系

### TypeSafe Jev

公开定位：

```text
unstructured state
      ↓
System One Model
      ↓
typed probabilistic decisions
```

TypeSafe 声称 Jev 使用 RLCD：

```text
Reinforcement Learning for Calibrated Decisions
```

重点不是生成文字，而是让概率更能反映 epistemic uncertainty。

但 Jev 的完整 architecture / RLCD recipe 没有开源。

### Laya

Laya 是非常重要的开源参考。

其公开仓库描述的训练方式包括：

- policy 输出 probability distribution
- exploration 在 logits 上加噪声
- reward 使用 strictly proper scoring rules
- log score / spherical score
- ordinal task 使用 ranked probability score
- REINFORCE + group-mean baseline
- 后处理拟合 temperature

这说明：

**Decision Model 的概率训练与校准并不是只能靠 TypeSafe 私有技术才能实现。**

不过 DecisionOps 不要求用户必须使用 Laya；它应该 model-agnostic。

---

## 10. Proper Scoring Rules

如果未来进入训练阶段，需要理解这个概念。

一个 **strictly proper scoring rule** 的重要性质是：

> 当模型诚实报告自己真实相信的概率分布时，期望 reward 最大。

最常见的是 Log Score：

```text
reward = log p(y_true)
```

如果真实答案是 A：

```text
A = 0.9 → reward 高
A = 0.2 → reward 低
```

而过度自信且答错会受到非常强的惩罚。

这类目标函数非常适合 Decision Model。

---

## 11. 为什么“高 confidence”仍然不能永久相信

即使模型在 calibration dataset 上非常准，生产环境仍有：

```text
Distribution Shift
```

例如：

- 中文流量突然增加
- 新业务类型上线
- 用户输入长度变化
- fraud pattern 改变
- 模型版本升级
- prompt/schema 改变

之前：

```text
confidence 0.9 → accuracy 94%
```

生产一个月后可能变成：

```text
confidence 0.9 → accuracy 81%
```

所以 DecisionOps 不能只做一次 calibration。

需要：

```text
Calibration
 ↓
Deploy
 ↓
Collect Outcome
 ↓
Drift Detection
 ↓
Recalibration / Recertification
```

这也是后续真正形成产品价值的地方。

---

## 12. 我们最终要提供的不是 Confidence，而是 Reliability Profile

模型输出：

```json
{
  "choice": "refund",
  "raw_probability": 0.92
}
```

DecisionOps 输出：

```json
{
  "choice": "refund",
  "raw_probability": 0.92,
  "calibrated_probability": 0.84,
  "policy": {
    "target_risk": 0.01,
    "decision": "escalate"
  },
  "reliability": {
    "coverage": 0.41,
    "observed_error_rate": 0.012,
    "calibration_status": "valid"
  }
}
```

这就是项目真正的核心差异。

---

## 13. 核心架构认知

```text
                    DecisionOps

Model / Decision Provider
        ↓
Raw probability
        ↓
Calibration Layer
        ↓
Calibrated Reliability
        ↓
Risk / Coverage Engine
        ↓
Policy
   ┌────┼─────────┐
   ↓    ↓         ↓
 AUTO  ESCALATE  HUMAN
        ↓
     Outcome
        ↓
 Reliability Store
        ↓
 Drift / Recalibration
```

---

## 14. 第一版不要做的事情

不要一开始做：

- 自己训练 Jev clone
- 自己发明 RLCD
- 通用 Agent Framework
- 完整 AI Gateway
- 多 Agent orchestration
- prompt management
- vector database
- 全功能 observability platform

第一版只验证：

> **给定一个 Decision Model 和一份有 ground truth 的 dataset，我们能否准确描述：它什么时候值得相信？**

---

## 15. MVP 的输入 / 输出

### 输入 Dataset

```json
{"id":"1","state":"...","question":"...","choices":["A","B","C"],"label":"B"}
```

### Model Adapter 输出

```json
{
  "prediction": "B",
  "probabilities": {
    "A": 0.07,
    "B": 0.88,
    "C": 0.05
  },
  "latency_ms": 31
}
```

### DecisionOps Eval 输出

```text
Accuracy: 86.4%
ECE: 0.071
Brier: 0.132
NLL: 0.418

Target risk <= 1%
Coverage: 34.7%
Threshold: 0.964

Target risk <= 5%
Coverage: 71.2%
Threshold: 0.872
```

---

## 16. 推荐先掌握的论文

### Level 1 - 必读

1. **On Calibration of Modern Neural Networks** - Guo et al., ICML 2017  
   理解：Calibration、ECE、Temperature Scaling。

2. **SelectiveNet: A Deep Neural Network with an Integrated Reject Option** - Geifman & El-Yaniv, ICML 2019  
   理解：Selective Prediction、Reject Option、Risk-Coverage。

3. **Conformal Risk Control** - Angelopoulos et al.  
   理解：如何从 calibration set 推导风险控制 threshold。

### Level 2 - 和当前产品直接相关

4. **Conformal LLM Routing with Distribution-Free Safety Guarantees** - ACL 2026 SRW  
   理解：如何把 risk control 用在 cheap / expensive LLM routing。

5. **TypeSafe AI - Introducing System One Models & Jev** - 2026  
   理解：System-One 产品范式、typed probabilistic decisions、RLCD 目标。

6. **Laya 开源实现**  
   理解：公开的 System-One-like architecture、proper scoring rules、RL-style calibration training。

---

## 17. 推荐参考的开源项目

### 模型 / System-One

- Laya  
  https://github.com/he-jev/laya

- TypeSafe System One Adapter  
  https://github.com/typesafe-ai/system-one-adapter-python

- One System  
  https://github.com/rawwerks/one-system

### Calibration / Risk Control

- Conformal Risk Control  
  https://github.com/aangelopoulos/conformal-risk

- Conformal Risk Training  
  https://github.com/chrisyeh96/conformal-risk-training

### 论文

- Calibration, ICML 2017  
  https://proceedings.mlr.press/v70/guo17a.html

- SelectiveNet, ICML 2019  
  https://proceedings.mlr.press/v97/geifman19a.html

- Conformal LLM Routing, ACL 2026  
  https://aclanthology.org/2026.acl-srw.70/

- TypeSafe Jev  
  https://typesafe.ai/blog/introducing-system-one-models-and-jev

---

## 18. 开发前必须牢记的五句话

1. **Softmax 不是可信度证明。**
2. **Calibration 是群体统计性质，不保证单个样本一定正确。**
3. **真正的自动化策略应该围绕 risk / coverage，而不是拍脑袋 confidence threshold。**
4. **任何 certificate 都依赖数据分布假设；distribution shift 必须监控。**
5. **DecisionOps 的核心不是创造一个更聪明的模型，而是量化和管理“什么时候值得相信模型”。**

---

## 19. 最终心智模型

把整个项目记成这一条链：

```text
AI gives a probability
        ↓
Do not trust it yet
        ↓
Calibrate it
        ↓
Measure risk vs coverage
        ↓
Certify / select threshold
        ↓
AUTO or ESCALATE
        ↓
Observe real outcome
        ↓
Detect drift
        ↓
Recalibrate
```

这就是 DecisionOps 的理论和产品核心。
