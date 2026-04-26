# Qwen2.5-1.5B Quantization Evaluation Results

以下是自动整理的各项评测指标汇总。对比涵盖了不同位宽（Bit-width）、不同校准策略在语言建模（Perplexity, PPL）以及零样本下游任务（Zero-shot Accuracy）上的表现。

> [!NOTE]
> - **PPL (Wikitext2)**: 困惑度，数值**越低越好**。
> - **Hellaswag / PiQA**: 零样本准确率，数值**越高越好**。

| Bit-width | Strategy | PPL (Wikitext2) ↓ | Hellaswag (%) ↑ | PiQA (%) ↑ |
| :---: | :--- | :---: | :---: | :---: |
| **FP16** | Full Precision (BF16) Baseline | **9.27** | **50.14** | **75.41** |
| | | | | |
| **8-bit** | MoQ End-to-End | **9.65** | **49.95** | **75.63** |
| **8-bit** | Intermediate MSE | 9.69 | 49.83 | 75.52 |
| **8-bit** | Static (Baseline) | 20.38 | 40.60 | 65.61 |
| | | | | |
| **6-bit** | MoQ End-to-End | **44.35** | **41.14** | **67.90** |
| **6-bit** | Intermediate MSE | 18488.80 | 26.35 | 54.24 |
| **6-bit** | Static (Baseline) | 11252.02 | 26.16 | 51.90 |
| | | | | |
| **4-bit** | MoQ End-to-End | 363267.44 | 25.25 | 52.77 |
| **3-bit** | MoQ End-to-End | 382094.81 | 25.59 | 51.90 |
| **2-bit** | MoQ End-to-End | 5805343.50 | 25.38 | 53.32 |

---

### 💡 核心观察与分析：

1. **8-bit 的绝对碾压：**
   在 8-bit 下，MoQ 和 Intermediate MSE 都展现出了近乎无损（相比 FP16）的表现。而 `Static` 静态量化（所有层强行使用相同格式）则出现了明显的掉点（PPL 从 9.27 暴涨到 20.38）。这证明了**混合精度格式分配**在 8-bit 下具有极大的价值。

2. **6-bit 的生存分水岭：**
   当预算收紧到 6-bit 时，传统方案（Static 和 Intermediate MSE）的 PPL 彻底崩溃（高达 1万+），下游任务准确率直接跌至随机瞎猜的水平。
   而 **MoQ End-to-End** 依然保持了 44.35 的 PPL，且在 Hellaswag 和 PiQA 上保留了可观的准确率（相比 FP16 仅掉点约 9%），展现出了强大的韧性。

3. **极低比特（≤4-bit）的挑战：**
   当位宽降至 4-bit 及以下时，即使是 MoQ 也无法力挽狂澜，模型完全崩溃。这说明对于 1.5B 这样参数量较小的模型，4-bit 以下的信息损失是毁灭性的（或需要引入更先进的权重量化/微调技术而非仅后训练混合精度）。
