# Qwen2.5-1.5B Quantization Evaluation Results

以下是自动整理的各项评测指标汇总。对比涵盖了不同位宽（Bit-width）、不同校准策略在语言建模（Perplexity, PPL）以及零样本下游任务（Zero-shot Accuracy）上的表现。

> [!NOTE]
> - **PPL (Wikitext2)**: 困惑度，数值**越低越好**（目前正在后台重新评测）。
> - **Hellaswag / PiQA**: 零样本准确率，数值**越高越好**。

### 1. 核心量化策略对比 (MoQ vs. Intermediate MSE vs. Static)

| Bit-width | Strategy | PPL (Wikitext2) ↓ | Hellaswag (%) ↑ | PiQA (%) ↑ |
| :---: | :--- | :---: | :---: | :---: |
| **FP16** | Full Precision (BF16) Baseline | **9.27** | **50.23** | **76.01** |
| | | | | |
| **8-bit** | MoQ End-to-End | 9.65 | 50.07 | 75.63 |
| **8-bit** | Intermediate MSE | 9.69 | 50.53 | 75.73 |
| **8-bit** | Static (Baseline) | 20.38 | 40.60 | 65.61 |
| | | | | |
| **6-bit** | MoQ End-to-End | 44.35 | 41.14 | 67.90 |
| **6-bit** | Intermediate MSE | 18488.80 | 41.24 | 70.57 |
| **6-bit** | Static (Baseline) | 11252.02 | 26.16 | 51.90 |
| | | | | |
| **4-bit** | MoQ End-to-End | 363267.44 | 25.25 | 52.77 |
| **4-bit** | Intermediate MSE | - | 48.94 | 75.52 |

*(注：部分较差的 PPL 等待后台重测修正)*

### 2. 4-bit 静态数据格式深度对比 (Format Deep-Dive)

在 4-bit 下，探讨不同数据格式对性能的影响（Static 量化）：

| Format | Strategy | PPL (Wikitext2) ↓ | Hellaswag (%) ↑ | PiQA (%) ↑ |
| :---: | :--- | :---: | :---: | :---: |
| **INT4** | Static | 10087410.00 | 25.53 | 52.67 |
| **INT4+ACIQ** | Static | 721.87 | 30.00 | 64.36 |
| **MXFP4** | Static | 12.00 | 46.25 | 73.01 |
| **NVFP4** | Static | 10.43 | 48.02 | 74.65 |
| **NF4** | Static | 9.93 | 48.94 | 75.52 |

---

### 3. Weight-Only 权重专属量化 (Phase 1)

以下为纯权重（Weight-Only）量化的 PPL 结果，其零样本任务正在后台评测：

| Bit-width | Strategy | PPL (Wikitext2) ↓ | Hellaswag (%) ↑ | PiQA (%) ↑ |
| :---: | :--- | :---: | :---: | :---: |
| **FP16** | Full Precision (Baseline) | **9.27** | **50.23** | **76.01** |
| | | | | |
| **8-bit** | Intermediate MSE | 9.35 | 50.49 | 75.90 |
| **8-bit** | Cosine | 9.35 | Running... | Running... |
| **8-bit** | MoQ E2E | 9.29 | Running... | Running... |
| | | | | |
| **6-bit** | Intermediate MSE | 40.89 | 41.20 | 71.06 |
| **6-bit** | MoQ E2E | 9.50 | Running... | Running... |
| | | | | |
| **4-bit** | Intermediate MSE | 9.93 | Running... | Running... |
| **4-bit** | Cosine | 9.93 | Running... | Running... |
| **4-bit** | MoQ E2E | 9.92 | Running... | Running... |

---

### 💡 核心观察与分析：

1. **8-bit 的绝对碾压：**
   在 8-bit 下，MoQ 和 Intermediate MSE 都展现出了近乎无损（相比 FP16）的表现。证明了**混合精度格式分配**在 8-bit 下具有极大的价值。

2. **6-bit 的生存分水岭：**
   当预算收紧到 6-bit 时，6-bit Intermediate MSE 在零样本任务上能保持 41.24% / 70.57%，优于完全静态的格式分配。MoQ 也在持续展现韧性。

3. **4-bit 数据格式的显著差异：**
   在 4-bit 静态量化实验中，数据格式对困惑度有决定性影响：
   - 传统的 **INT4** 表现最差（PPL 高达千万级），即使加上 ACIQ 也只能降至 721。
   - **MXFP4** (PPL 12.00) 和 **NVFP4** (PPL 10.43) 能够极大地挽回性能。
   - **NF4** 的 PPL (9.93) 表现最好，大幅领先其他静态格式。
   - 这意味着在极低位宽下，选择具备更密集的中心点分布（如 NF4）的数据格式至关重要。
   
4. **Weight-Only (权重专属量化) 表现极佳：**
   - 相比于仅激活量化，权重专属量化在各策略下表现均显著提升。
   - 即使在 4-bit 下，Weight-Only 中 Intermediate MSE, Cosine 和 MoQ E2E 的 PPL 都能保持在 9.9 左右的优异水平。
   - 我们正在等待完整的 Hellaswag / PiQA 零样本评测结果来进一步验证这些模型的最终表现。
