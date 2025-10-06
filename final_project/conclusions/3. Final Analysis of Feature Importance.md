| #  | **Feature**        | **Cluster Separation**                       | **Correlation / MI / RF**              | **Interpretation**                                     |
| -- | ------------------ | -------------------------------------------- | -------------------------------------- | ------------------------------------------------------ |
| 1  | **LL\_mean**       | Excellent separation (distinct 0/1 clusters) | Corr \~0.28, MI \~0.27, RF \~0.08      | ✅ **Highly consistent — dominant single predictor**    |
| 2  | **1MTA\_mean**     | Moderate separation (right shift for 1s)     | Corr \~0.30, MI \~0.30, RF \~0.11      | ✅ **Strong across all metrics — key predictor**        |
| 3  | **cal\_PA\_mean**  | Strong separation (clear left shift)         | Corr \~0.17, MI \~0.02, RF \~0.08      | ⚡ **Good visual separation, moderate metric strength** |
| 4  | **MT\_calA\_mean** | Some separation (partial overlap)            | Corr \~0.11, MI \~0.11, RF \~0.03      | ⚠️ **Captures partial trends — possibly secondary**    |
| 5  | **ML\_mean**       | Good separation, distinct peaks              | Corr \~-0.24, MI low, RF \~0.05        | ⚠️ **Visually meaningful but statistically weak**      |
| 6  | **MA\_mean**       | Heavy overlap, weak visual pattern           | Corr \~-0.14, MI very low, RF \~0.02   | ❌ **Poor predictive value**                            |
| 7  | **TCA\_mean**      | Significant overlap                          | Corr \~0.08, MI very low, RF \~0.04    | ❌ **Low utility**                                      |
| 8  | **TUA\_mean**      | Heavy overlap, nearly identical              | Corr \~0.03, MI \~0.04, RF \~0.05      | ❌ **Minimal signal, high noise**                       |
| 9  | **5MTCA\_mean**    | Almost complete overlap                      | Corr \~-0.28, MI \~0.02, RF \~0.08     | ⚠️ **Conflicting metrics — possibly spurious**         |
| 10 | **MCL\_mean**      | Complete overlap                             | Corr ≈ 0, MI \~0.09, RF \~0.02         | ❌ **No separation — limited relevance**                |
| 11 | **MCL\_var**       | Spike at zero (no spread)                    | Corr \~0.12, MI very low, RF \~0.03    | ❌ **Statistically flat — weak**                        |
| 12 | **1MTA\_var**      | Spike at zero (no real separation)           | Corr \~0.21, MI \~0.13, RF \~0.02      | ⚡ **Statistically interesting but visually poor**      |
| 13 | **MT\_calA\_var**  | Spike at zero (flat visually)                | Corr ≈ 0, MI \~0.17, RF \~0.04         | ⚡ **Captures hidden non-linear effects**               |
| 14 | **LL\_var**        | Spike at zero (no shape)                     | Corr \~0.19, MI \~0.03, RF \~0.02      | ⚠️ **Slight linear signal, not visually clear**        |
| 15 | **cal\_PA\_var**   | Spike at zero                                | Corr \~0.15, MI very low, RF \~0.04    | ⚠️ **Mild signal in metrics, not visual**              |
| 16 | **MA\_var**        | Spike at zero                                | Corr \~-0.14, MI very low, RF \~0.02   | ❌ **Poor across all measures**                         |
| 17 | **ML\_var**        | Spike at zero                                | Corr \~-0.30, MI very low, RF very low | ❌ **Highly uninformative**                             |
| 18 | **5MTCA\_var**     | Spike at zero                                | Corr \~-0.28, MI very low, RF low      | ❌ **No contribution to model**                         |
| 19 | **TUA\_var**       | Spike at zero                                | Corr \~0.08, MI \~0.04, RF \~0.03      | ❌ **Flat, unhelpful**                                  |
| 20 | **TCA\_var**       | Spike at zero                                | Corr \~-0.17, MI very low, RF \~0.10   | ⚠️ **RF anomaly — possible noise capture**             |