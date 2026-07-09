# Training Cost Estimation Methodology

This document summarizes the direct GPU rental cost for the paper training runs.

The paper uses Verda (formerly DataCrunch) on-demand H100 SXM5 pricing as the reference rental rate.
This is a good fit for the paper context because it is a European provider with Nordic data centers, GDPR-aligned compliance positioning, 100% renewable energy, and the same class of hardware used for training.

- Paper H100 SXM5 rental rate: $2.29 per GPU-hour

## 1. Cost Formula

For a run that uses `g` GPUs for `t` seconds, total GPU-hours and cost are:

$$
\text{GPU-hours} = g \cdot \frac{t}{3600}
$$

$$
\text{cost} = \text{GPU-hours} \cdot 2.29
$$

## 2. SFT Training Cost

Provided runtime:

- Duration: 6,060 s (1 h 41 m)
- GPUs: 4 x H100

Computation:

- GPU-hours: $4 \cdot \frac{6060}{3600} = 6.7333$
- Cost: $6.7333 \cdot 2.29 = 15.4193$

Rounded total:

- SFT cost: **$15.42**

## 3. RL Training Cost

Provided runtime:

- Duration: 41,493 s (11 h 31 m 33 s) active training time to the deployed checkpoint (step 300 of 1,000)
- GPUs: 4 x H100

Computation:

- GPU-hours: $4 \cdot \frac{41493}{3600} = 46.1033$
- Cost: $46.1033 \cdot 2.29 = 105.3766$

Rounded total:

- RL cost: **$105.38**

## 4. Combined Training Cost

- Total cost: $15.4193 + 105.3766 = 120.7959$
- Rounded combined cost: **$120.80$**

## Reference

- **Verda H100 pricing:** [Verda products](https://verda.com/products)
- **Provider context used in the paper:** European provider, Nordic data centers, GDPR/compliance positioning, and 100% renewable energy
