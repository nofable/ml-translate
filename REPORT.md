After stage 1 in Optimization plan:
short commit hash: c13bc30
BLEU score: 0.4248

After stage 2.1 Learning Rate Scheduling:
short commit hash: f5b0c4d
BLEU score: 0.4568

After 2.2 Gradient Clipping with max grad 3:
short commit hash: 0b25e64
BLUE score: 0.4245

## Hardware
- **GPU:** 1× NVIDIA GeForce RTX 5060 (8 GB VRAM)
- **CPU:** AMD EPYC 7K62 (48 cores)
- **System RAM:** 128 GB
- **Storage:** 4 TB NVMe SSD
- **PCIe:** PCIe 4.0 x8

## Software Environment
- **Container Image:** vastai/pytorch (Docker)
- **Operating System:** Linux (containerized)
- **Framework:** PyTorch 2.10.0+cu130
- **CUDA Toolkit:** 13.0 (nvcc V13.0.88)
- **cuDNN:** 91501
- **NVIDIA Driver:** 580.82.09
- **CUDA Runtime (driver):** 13.0

28.54 seconds to running the full training with early stopping.
BLEU Score: 0.4892
