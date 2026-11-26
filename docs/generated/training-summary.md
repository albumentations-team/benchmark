# ImageNet training-step results

Each cell trains on the batches an input path delivers and reports images per second over 150 batches. Every recipe
starts with RandomResizedCrop and HorizontalFlip. Ranges are exact order-statistic intervals over 5 seeds; they
cover the median with 93.75% probability. *Of ceiling* divides each cell's rate by the model-only rate measured in
the same process. *vs AX* is the ratio of medians with an exact 95% interval; *tied* means the interval includes
1.00.

## ResNet-18, batch 256

Model-only ceiling: 1,703 images/s (range 1,670–1,768 over 140 cells).

### Crop and flip only

| Path | Images/s | Of ceiling | vs AX | Loader wait | Peak VRAM (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| AlbumentationsX | 1,734 (1,708–1,750) | 99% (99–100) | reference | 0% | 4,842 |
| Pillow | 1,716 (1,698–1,734) | 98% (98–98) | 0.99× tied | 1% | 4,844 |
| TorchVision CPU | 1,715 (1,647–1,726) | 98% (97–98) | 0.99× tied | 0% | 4,844 |
| Pillow-AllSIMD | 1,663 (1,655–1,674) | 98% (97–98) | 0.96× slower | 0% | 4,844 |
| DALI | 1,659 (1,631–1,696) | 96% (96–97) | 0.96× slower | 80% | 6,464 |
| Kornia GPU | 1,490 (1,454–1,529) | 87% (86–89) | 0.86× slower | 28% | 4,848 |
| Kornia CPU | 1,349 (1,220–1,432) | 79% (71–83) | 0.78× slower | 38% | 4,842 |
| TorchVision GPU | 1,045 (1,028–1,059) | 61% (60–61) | 0.60× slower | 0% | 5,554 |

### ColorJitter

| Path | Images/s | Of ceiling | vs AX | Loader wait | Peak VRAM (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| AlbumentationsX | 1,714 (1,683–1,731) | 100% (99–100) | reference | 0% | 4,844 |
| DALI | 1,612 (1,605–1,626) | 95% (94–95) | 0.94× slower | 91% | 6,654 |
| TorchVision CPU | 973 (959–974) | 56% (56–58) | 0.57× slower | 47% | 4,842 |
| Kornia CPU | 640 (539–685) | 38% (32–40) | 0.37× slower | 72% | 4,844 |
| TorchVision GPU | 441 (439–448) | 26% (26–26) | 0.26× slower | 0% | 5,556 |
| Kornia GPU | 286 (285–288) | 17% (17–17) | 0.17× slower | 0% | 4,930 |

### ColorJitter+Erasing

| Path | Images/s | Of ceiling | vs AX | Loader wait | Peak VRAM (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| AlbumentationsX | 1,697 (1,685–1,716) | 100% (100–101) | reference | 2% | 4,842 |
| DALI | 1,614 (1,589–1,628) | 94% (93–95) | 0.95× slower | 92% | 6,656 |
| TorchVision CPU | 936 (928–944) | 55% (54–56) | 0.55× slower | 50% | 4,844 |
| Kornia CPU | 600 (510–632) | 36% (30–38) | 0.35× slower | 75% | 4,842 |
| TorchVision GPU | 381 (379–388) | 22% (22–23) | 0.22× slower | 0% | 5,556 |
| Kornia GPU | 285 (280–287) | 17% (16–17) | 0.17× slower | 0% | 4,930 |

### RandAugment

| Path | Images/s | Of ceiling | vs AX | Loader wait | Peak VRAM (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| AlbumentationsX | 1,701 (1,684–1,708) | 100% (100–100) | reference | 0% | 4,844 |
| Pillow-AllSIMD | 1,675 (1,667–1,680) | 99% (99–99) | 0.98× slower | 0% | 4,842 |
| DALI | 1,635 (1,621–1,653) | 96% (96–96) | 0.96× slower | 82% | 6,718 |
| TorchVision CPU | 1,516 (1,506–1,529) | 89% (89–91) | 0.89× slower | 15% | 4,842 |
| Pillow | 1,373 (1,329–1,407) | 79% (77–82) | 0.81× slower | 38% | 4,842 |
| Kornia CPU | 770 (652–795) | 46% (39–47) | 0.45× slower | 68% | 4,844 |
| TorchVision GPU | 582 (579–593) | 34% (34–35) | 0.34× slower | 0% | 5,566 |
| Kornia GPU | 254 (253–257) | 15% (15–15) | 0.15× slower | 0% | 5,016 |

## ResNet-50, batch 128

Model-only ceiling: 529 images/s (range 520–540 over 140 cells).

### Crop and flip only

| Path | Images/s | Of ceiling | vs AX | Loader wait | Peak VRAM (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| AlbumentationsX | 535 (532–536) | 99% (99–99) | reference | 0% | 7,542 |
| Pillow | 532 (526–535) | 99% (99–99) | 1.00× tied | 0% | 7,542 |
| TorchVision CPU | 529 (528–532) | 99% (99–99) | 0.99× slower | 0% | 7,542 |
| DALI | 523 (522–525) | 99% (99–99) | 0.98× slower | 53% | 8,778 |
| Kornia CPU | 523 (520–526) | 98% (98–99) | 0.98× slower | 0% | 7,542 |
| Pillow-AllSIMD | 522 (518–528) | 99% (98–99) | 0.98× slower | 0% | 7,542 |
| Kornia GPU | 520 (516–527) | 98% (98–99) | 0.97× slower | 0% | 7,546 |
| TorchVision GPU | 440 (437–442) | 83% (83–83) | 0.82× slower | 0% | 8,024 |

### ColorJitter

| Path | Images/s | Of ceiling | vs AX | Loader wait | Peak VRAM (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| AlbumentationsX | 529 (525–532) | 99% (99–100) | reference | 0% | 7,542 |
| Kornia CPU | 524 (523–526) | 99% (99–99) | 0.99× slower | 0% | 7,542 |
| TorchVision CPU | 523 (520–529) | 99% (99–99) | 0.99× tied | 0% | 7,542 |
| DALI | 518 (516–519) | 98% (97–98) | 0.98× slower | 71% | 8,842 |
| TorchVision GPU | 280 (278–281) | 53% (53–53) | 0.53× slower | 0% | 8,028 |
| Kornia GPU | 209 (206–210) | 40% (40–40) | 0.40× slower | 0% | 7,570 |

### ColorJitter+Erasing

| Path | Images/s | Of ceiling | vs AX | Loader wait | Peak VRAM (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| AlbumentationsX | 529 (527–531) | 100% (100–100) | reference | 0% | 7,542 |
| Kornia CPU | 523 (523–526) | 99% (99–99) | 0.99× slower | 0% | 7,542 |
| TorchVision CPU | 523 (520–524) | 99% (99–100) | 0.99× slower | 0% | 7,542 |
| DALI | 516 (515–518) | 98% (98–98) | 0.98× slower | 72% | 8,906 |
| TorchVision GPU | 255 (252–255) | 48% (48–49) | 0.48× slower | 0% | 8,028 |
| Kornia GPU | 207 (205–208) | 39% (39–40) | 0.39× slower | 0% | 7,570 |

### RandAugment

| Path | Images/s | Of ceiling | vs AX | Loader wait | Peak VRAM (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| AlbumentationsX | 528 (527–530) | 100% (100–100) | reference | 0% | 7,542 |
| Pillow-AllSIMD | 526 (522–529) | 100% (99–100) | 1.00× tied | 0% | 7,542 |
| Pillow | 525 (517–530) | 99% (98–100) | 0.99× tied | 0% | 7,542 |
| Kornia CPU | 524 (522–526) | 99% (99–99) | 0.99× slower | 0% | 7,542 |
| TorchVision CPU | 523 (518–526) | 99% (99–99) | 0.99× slower | 0% | 7,542 |
| DALI | 521 (520–523) | 98% (98–99) | 0.99× slower | 59% | 8,906 |
| TorchVision GPU | 331 (329–331) | 63% (62–63) | 0.63× slower | 0% | 8,030 |
| Kornia GPU | 193 (188–195) | 37% (36–37) | 0.36× slower | 0% | 7,616 |

Loader wait is the share of the window the training loop spent pulling the next batch. GPU work that a path
runs after the batch arrives (Kornia GPU, TorchVision GPU) belongs to the step, and DALI's wait includes
its own GPU work for that batch.
