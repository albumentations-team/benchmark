# ImageNet training-step results

Each cell trains on the batches an input path delivers and reports images per second over 150 batches. Every recipe
starts with RandomResizedCrop and HorizontalFlip. Ranges are exact order-statistic intervals over 5 seeds; they
cover the median with 93.75% probability. *Of ceiling* divides each cell's rate by the model-only rate measured in
the same process. *vs AX* is the ratio of medians with an exact 95% interval; *tied* means the interval includes
1.00.

## ResNet-18, batch 256

Model-only ceiling: 1,707 images/s (range 1,659–1,773 over 140 cells).

### Crop and flip only

| Path | Images/s | Of ceiling | vs AX | Loader wait | Peak VRAM (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| AlbumentationsX | 1,749 (1,732–1,767) | 100% (99–100) | reference | 0% | 4,842 |
| TorchVision CPU | 1,706 (1,643–1,725) | 98% (97–98) | 0.98× slower | 0% | 4,844 |
| Pillow | 1,697 (1,670–1,732) | 98% (98–98) | 0.97× slower | 0% | 4,844 |
| DALI | 1,669 (1,613–1,716) | 97% (96–98) | 0.95× slower | 79% | 6,464 |
| Pillow-AllSIMD | 1,664 (1,663–1,673) | 98% (98–98) | 0.95× slower | 0% | 4,844 |
| Kornia GPU | 1,469 (1,402–1,522) | 85% (83–89) | 0.84× slower | 29% | 4,848 |
| Kornia CPU | 1,401 (1,253–1,448) | 82% (73–83) | 0.80× slower | 46% | 4,842 |
| TorchVision GPU | 1,048 (1,026–1,066) | 61% (60–61) | 0.60× slower | 0% | 5,530 |

### ColorJitter

| Path | Images/s | Of ceiling | vs AX | Loader wait | Peak VRAM (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| AlbumentationsX | 1,744 (1,709–1,757) | 100% (99–100) | reference | 1% | 4,844 |
| DALI | 1,565 (1,542–1,606) | 92% (92–93) | 0.90× slower | 83% | 6,910 |
| TorchVision CPU | 964 (910–978) | 56% (53–58) | 0.55× slower | 49% | 4,842 |
| Kornia CPU | 680 (583–685) | 39% (34–41) | 0.39× slower | 78% | 4,844 |
| TorchVision GPU | 439 (439–451) | 26% (26–26) | 0.25× slower | 0% | 5,534 |
| Kornia GPU | 287 (278–291) | 17% (16–17) | 0.16× slower | 0% | 4,930 |

### ColorJitter+Erasing

| Path | Images/s | Of ceiling | vs AX | Loader wait | Peak VRAM (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| AlbumentationsX | 1,738 (1,701–1,753) | 100% (100–100) | reference | 1% | 4,842 |
| DALI | 1,562 (1,536–1,591) | 92% (91–92) | 0.90× slower | 83% | 6,912 |
| TorchVision CPU | 933 (899–941) | 55% (53–56) | 0.54× slower | 50% | 4,844 |
| Kornia CPU | 626 (522–630) | 36% (31–37) | 0.36× slower | 74% | 4,842 |
| TorchVision GPU | 378 (378–387) | 23% (22–23) | 0.22× slower | 0% | 5,534 |
| Kornia GPU | 283 (279–288) | 17% (16–17) | 0.16× slower | 0% | 4,930 |

### RandAugment

| Path | Images/s | Of ceiling | vs AX | Loader wait | Peak VRAM (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| AlbumentationsX | 1,726 (1,698–1,745) | 100% (100–100) | reference | 0% | 4,844 |
| Pillow-AllSIMD | 1,656 (1,651–1,676) | 99% (98–99) | 0.96× slower | 0% | 4,842 |
| DALI | 1,631 (1,593–1,669) | 96% (95–96) | 0.95× slower | 82% | 6,654 |
| TorchVision CPU | 1,503 (1,319–1,527) | 89% (78–90) | 0.87× slower | 17% | 4,842 |
| Pillow | 1,379 (1,350–1,401) | 80% (80–82) | 0.80× slower | 38% | 4,842 |
| Kornia CPU | 799 (611–800) | 46% (36–48) | 0.46× slower | 71% | 4,844 |
| TorchVision GPU | 580 (576–590) | 34% (34–35) | 0.34× slower | 0% | 5,540 |
| Kornia GPU | 253 (243–257) | 15% (14–15) | 0.15× slower | 0% | 5,016 |

## ResNet-50, batch 128

Model-only ceiling: 529 images/s (range 519–541 over 140 cells).

### Crop and flip only

| Path | Images/s | Of ceiling | vs AX | Loader wait | Peak VRAM (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| AlbumentationsX | 532 (528–535) | 99% (99–99) | reference | 0% | 7,542 |
| Pillow | 532 (527–536) | 99% (99–99) | 1.00× tied | 0% | 7,542 |
| TorchVision CPU | 532 (520–533) | 99% (98–99) | 1.00× tied | 0% | 7,542 |
| Pillow-AllSIMD | 525 (522–527) | 99% (99–99) | 0.99× slower | 0% | 7,542 |
| Kornia CPU | 523 (518–525) | 98% (98–99) | 0.98× slower | 0% | 7,542 |
| DALI | 522 (518–525) | 99% (98–99) | 0.98× slower | 53% | 8,650 |
| Kornia GPU | 521 (515–529) | 98% (98–99) | 0.98× slower | 0% | 7,546 |
| TorchVision GPU | 439 (438–442) | 83% (82–84) | 0.83× slower | 0% | 8,034 |

### ColorJitter

| Path | Images/s | Of ceiling | vs AX | Loader wait | Peak VRAM (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| TorchVision CPU | 528 (518–531) | 99% (98–100) | 1.00× tied | 0% | 7,542 |
| AlbumentationsX | 527 (522–530) | 99% (99–100) | reference | 0% | 7,542 |
| Kornia CPU | 524 (521–527) | 99% (99–99) | 1.00× tied | 0% | 7,542 |
| DALI | 515 (512–518) | 98% (97–98) | 0.98× slower | 58% | 8,970 |
| TorchVision GPU | 279 (277–281) | 53% (52–53) | 0.53× slower | 0% | 8,036 |
| Kornia GPU | 209 (206–211) | 40% (39–40) | 0.40× slower | 0% | 7,570 |

### ColorJitter+Erasing

| Path | Images/s | Of ceiling | vs AX | Loader wait | Peak VRAM (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| AlbumentationsX | 527 (524–531) | 100% (100–100) | reference | 0% | 7,542 |
| TorchVision CPU | 526 (520–528) | 99% (99–100) | 1.00× tied | 0% | 7,542 |
| Kornia CPU | 524 (520–526) | 99% (99–99) | 0.99× tied | 0% | 7,542 |
| DALI | 514 (511–516) | 98% (97–98) | 0.98× slower | 59% | 8,842 |
| TorchVision GPU | 255 (251–256) | 49% (48–49) | 0.48× slower | 0% | 8,038 |
| Kornia GPU | 207 (204–209) | 39% (38–40) | 0.39× slower | 0% | 7,570 |

### RandAugment

| Path | Images/s | Of ceiling | vs AX | Loader wait | Peak VRAM (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pillow-AllSIMD | 528 (525–529) | 100% (100–100) | 1.00× tied | 0% | 7,542 |
| AlbumentationsX | 527 (523–530) | 100% (100–100) | reference | 0% | 7,542 |
| TorchVision CPU | 526 (518–527) | 99% (99–99) | 1.00× tied | 0% | 7,542 |
| Pillow | 525 (524–533) | 99% (99–100) | 1.00× tied | 0% | 7,542 |
| Kornia CPU | 524 (519–526) | 99% (99–99) | 0.99× tied | 0% | 7,542 |
| DALI | 520 (517–523) | 99% (98–99) | 0.99× slower | 57% | 8,714 |
| TorchVision GPU | 331 (327–333) | 63% (62–63) | 0.63× slower | 0% | 8,040 |
| Kornia GPU | 191 (187–193) | 36% (36–37) | 0.36× slower | 0% | 7,616 |

Loader wait is the share of the window the training loop spent pulling the next batch. GPU work that a path
runs after the batch arrives (Kornia GPU, TorchVision GPU) belongs to the step, and DALI's wait includes
its own GPU work for that batch.
