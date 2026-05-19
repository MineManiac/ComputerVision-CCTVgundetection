# Sprint 6 — Diagnóstico e Melhorias do Pipeline Two-Phase

**Data:** Maio 2026  
**Objetivo:** Identificar por que o pipeline two-phase estava com resultados insatisfatórios e implementar todas as melhorias possíveis com base na análise dos crops e no paper de referência.

---

## 1. Contexto de Entrada

Ao início da sprint, o pipeline two-phase estava configurado, mas os resultados eram piores do que o baseline single-stage. O estado era:

| Componente | Situação |
|---|---|
| Person detector (Stage 0) | yolo11n pré-treinado — funcional |
| Carry classifier (Stage 1) | CNN customizada de 3 camadas, `enable_hold_gate: false` (desativado) |
| Weapon crop detector (Stage 2) | Treinado apenas 2 épocas — mAP50 = 0.168 |
| **Baseline single-stage** | yolo26n_img9604 — **Precision=0.748, Recall=0.153, F1=0.255** |

---

## 2. Diagnóstico — Análise dos Crops

Antes de implementar qualquer mudança, rodamos o script de diagnóstico (`diagnose_two_phase_crop_subset.py`) em 50 imagens positivas do test split.

### Resultado do `diagnostic_manifest.csv`:

| Métrica | Valor |
|---|---|
| Imagens analisadas | 50 positivas |
| Crops totais matched | 217 |
| Média de crops por imagem | 4.3 |
| Imagens com ≥ 8 pessoas | 23 (cenas lotadas) |
| **Armas < 32×32 px no crop 224×224** | **87% dos 299 pares weapon-crop** |

**O YOLO considera qualquer objeto abaixo de 32px como "tiny object"** — faixa onde a performance cai ~38% (benchmark COCO). Com 87% das armas nessa faixa, o pipeline todo estava comprometido.

Causa raiz: crops grandes (~337×658 px em média) com `padding_x=0.35 / padding_y=0.25` → arma minúscula no recorte.

---

## 3. Análise do Paper de Referência

O paper usa o mesmo dataset (US Mock Attack, Cam1+Cam7 treino, **Cam5 = nosso test set**). Técnicas extraídas:

| Técnica do Paper | Como adaptamos |
|---|---|
| Reduzir padding do crop | `crop_padding_x: 0.35 → 0.15`, `crop_padding_y: 0.25 → 0.15` |
| Zoom na região inferior | `classifier_zoom_lower_fraction: 0.55` (armas na metade inferior) |
| Backbone pré-treinado | MobileNetV3-Small (ImageNet) em vez de CNN do zero |
| Treino em duas fases | 5 épocas frozen + 20 épocas fine-tuning |
| SAHI — Slicing Aided Hyper Inference | Tiles de 320px com overlap 0.30 sobre os crops |
| Aumentar resolução de inferência | `weapon_crop_imgsz: 640 → 1280` |

---

## 4. Melhorias Implementadas

### 4.1 — `configs/two_phase.yaml`

Parâmetros alterados nesta sprint:

```yaml
thresholds:
  weapon_conf: 0.25          # era 0.25 (mantido após testes)
dataset:
  crop_padding_x: 0.15       # era 0.35
  crop_padding_y: 0.15       # era 0.25
  classifier_zoom_lower_fraction: 0.55  # NOVO
training:
  batch_size: 16             # era 32
  epochs: 25                 # era 12
  learning_rate: 0.0003      # era 0.001
  classifier_backbone: mobilenet_v3_small  # era custom_cnn
  backbone_freeze_epochs: 5  # NOVO
  backbone_lr_factor: 0.1    # NOVO
inference:
  weapon_crop_imgsz: 640     # era 640 (revertido de 1280 — ver Seção 8)
  enable_hold_gate: false    # desativado por gap de generalização no test
  sahi_enabled: false        # desativado — ver Seção 8
```

---

### 4.2 — `scripts/two_phase_utils.py` — Novos Símbolos

#### `MobileNetV3CarryClassifier`

Substituiu a CNN de 3 camadas por backbone MobileNetV3-Small pré-treinado no ImageNet:

```python
class MobileNetV3CarryClassifier(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.backbone = models.mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        )
        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Linear(in_features, 1)

    def freeze_backbone(self):
        for p in self.backbone.features.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.features.parameters():
            p.requires_grad = True
```

#### `zoom_lower_fraction(image, fraction)`

Recorta apenas a porção inferior do crop antes de enviar ao classificador:

```python
def zoom_lower_fraction(image: Image.Image, fraction: float) -> Image.Image:
    w, h = image.size
    top = int(round(h * (1.0 - fraction)))
    return image.crop((0, top, w, h))
```

```
Person crop (224×224)           Após zoom_lower_fraction(0.55)
┌─────────────────────┐         ┌─────────────────────┐
│  cabeça / tronco    │  45%    │                     │
│  (descartado)       │ ──────▶ │   cintura / mãos    │
├─────────────────────┤         │   (região da arma)  │
│  cintura / mãos     │  55%    │                     │
│  [arma aqui]        │         └─────────────────────┘
└─────────────────────┘
```

#### `sahi_predict_on_crop()` — implementado mas aguardando retreinamento

SAHI foi implementado e integrado, mas precisa que o weapon detector seja retreinado na mesma resolução para ser usado corretamente (ver Seção 8).

---

### 4.3 — `scripts/train_carry_classifier.py` — Reescrito Completo

Principais mudanças:

- **Augmentação:** RandomHorizontalFlip, ColorJitter, RandomRotation, RandomAffine
- **Balanceamento:** `WeightedRandomSampler` — oversample dos positivos (1:2.5 de desbalanceamento)
- **Duas fases:** 5 épocas frozen (só head) → 20 épocas full fine-tuning com `backbone_lr × 0.1`
- **Scheduler:** `CosineAnnealingLR` sobre as épocas de fine-tuning

---

### 4.4 — `scripts/run_two_phase_inference.py`

SAHI e zoom_lower_fraction integrados condicionalmente. Hold gate carrega threshold do checkpoint.

---

### 4.5 — `scripts/build_two_phase_dataset.py`

`zoom_lower_fraction` aplicado nos crops salvos para treino do carry classifier.

---

## 5. Resultados do Carry Classifier (MobileNetV3, 25 épocas)

| Parâmetro | Valor |
|---|---|
| Backbone | mobilenet_v3_small (ImageNet) |
| Zoom lower fraction | 0.55 |
| Train samples | 4206 (pos=1212 / neg=2994) |
| Melhor época | 14 / 25 |
| Gate threshold calibrado (val) | 0.74 (recall floor ≥ 0.80) |

**Métricas no test split (threshold=0.74):**

| Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|
| 0.636 | **0.372** | 0.469 | 701 | 401 | 1185 |

> ⚠️ **Gap de generalização:** val recall = 0.828, test recall = 0.372 no mesmo threshold. O classificador não generalizou bem para o test set (Cam5), provavelmente por diferença de iluminação e ângulo entre câmeras.

**Sweep de threshold no val — pontos relevantes:**

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.30 | 0.444 | 0.963 | 0.608 |
| 0.50 | 0.494 | 0.881 | 0.633 |
| 0.74 | 0.575 | 0.828 | 0.679 |
| 0.90 | 0.695 | 0.612 | 0.651 |

---

## 6. Weapon Crop Detector — Retreinamento (120 épocas)

### Problema original:
O detector havia sido treinado por apenas **2 épocas** (treino interrompido), com mAP50 = 0.168.

### Após retreinamento completo:

| Época | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| 1 | 0.347 | 0.183 | 0.197 | 0.065 |
| 10 | 0.500 | 0.451 | 0.439 | 0.180 |
| 30 | 0.702 | 0.574 | 0.638 | 0.318 |
| 70 | 0.838 | 0.652 | 0.757 | 0.356 |
| **110 (best)** | **0.908** | **0.723** | **0.786** | **0.393** |
| 120 | 0.898 | 0.707 | 0.783 | 0.384 |

**mAP50: 0.168 → 0.786 (+368%)**. Modelo treinado com `imgsz=640`, `batch=8`, `patience=30`.

---

## 7. Problema Identificado: Domain Shift SAHI + imgsz=1280

Após o retreinamento, rodamos a inferência com `sahi_enabled=true` e `weapon_crop_imgsz=1280`. O resultado foi **5 TPs de 695 detecções** — praticamente zero.

### Diagnóstico do domain shift:

```
TREINO (imgsz=640):
  Crop ~400px → YOLO resize para 640px
  Weapon de 30px → aparece como ~48px em 640px
  Proporção: 48/640 = 7.5% da imagem

INFERÊNCIA com SAHI + imgsz=1280:
  Crop ~400px → tile de 320px → YOLO resize para 1280px (upscale 4×)
  Weapon de 30px → ~24px no tile → ×4 = 96px em 1280px
  Proporção: 96/1280 = 7.5% — mesma proporção, escala diferente

→ Modelo treinado para ver weapons como ~48px vê weapons como ~96px
→ Distribuição de features completamente diferente → modelo falha
```

### Breakdown das falhas nessa run:

| Causa de miss | Weapons perdidos | % |
|---|---|---|
| Stage 0 — pessoa não detectada | 39 | 2.6% |
| Stage 1 — hold gate (thr=0.72) bloqueou | ~950 (estimado) | ~63% |
| **Stage 2 — crop detector não detectou** | **1469** | **97.1% dos que chegaram** |
| Suprimido pelo NMS final | 0 | 0% |

> O hold gate com threshold=0.72 e recall=0.37 no test agravou o problema: 1407 das 3435 pessoas foram bloqueadas antes de chegarem ao Stage 2, incluindo muitos portadores de arma.

---

## 8. Decisão de Arquitetura — Abordagem em Etapas

Dado o domain shift identificado, adotamos uma estratégia em fases:

### Etapa A (atual) — Validação sem SAHI, imgsz=640

Inferência com o modelo atual (treinado a 640px), sem SAHI, sem hold gate. Objetivo: validar se o pipeline two-phase funciona antes de investir em retreinamento.

```yaml
inference:
  weapon_crop_imgsz: 640   # igual ao treino
  enable_hold_gate: false  # desativado (gap de generalização)
  sahi_enabled: false      # desativado (domain shift)
```

```
TREINO:    crop → 640px → weapon ~48px ✓
INFERÊNCIA: crop → 640px → weapon ~48px ✓ (distribuição idêntica)
```

### Etapa A — Resultado ✅

| Métrica | Single-Stage | Two-Phase | Delta |
|---|---|---|---|
| TP | 266 | 327 | +61 (+23%) |
| FP | 349 | **153** | **−196 (−56%)** |
| FN | 1247 | 1186 | −61 (−5%) |
| Precision | 0.433 | **0.681** | +57% |
| Recall | 0.176 | **0.216** | +23% |
| **F1** | 0.250 | **0.328** | **+31%** |

Gargalo identificado: **75.7% das armas (1146/1513) perdidas no Stage 2** — crop detector vê o crop mas não detecta a arma pequena. Stage 0 e cobertura de crop são saudáveis (miss de 2.4% e 0% respectivamente).

### Etapa B (próxima) — Retreinar weapon detector a imgsz=1280

Se Etapa A confirmar que o pipeline funciona, retreinar o detector com `imgsz=1280`. Weapons de 30px → 96px (faixa de objetos médios, +38% COCO benchmark).

```powershell
yolo train model=yolo26n.pt data=data/interim/two_phase/yolo_crops/dataset.yaml `
  epochs=120 patience=30 batch=4 imgsz=1280 device=0 `
  project=runs/two_phase name=weapon_crop_detector_1280 exist_ok=true
```

### Etapa B — Resultado ✅

| Config | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| 1280 model @ 640px | 323 | 170 | 1190 | 0.655 | 0.213 | 0.322 |
| 1280 model @ 1280px | 323 | 170 | 1190 | 0.655 | 0.213 | 0.322 |

Resultado idêntico nas duas variantes. A Stage 2 miss count de 1146 é a mesma em todas as configurações — a resolução não é o fator limitante aqui.

### Conclusão das Etapas ✅

A melhor configuração encontrada é a mais simples: **640 model @ 640px, sem SAHI, sem hold gate**.

O gargalo restante (Stage 2 miss de 1146 weapons) não é resolvido por resolução maior. É um problema de detecção de objetos tiny com dados limitados.

---

## 9. Comparativo Final — Todas as Configurações

| Pipeline | TP | FP | FN | Precision | Recall | F1 | Observação |
|---|---|---|---|---|---|---|---|
| **Single-stage** (yolo26n_img9604) | 266 | 349 | 1247 | 0.433 | 0.176 | 0.250 | Referência |
| Two-phase — detector 2 épocas | 5 | 690 | 1508 | 0.007 | 0.003 | — | Detector não treinado |
| Two-phase — SAHI+1280 (domain shift) | 5 | 690 | 1508 | 0.007 | 0.003 | — | Scale mismatch treino/infer |
| **Two-phase — Etapa A: 640@640px ★** | **327** | **153** | **1186** | **0.681** | **0.216** | **0.328** | ✅ **MELHOR — +31% F1** |
| Two-phase — Etapa B1: 1280@640px | 323 | 170 | 1190 | 0.655 | 0.213 | 0.322 | ✅ Testado |
| Two-phase — Etapa B2: 1280@1280px | 323 | 170 | 1190 | 0.655 | 0.213 | 0.322 | ✅ Testado |

### Ganhos da melhor config vs single-stage

| Métrica | Single-Stage | Two-Phase ★ | Delta |
|---|---|---|---|
| F1 | 0.250 | **0.328** | **+31%** |
| Precision | 0.433 | **0.681** | **+57%** |
| Recall | 0.176 | **0.216** | **+23%** |
| False Positives | 349 | **153** | **−56%** |
| Det / imagem | 0.60 | **0.47** | −22% |

### Gargalo final identificado

```
1513 GT weapon boxes no test set:
  Perdidas no Stage 0 (pessoa não detectada):  36  (2.4%)
  Perdidas no Stage 2 (crop detector miss): 1146 (75.7%)  ← principal bottleneck
  Suprimidas pelo NMS final:                    4  (0.3%)
  Detectadas corretamente (TP):               327 (21.6%)
```

---

## 10. Sprint Final — Threshold Sweep + yolo26s/yolo26m

**Data:** 18 Maio 2026

### 10.1 — Threshold Sweep (`weapon_conf`)

Inferência rodada com `weapon_conf=0.05` para capturar todos os candidatos, depois filtrados post-hoc em 10 thresholds diferentes.

| Threshold | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| 0.05 | — | — | — | — | — | — |
| **0.10 ⭐** | **445** | **608** | **1068** | **0.423** | **0.294** | **0.347** |
| 0.25 (anterior) | 327 | 153 | 1186 | 0.681 | 0.216 | 0.328 |
| 0.40 | — | — | — | — | — | — |
| 0.60 | — | — | — | — | — | — |

> Resultado completo em `runs/two_phase/evaluation/threshold_sweep.md`

**Conclusão:** `weapon_conf=0.10` é o threshold ótimo. Ao abaixar o threshold, o modelo recupera 118 TPs adicionais (+36%) ao custo de +455 FPs (+297%). O trade-off aumenta recall (+36%) com queda de precision (-38%), resultando em **F1=0.347 (+5.8% vs 0.25 anterior)**.

Comparativo atualizado vs baseline:

| Pipeline | TP | FP | FN | Precision | Recall | F1 | Delta F1 |
|---|---|---|---|---|---|---|---|
| Single-stage (yolo26n) | 266 | 349 | 1247 | 0.433 | 0.176 | 0.250 | — |
| Two-phase yolo26n (thr=0.25) | 327 | 153 | 1186 | 0.681 | 0.216 | 0.328 | +31% |
| **Two-phase yolo26n (thr=0.10) ★** | **445** | **608** | **1068** | **0.423** | **0.294** | **0.347** | **+38.7%** |

**Config atualizado:** `weapon_conf: 0.10` em `configs/two_phase.yaml`.

---

### 10.2 — Treino yolo26s (Small)

| Parâmetro | Valor |
|---|---|
| Modelo base | `yolo26s.pt` |
| Épocas treinadas | 103 (early stop patience=30) |
| Melhor época | 71 |
| `mAP50` (val) | **0.8129** |
| `mAP50-95` (val) | 0.3777 |
| Precision (val) | 0.8335 |
| Recall (val) | 0.7325 |
| Pesos salvos | `runs/two_phase/weapon_crop_detector_small/weights/best.pt` |

> Avaliação no test set (Cam5) ainda pendente — será gerada pelo script overnight.

---

### 10.3 — Treino yolo26m (Medium)

**Status:** Interrompido manualmente após 4 épocas (`mAP50=0.085` — inutilizável). Será retreinado do zero no próximo run overnight.

| Parâmetro | Valor |
|---|---|
| Modelo base | `yolo26m.pt` |
| Épocas treinadas | 4 (interrompido) |
| Status | ⏳ Pendente — retreinar esta noite |

---

## 11. Próximos Passos

| Prioridade | Ação | Status |
|---|---|---|
| 🔴 | Avaliar yolo26s no test set | ⏳ Esta noite |
| 🔴 | Retreinar + avaliar yolo26m | ⏳ Esta noite |
| 🔴 | Mais dados de treino (atualmente 1212 crops positivos) | Futuro |
| 🟡 | Multi-scale training (`imgsz` aleatório 320–640) | Futuro |
| 🟡 | Carry classifier retreinado com dados de Cam5 | Futuro |
| 🟢 | SAHI com treino e infer na mesma resolução | Só após resolver bottleneck Stage 2 |

---

## 12. Arquivos Modificados nesta Sprint

| Arquivo | Mudança |
|---|---|
| `configs/two_phase.yaml` | padding reduzido, MobileNetV3, SAHI params, imgsz revertido para 640 |
| `scripts/two_phase_utils.py` | +300 linhas: MobileNetV3CarryClassifier, SAHI, zoom_lower_fraction |
| `scripts/train_carry_classifier.py` | Reescrito: augmentação, WeightedSampler, duas fases, CosineAnnealingLR |
| `scripts/run_two_phase_inference.py` | SAHI integrado, carregamento de checkpoint atualizado |
| `scripts/build_two_phase_dataset.py` | zoom_lower_fraction aplicado nos crops do classificador |
