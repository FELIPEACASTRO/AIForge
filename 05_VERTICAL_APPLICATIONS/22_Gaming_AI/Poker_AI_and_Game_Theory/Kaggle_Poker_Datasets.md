# Kaggle — Poker Datasets

> Curated index of **poker** datasets on Kaggle (pulled live via the Kaggle API) — hand classification, real online hand histories, heads-up gameplay, and card computer-vision. Starting points for poker ML research.

> ⚠️ **Research & education only.** Poker involves real-money risk and high variance; after rake, most players lose long-term. Nothing here is gambling advice, and **poker bots are banned by real-money sites**. If gambling is a problem: [BeGambleAware](https://www.begambleaware.org/), [GamCare](https://www.gamcare.org.uk/), [Gambling Therapy](https://www.gamblingtherapy.org/); 🇧🇷 [Jogo Responsável](https://www.gov.br/fazenda/pt-br/assuntos/loterias/jogo-responsavel), CVV 188.

## 🃏 Hand-classification benchmarks (classic ML)

| Dataset | Content | Link |
|---|---|---|
| **UCI Poker Hand Dataset** (mirror) | The classic 1M+ five-card hand-classification benchmark (10 classes) | https://www.kaggle.com/datasets/rasvob/uci-poker-hand-dataset |
| Poker Hand Classification | UCI-style classification task | https://www.kaggle.com/datasets/dysphoria/poker-hand-classification |
| Poker Hand Dataset (alphaepsilon) | UCI-format train/test | https://www.kaggle.com/datasets/alphaepsilon/poker-hand-dataset |
| All 5-Card Poker Combinations | Complete enumeration of 5-card hands | https://www.kaggle.com/datasets/benjaminsmith/all-5-hand-poker-card-combinations |
| 100,000 Five-Card-Draw Hands | Simulated 5-card-draw hands | https://www.kaggle.com/datasets/ivsalmon/100000-five-card-draw-poker-hands |
| Simulated Poker Hands | Synthetic hands for ML | https://www.kaggle.com/datasets/flynn28/simulated-poker-hands |

## ♠️ Real gameplay / hand histories

| Dataset | Content | Link |
|---|---|---|
| **Poker Hold'Em Games** (smeilz) | Real hold'em game logs | https://www.kaggle.com/datasets/smeilz/poker-holdem-games |
| **Online Poker Games** (murilogmamaral) | Online poker hand histories | https://www.kaggle.com/datasets/murilogmamaral/online-poker-games |
| Poker Heads-Up Gameplay (official Kaggle) | Heads-up match gameplay logs | https://www.kaggle.com/datasets/kaggle/poker-heads-up-gameplay |
| Poker Spin & Go (1€) | Real Spin & Go tournament hands | https://www.kaggle.com/datasets/bowaka/poker-spin-and-go-1-euro |
| Mandine's Real Poker Hands (MRPH) | Real-hands dataset | https://www.kaggle.com/datasets/arnaudlewandowski/mandines-real-poker-hands-mrph-dataset |
| WPN Blitz 10NL | Fast-fold cash-game hands (Winning Poker Network) | https://www.kaggle.com/datasets/viv6369/wpn-blitz-10nl |
| Poker Flop Aggregations | Aggregated flop stats | https://www.kaggle.com/datasets/chrisjackson7/5ph-aggregations |
| Andrew's Preflop Calls | Preflop decision data | https://www.kaggle.com/datasets/andrewmingwang/andrews-preflop-calls |

## 👁️ Computer vision & misc

| Dataset | Content | Link |
|---|---|---|
| Playing Cards Object Detection | Card detection images (YOLO-style CV) | https://www.kaggle.com/datasets/andy8744/playing-cards-object-detection-dataset |
| Poker Cards — Suits & Numbers | Card-image classification | https://www.kaggle.com/datasets/mehrdadkianiosh/poker-cards-suits-and-numbers |
| Gambling Behavior (Bustabit) | Player-behavior analytics (responsible-gambling research) | https://www.kaggle.com/datasets/kingabzpro/gambling-behavior-bustabit |
| Poker Datasets (brijeshbmehta) | Multi-file poker data collection | https://www.kaggle.com/datasets/brijeshbmehta/pokerdatasets |

## 🔌 Pull it yourself (API)
```bash
pip install kaggle   # kaggle.json in ~/.kaggle/
kaggle datasets list -s "poker" --sort-by votes
kaggle datasets download -d rasvob/uci-poker-hand-dataset
```

## Related in AIForge
- Poker AI Milestones & Research · Datasets & Hand Histories · Solvers & Open-Source Tools
- Parent vertical: [`../`](../) (Gaming AI)

**Keywords:** Kaggle poker, poker dataset, UCI poker hand, hand history dataset, poker machine learning, dataset de poker, histórico de mãos, poker ML.
