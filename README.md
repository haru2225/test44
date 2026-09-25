# test44 — single-phase SiO2 crystal ablation of test40

The original single-phase ablation now uses a periodic Bloch-wave EGNN,
adapted from `mila-iqia/diffusion_for_multi_scale_molecular_dynamics`.
`egnn_vendor.py` contains E_GCL/EGNN; `test44.py:Score` supplies the periodic
embedding and score projection. Provenance and MIT notice are in the vendor
module and `licenses/diffusion_for_multi_scale_molecular_dynamics-MIT.txt`.

## Architecture and scope

`egnn_vendor.py` now vendors the upstream `models/egnn.py` implementation
from `mila-iqia/diffusion_for_multi_scale_molecular_dynamics` (commit noted
in the module). `test44.py:Score` follows its `EGNNScoreNetwork` wrapper:
source first cubic Bloch shell (3 reciprocal vectors / 6 coordinates),
interleaved cos/sin uplift, sigma plus atom-type one-hot node inputs,
directed fully connected minimum-image graph, matching Gamma projection,
and sigma-normalized score convention. The SiO4 bead is the sole real atom
class; the MASK channel is retained for source-compatible input dimensions.
The architecture defaults match the source EGNN experiment template:
4 graph layers and 256-wide message/node/coordinate MLPs with four hidden
MLP layers. `WIDTH`, `LAYERS`, and `HIDDEN_LAYERS` can be overridden.

The Cartesian `sigma * score` convention is used consistently in the target
and VE-SDE reverse update. The optimizer uses AdamW with weight decay
`5e-8`, matching the source experiment configuration; learning rate remains
test44’s `2e-4` so the weight-decay comparison changes one setting at a time.
Earlier test44 checkpoints used the opposite target sign, divided the projection
by cell lengths, used a 13-vector grid and a 6 A cutoff graph. New runs use
`results/egnn_source_wd5e8`; older checkpoint formats are rejected. The original source supports fully connected edges
by default, and this test uses that path. For 64 beads this is 4032 directed
edges per frame; minimum-image distances are computed from the noisy positions.

This reproduces the source EGNN wrapper design for the test44 coarse-grained
single-species task. The dataset, wrapped-Gaussian target implementation,
training loop, and output structure remain test44-specific, so it is not an
identical end-to-end reproduction of the paper's atomistic experiments.

## Data

`examples/crystal/`: 62 frames, extracted the same way as test40's crystal
data (`md/traj_0.lammpstrj`/`traj_1.lammpstrj`, β-cristobalite NPT MD,
`--index 1000::300 --split-gap 2`). `md/silica_beta_cristobalite.in` +
init data are bundled so a fresh `STAGE=lammps`/`full` run can regenerate or
extend it; see `licenses/ScoreMD-MIT.txt` for that source's provenance.

## スパコンでの実行（PBS / GPU 1台）

既存の環境では `cd test44 && git pull --ff-only` で更新します。
新規の場合は以下で取得してください。

```bash
git clone git@github.com:haru2225/test44.git
cd test44
```

サイト指定の Singularity または Apptainer モジュールをロードしてください。
初回のみコンテナをビルドします（ビルド可能なLinux環境とネット接続が必要）。
既存の同じ `Singularity.def` で作ったイメージは再利用できます。
コードは実行時にマウントするため、EGNN更新だけなら再ビルド不要です。

```bash
singularity build test44.sif Singularity.def
# Apptainerの場合: apptainer build test44.sif Singularity.def
```

`run_test44.pbs` は既存の `sg8` キュー、GPU 1台、CPU 8、32 GB、20時間の設定です。
使用施設に合わせてキュー名・課題番号を変更してください。
以下の `PROJECT_ID` は実際の課題番号に置き換えます。

```bash
# GPUで実データのforward/backwardを確認し、CPUテストを実行
qsub -P PROJECT_ID -l walltime=00:10:00 -v STAGE=check run_test44.pbs

# 短い学習確認（完了してから次の生成確認へ）
qsub -P PROJECT_ID -l walltime=00:10:00 -v UPDATES=2,WIDTH=16,LAYERS=2,BATCH_SIZE=1,TRAIN_DIR=results/egnn_smoke,TIME_BUDGET_HOURS=0.1 run_test44.pbs
qsub -P PROJECT_ID -l walltime=00:10:00 -v STAGE=generate,STEPS=20,TRAIN_DIR=results/egnn_smoke,GENERATED_DIR=results/egnn_smoke_sample,TIME_BUDGET_HOURS=0.1 run_test44.pbs

# 比較用に新しい出力先を指定して学習（weight decay=5e-8）
qsub -P PROJECT_ID -v TRAIN_DIR=results/egnn_source_wd5e8 run_test44.pbs

# 中断後の再開。同じ幅・層数・ノイズ設定などを使用すること
qsub -P PROJECT_ID -v RESUME=1,UPDATES=30000,WIDTH=256,LAYERS=4,HIDDEN_LAYERS=4,WEIGHT_DECAY=5e-8,TRAIN_DIR=results/egnn_source_wd5e8 run_test44.pbs

# 学習完了後に同じcheckpointから生成
qsub -P PROJECT_ID -v STAGE=generate,TRAIN_DIR=results/egnn_source_wd5e8,GENERATED_DIR=results/egnn_source_wd5e8_sample run_test44.pbs
```

ジョブは自動的には順番待ちしません。学習ログの完了を確認してから対応する
生成ジョブを投入してください。途中保存で終了した場合の終了コードは75です。
短時間確認で生成品質は評価できません。本学習の損失と構造指標を確認してください。
旧チェックポイント のチェックポイントは再利用できません。

主な環境変数: `TRAIN_DIR`, `GENERATED_DIR`, `DATASET_PATH`, `SIF_IMAGE`,
`WIDTH`, `LAYERS`, `HIDDEN_LAYERS`, `BATCH_SIZE`, `UPDATES`, `SIGMA_MIN`,
`SIGMA_MAX`, `LEARNING_RATE`, `WEIGHT_DECAY`, `SEED`。外部ディレクトリを使う場合は
`EXTRA_BIND=/absolute/path:/absolute/path` も指定します。
walltimeを変更した場合は `WALLTIME_HOURS` または `TIME_BUDGET_HOURS` も合わせます。

生成結果は `final.extxyz`, `final.data`, `positions.npy` に保存されます。
評価はCPUで実行できます（施設の計算ノード利用方針に従ってください）。

```bash
singularity exec --bind "$PWD:$PWD" --pwd "$PWD" test44.sif \
  python test44.py evaluate --dataset examples/crystal \
  --sample results/egnn_source_seed1337/final.extxyz --output results/egnn_evaluation.json
```

## Local checks

```bash
python -m venv .venv
. .venv/bin/activate
pip install torch==2.6.0
pip install -r requirements.txt
python -m pytest -q
```

The test suite covers periodic geometry, permutation behavior, score-target
conventions, gradients, and the prepare/train/generate/evaluate workflow.

## test44 (test43 との差分)

- 学習ノイズ: `--sigma-max` 既定 1.5 Å（最近接ビーズ間距離 2.96 Å の約50%。Yang & Schwalbe-Koda の
  「学習最大ノイズ ≈ 最近接距離の約50%」に倣う）。test43 は 8 Å。
- 生成: 一様乱数から開始し、各ステップに追加ノイズ（`--extra-noise`, 既定 2.0 Å, PBS では `EXTRA_NOISE`）を
  加える。ノイズは最初の 97% のステップで線形に 0 まで減らし、最後の 3% はノイズなし
  （`--extra-noise-final-fraction`）。既定ステップ数は 3000。`--extra-noise 0` で test43 と同じ生成。
  論文の 1.0 Å は学習最大 0.75 Å の約1.33倍の経験値で、2.0 Å はそれをビーズ間距離に換算した初期値。
  厳密な再現ではなく、生成品質を見て調整すること。
- チェックポイント形式 `test44-crystal-source-egnn-v1`。
