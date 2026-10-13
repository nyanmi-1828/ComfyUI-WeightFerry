# ComfyUI-WeightFerry

<img src="assets/logo.png" width="128" alt="WeightFerry のロゴ: モデルの層を積んで波を渡る船" align="right">

[English](README.md) | 日本語 — 作者: [nyanmi-1828](https://github.com/nyanmi-1828)

**VRAM に収まらない拡散モデルを、ComfyUI 自身の計算のまま、ComfyUI の中で動かします。**

**向いている人:** VRAM が 2〜8GB の GPU の人、または VRAM より大きいモデルを動かしたい人。モデルがすでに VRAM に収まるときはストリーミングが自動でオフになり、ほかの最適化（Anima / Cosmos-Predict2 のブロックの融合、拡大の半精度化）で少し速くなります。

* **小さい GPU で大きいモデル。** Cosmos-Predict2 **14B**（fp8）が、8GB の RTX 4060 で VRAM **約 3GB** で動きます。ComfyUI 標準で動かしたときのピーク 7GB に対して、速さはほぼ同じです。Qwen-Image **20B** は 3.3GB。7B までの LLM のテキストエンコーダーも流せます。
* **出力は同じ。** 変えるのは重みの置き場所だけです。ストリーミングのすべてのモードで、同じモデルを ComfyUI 標準で動かした結果と、同じプロセス・同じ設定で**ビット単位で一致**することを確かめています（活性ストリーミング〔小さい GPU でとても大きい画像のときに使う〕は小さなテストでだけ。それを使った低 VRAM の通しの実行は、わずかに違う〔LPIPS 約 0.008〕: [確かめ方](#確かめ方)）。任意の高速化のいくつか、Ultimate SD Upscale+ の中の拡大モデルの float16（既定）、全体への既定の変更 1 つ〔TF32〕は、画像をわずかに変えます: [このパッケージが全体に与える影響](#このパッケージが全体に与える影響)。
* **VRAM が厳しいときに速い: RTX 4060 で最大 1.7 倍（Colab の L4 では 2.9 倍）。** 次のブロックを、今のブロックの計算中に送ります。VRAM の上限 2GB で、Anima は ComfyUI の Dynamic VRAM（ComfyUI 自身の、必要に応じたオフロード）より **1.65 倍**速く、GGUF の Wan 2.1 14B は上限 4GB で ComfyUI-GGUF 単体より **1.7 倍**速く動きます（RTX 4060。Colab の L4 では 2.9 倍）。Colab の L4 で、Wan 2.1・Z-Image・Lumina 2 は上限 4GB で Dynamic VRAM と ±7% 以内で、わずかに遅いものもあります（[実測の結果](#実測の結果)）。空きの VRAM / RAM から自分で設定を決めます。
* **Anima 向け**（Cosmos-Predict2 2B を元にしたアニメ調の画像モデル）: 結果を変えずにブロックを速くする処理（融合）と、画像をわずかに変える代わりに速くする任意の機能 2 つ（TaylorSeer、CFG の早期停止）。

**仕組み:** モデルのブロックはシステムの RAM（または SSD）に置き、計算する直前に GPU へ送ります。送るのは前のブロックの計算と重ねて行います。計算そのものは ComfyUI のものです。

> 位置づけ: 1 人で進めている研究プロジェクトです。

## 動作確認済みモデル

| モデル | 確認したこと |
|---|---|
| **Anima**（Cosmos-Predict2 2B 系） | 全部のノード。VRAM 8GB と、上限 4GB / 2GB |
| Cosmos-Predict2 **14B**、Qwen-Image **20B** | ストリーミング、ビット一致（RTX 4060 8GB、RAM 32GB） |
| Wan 2.1 T2V 14B、Z-Image Turbo、Lumina Image 2.0 | ストリーミング、ビット一致（Colab L4、VRAM の上限 4GB / 2GB）。Wan は画像（1 フレーム） |
| GGUF のモデル（ComfyUI-GGUF）: Wan 2.1 14B、Cosmos-Predict2 14B | ストリーミング、ビット一致（LoRA ありも） |
| テキストエンコーダー: Qwen3 0.6B、Qwen2.5-VL 7B、T5-XXL | ストリーミング、ビット一致 |

ほかの DiT（Flux、SD3、動画）は未確認です。UNet のモデル（SDXL、SD 1.5）には対応していません。

## 必要なもの

* **GPU**: NVIDIA。Ada（RTX 40 番台、L4）で動作を確認しています。
* **ComfyUI**: 0.38.0 以降。
* **RAM**: モデルのファイルの大きさ + 数 GB が目安（Anima は 16GB で足ります）。足りないときは [最小構成](#最小構成vram-が少ないとき) の SSD のキャッシュを参照。
* **OS**: Windows 11 または Linux。
* **任意**: [SageAttention](https://github.com/thu-ml/SageAttention)、Triton（Windows では `triton-windows`。ブロックの融合に使う）。

## 入れ方

ComfyUI の中から入れる場合: **Manager**（Custom Nodes Manager）を開いて **WeightFerry** を検索し、インストールして ComfyUI を再起動します。[Comfy Registry](https://registry.comfy.org) に `comfyui-weightferry` として公開しています。

手動で入れる場合:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/nyanmi-1828/ComfyUI-WeightFerry
```

ComfyUI を再起動すると、ノードのメニューの **WeightFerry/** の下に出ます。読み込むと、いくつかの設定が全体に効きます（TF32 をオン、Anima のブロックの融合など）: [このパッケージが全体に与える影響](#このパッケージが全体に与える影響)。アンインストールはフォルダを消して再起動するだけです。

## 最小構成（VRAM が少ないとき）

1. いつも通り拡散モデルを読み込み（Load Diffusion Model / Load Checkpoint / Unet Loader (GGUF)）、続けて LoRA を読み込む。
2. その直後に **⚡ DiT Layer Streaming** を置く。設定はすべて既定値のまま（`auto_config` はオン）。

   ```
   Load Diffusion Model -> (LoRA) -> ⚡ DiT Layer Streaming -> KSampler
   ```

3. テキストエンコーダーも VRAM に収まらないときは、CLIP の読み込みの後ろに **⚡ Text Encoder Layer Streaming** を置く。
4. 生成して、コンソールを見る。モデルが収まらないときは、決めた内容が出ます:

   ```
   （Cosmos-Predict2 14B、36 ブロック、8GB の GPU での例）
   [WeightFerry] DiT Layer Streaming [auto]: {..., 'decision': 'stream 26 of 36 blocks, 10 resident', ...}
   [WeightFerry] DiT Layer Streaming: Registering Late Offload (10/36 resident, 26 streaming blocks, ...)
   ```

   収まるとき: `[WeightFerry] DiT Layer Streaming [auto]: the whole model fits in VRAM; streaming not used (ComfyUI is faster then).`。大きい画像で VRAM が足りないときは、活性ストリーミングと TaylorSeer のキャッシュの RAM 置きが自動で入り、そのこともログに出ます。VRAM が共有 GPU メモリにあふれたときは警告が出ます（[困ったとき](#困ったとき)）。

モデルを読み込んだ後の最初の生成は、数秒長くかかります（重みを RAM に一度取り込むため）。RAM が少ないときは、入りきらないブロックが ComfyUI の temp フォルダのキャッシュファイルに置かれます（速い SSD に置くには `WEIGHTFERRY_DISK_DIR`）。

## ノード

ストリーミングのノードは [動作確認済みモデル](#動作確認済みモデル) のどれにも使えます。TaylorSeer は Anima 専用、RoPE Scale は Anima / Cosmos-Predict2 専用です。

| メニュー | ノード | 内容 | 画質への影響 |
|---|---|---|---|
| WeightFerry/streaming | ⚡ DiT Layer Streaming (low VRAM) | 拡散モデルのブロックを RAM / SSD からストリーミング。モデルの読み込み（LoRA の後）の直後に置く | なし（ビット一致）。`sage_attention` をオンにすると、ComfyUI の `--use-sage-attention` と同じく SageAttention に替わる |
| WeightFerry/streaming | ⚡ Text Encoder Layer Streaming (low VRAM) | 同じ仕組みを ComfyUI 自身のテキストエンコーダー（T5、Llama / Qwen の層）に | なし（ビット一致） |
| WeightFerry/streaming | CLIP Text Encode (Streamed LLM) | テキストエンコーダーを RAM から 1 層ずつ動かす CLIP Text Encode。どのテキストエンコーダーにも使える | なし（ビット一致） |
| WeightFerry/streaming | CLIP Text Encode (Streamed LLM, from File) | Anima 専用: 同じ処理で、テキストエンコーダーのファイルを選ぶもの（`model_name`） | なし（ビット一致） |
| WeightFerry/speed | TaylorSeer | Anima 専用: 予測できると判断したステップでは、キャッシュした特徴量のテイラー展開で出力を予測し、計算を省く | **わずかにあり**: 予測したステップの分だけ変わる |
| WeightFerry/speed | CFG Early-Stop | サンプリングの前半だけ CFG をかけ、後半は CFG 1.0（負のプロンプトの計算なし） | **わずかにあり**: 最後のほうのステップで負のプロンプトが効かない |
| WeightFerry/sampling | ReAL Sampler / TORS Scheduler | ステップ幅を変えるサンプラーとスケジューラー（KSampler にも `real` / `tors` として追加） | 別のサンプラー / スケジュールなので別の画像になる（ほかを選んだときと同じ） |
| WeightFerry/sampling | Sampler Choice (shared sampler / scheduler) | 複数の KSampler で共通に使うサンプラー / スケジューラーの設定 | なし（設定を配るだけ） |
| WeightFerry/upscale | Upscale Image (using Model, half precision) | ComfyUI の拡大モデルのノードに float16 / bfloat16 の選択肢を足したもの。既定の `float32` は ComfyUI のノードと同じ | `float32` ならなし。`float16` はごくわずか（見た目は同じ） |
| WeightFerry/upscale | RoPE Scale (large images) | Anima / Cosmos-Predict2 専用: 学習時の 2 倍の大きさをタイルなしで描き直すための位置の埋め込みの調整（NTK / 補間 / YaRN 風） | あり（学習時より大きい画像のための調整） |
| WeightFerry/upscale | Ultimate SD Upscale+ (tile skip, high-pass noise) | [Ultimate SD Upscale](https://github.com/ssitu/ComfyUI_UltimateSDUpscale) を同梱し、2 つの任意の機能を足したもの: 平坦なタイルを描き直さない、タイルごとの初期ノイズの低い周波数を抑える | 既定ではごくわずか（拡大モデルが float16）。2 つの機能は仕組み上変える。元のノードと同じ出力になるのは、2 つとも 0 で float32 のときだけ（[拡大](#拡大weightferryupscale)） |
| WeightFerry/upscale | USDU Tile Calculator | 画像の縦横比から Ultimate SD Upscale のタイルの大きさ（と上の 2 つの値）を計算 | なし（値を計算するだけ） |
| WeightFerry/vae | VAE Decode (Exact Tiles, low VRAM) | Wan 2.1 の VAE（Anima、Qwen-Image）用の VAE Decode。VRAM が足りないときは継ぎ目の出ない分割、収まるときは分割なし | 分割なしならなし。分割するとわずか（[このノードの節](#vae-decode-exact-tiles-low-vramweightferryvae)） |
| WeightFerry/prototype (not adopted) | CFG Negative Cache、Coarse Global Attention、TGATE | 計測して**採用しなかった**試作。参考のために残してある。`mode` の既定が `off` で、変えるまで何もしない | あり（結果と理由: [enhancers/prototypes.py](enhancers/prototypes.py)） |

## 使い方

各ノードのつなぎ方です。表はノードの既定値です。例のワークフローで使っている値は [examples/README_ja.md](examples/README_ja.md) にあります。

### モデルのつなぎ方（Anima）

```
UNET の読み込み → LoRA → ⚡ DiT Layer Streaming → CFG Early-Stop → TaylorSeer → KSampler
```

生成の段がほかにもある（顔の仕上げ、拡大など）ときは、同じストリーミングのノードの出力から、段ごとに CFG Early-Stop と TaylorSeer を 1 組ずつ置きます。

* **モデルの精度**: 公式の bf16 のファイルはそのまま使えます。int8 ConvRot 版（[変換ツール](#変換ツールtools)で作る）が RTX 4060 では最も速く、fp8 より bf16 に近い画質です（同じシードで仕上げた画像の bf16 との LPIPS が、fp8 の 0.42 に対して 0.31）。ブロックの融合は int8 ConvRot と scaled fp8 に効きます。Cosmos-Predict2 14B でも同じで、int8 ConvRot（14.3GB、scaled fp8 と同じ大きさ）は bf16 との LPIPS が 0.030（scaled fp8 は 0.33 で、絵が目に見えて変わる）、速さは 5.30 s/it（scaled fp8 は 6.89。896x1472、30 ステップ、CFG 4、RTX 4060）です。
* **⚡ DiT Layer Streaming**: `auto_config` をオン（既定）にすると、空きの VRAM / RAM から `resident_blocks` と `tiering` を決めます。ほかの設定はどちらのときも効きます。`vram_budget_gb`（0 = なし。既定で、いちばん速い）は ComfyUI が使ってよい VRAM の上限で、ゲームなどほかの GPU のアプリと並べて動かすときに使います。自動の設定はその中で決まり、画像は変わらず速さだけが変わります（ゆるい上限です。VAE や拡大のピークは超えることがあり、そのときは一度だけ警告します）。

### CFG Early-Stop（WeightFerry/speed）

| 設定 | 既定値 |
|---|---|
| `early_stop_percent` | 0.65 |
| `decay_mode` | `hard_stop` |

`linear_decay` は、しきい値で一気に 1.0 にせず、`early_stop_percent` の時点の元の値から最後のステップの 1.0 まで少しずつ下げます。CFG を最後まで効かせたいときは、このノードをオフ（Ctrl+B）。本体・顔の仕上げ・Ultimate SD Upscale+ のワークフロー（int8 ConvRot、RTX 4060）で測ると、オフにすると（段ごとの CFG Early-Stop をすべて）1 枚が約 27% 遅くなります。

### TaylorSeer（WeightFerry/speed）

| 設定 | 既定値 |
|---|---|
| `mode`（`adaptive` か `step`） | `adaptive` |
| `rel_l1_thresh`（`adaptive`） | 0.08 |
| `step_interval`（`step`） | 2 |
| `max_order` | 1 |
| `first_enhance`（最初に必ず計算するステップ数） | 2 |
| `last_enhance` | 1 |
| `max_predict_steps`（0 = 上限なし） | 0 |
| `cache_device` | `cuda` |

`rel_l1_thresh` を下げると計算するステップが増えます（TaylorSeer なしに近づき、遅くなる）。オフ（段ごとの TaylorSeer をすべて Ctrl+B）にすると、同じ計測で 1 枚が約 30% 遅くなります。

### サンプラーとスケジューラー（WeightFerry/sampling）

KSampler で `real` / `tors` を、ほかのサンプラー / スケジューラーと同じように選べます。**Sampler Choice** は 1 組のサンプラー / スケジューラーを複数の KSampler（本体・顔の仕上げ・Ultimate SD Upscale）に配り、まとめて切り替えられるようにします。

### 拡大（WeightFerry/upscale）

* **Ultimate SD Upscale**: **Ultimate SD Upscale+** を使います。Ultimate SD Upscale を同梱したものです。本家の ComfyUI_UltimateSDUpscale を別に入れる必要はなく、入っていてもそれぞれ別に動きます。
  * **USDU Tile Calculator** をつなぎます。入力は画像と `upscale_by`（USDU の倍率）、出力の `tile_width`・`tile_height`・`tile_padding`・`mask_blur`・`tile_skip_threshold`・`highpass_noise_strength` を USDU の同じ名前の入力へ。画像の縦横比からタイルの大きさを決めます。`tile_factor`（既定 4）はタイルの枚数の目安です。
  * Tile Calculator は `tile_skip_threshold` 1.0（空や無地の壁など平坦なタイルを描き直さない）と `highpass_noise_strength` 0.6（タイル間の色のずれを抑える）も渡すので、つなぐと 2 つともオンになります。0 でそれぞれオフ。
  * 拡大モデルは float16 で動かします（RTX 4060、4x-UltraSharp で 13.0 → 5.8 秒。最終画像はわずかに変わる、LPIPS 0.012）。`WEIGHTFERRY_USDU_UPSCALE_DTYPE=float32` で Ultimate SD Upscale と同じ float32 になります。2 つとも 0 で float32 なら、出力は元のノードとバイト単位で同じです。
  * VRAM が少ないときはこちらが速い拡大です: 上限 4GB / 2GB で 1 枚 52 / 65 秒（タイルなしは 61 / 74 秒、RTX 4060）。
* **タイルなし**（画像全体を 1 回で描き直す）: **Upscale Image (using Model)** → 縮小 → VAE Encode → KSampler。モデルは **RoPE Scale**（`mode` を `ntk` に。既定は `off`。2 倍なら `factor` 2.0）を通して、大きくしても学習時の位置の範囲に収めます。
* **Upscale Image (using Model)** 単体: `float32` は ComfyUI のノードと同じ。`float16` は約 2.2〜2.6 倍速く、見た目は同じです。
* ESRGAN 系のモデル（4x-UltraSharp など）では、このパッケージの拡大（このノードと Ultimate SD Upscale+ の拡大）は、モデルの密結合ブロックを追記用のバッファとまとめたカーネルで動かし、float16 では最後の段を出力 256 行ずつの帯に分けて計算します。512 のタイル 1 枚の VRAM のピークが下がり（プロセスのメモリで 2304 → 768MB）、出力はビット単位で同じです。これは NVIDIA と cuDNN（カーネルは Triton も）の場合で、それ以外では従来どおりに動きます（[docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) の `WEIGHTFERRY_UPSCALE_*`）。

### VAE Decode (Exact Tiles, low VRAM)（WeightFerry/vae）

VRAM が足りないとき（大きい画像、小さい GPU）に、Wan 2.1 の VAE（Anima、Qwen-Image）の VAE Decode / VAE Decode (Tiled) の代わりに使います。例のワークフローは、本体の画像に ComfyUI の VAE Decode、タイルなしの拡大に VAE Decode (Tiled) を使っています。そこで VRAM が足りなくなったときに、このノードに替えます。`mode` は `auto`: 空き VRAM に収まれば分割なし（VAE Decode と同じバイト）、収まらなければ `stage 2`（ピークが半分、ほぼ同じ画像）、さらに `stage 1` / `stage 0`（メモリはずっと少なく、丸めの差がもう少し出る。82% の画素が一致、差は最大 6 階調）。ComfyUI の分割 decode と違って継ぎ目は出ません。decoder の局所的な層だけを、受容野ちょうどの重なりで分割するためです。RTX 4060 の 1792×2944 で、分割なしのピーク 6.3GB、`stage 2` 3.1GB（99.7% の画素が一致、差は最大 1 階調）、`stage 0` 0.7GB。`tile_size` は 512 で十分です。ほかの VAE や動画の latent では、自動で VAE Decode と同じ処理になります。

### テキストエンコーダー（WeightFerry/streaming）

* **⚡ Text Encoder Layer Streaming**: VRAM に収まらないテキストエンコーダー（T5、Llama / Qwen）の CLIP の読み込みの後ろに置き、`auto_config` をオン。
* **CLIP Text Encode (Streamed LLM)**: CLIP Text Encode の代わりに使います（入力も同じ `clip`・`text`）。VRAM より大きい、どのテキストエンコーダーにも使え、条件付けはビット単位で同じです。Anima のテキストエンコーダー（Qwen3 0.6B）はこのパッケージの streamed LLM のエンジンで動かし（エンジン単体を Qwen2.5-VL 7B で測ると、ComfyUI のコードの約 1.7〜1.9 倍速い）、それ以外は ⚡ Text Encoder Layer Streaming と同じように、層を流しながら ComfyUI 自身のエンコードで動かします。例のワークフローのプロンプトはこれでエンコードしています。Anima の 0.6B は VRAM に収まりますが、結果は同じで、大きいテキストエンコーダーに替えるときもそのまま使えるためです。
  * `disk_cache`（既定はオフ）: 結果をディスクに残し、同じテキストと同じ CLIP なら、再起動の後も使い回します。キャッシュにあれば CLIP の入力を要求しないので、ほかに CLIP を使うノード（CLIP をつないだ顔の仕上げなど）がなければ、テキストエンコーダーを読み込みません（読み込みの時間も RAM も要らない）。キーは、テキスト、CLIP を作るノード（入力とモデルのファイル）、ComfyUI・PyTorch・GPU の版です。読み出した条件付けはビット単位で同じです。フォルダは `<ComfyUI のベース>/.cache/weightferry/te_cache`、上限 1GB（`WEIGHTFERRY_TE_CACHE_GB`）。
* **CLIP Text Encode (Streamed LLM, from File)**: Anima 専用。上と同じ処理で、`model_name` の入力があるもの。`(from clip)`（既定）なら CLIP の読み込みの重みを使い、ファイルを選ぶとそのファイルから読みます（量子化したテキストエンコーダーのとき。トークン化のために CLIP の入力は必要）。

## 例のワークフロー

[examples/](examples/) に、Anima のワークフローが 2 つあります。

* `anima_txt2img.json` — ベース生成 → 2 倍拡大（Ultimate SD Upscale またはタイルなし、任意）。ストリーミング、TaylorSeer、CFG の早期停止、モデルの精度の切り替え入り。
* `anima_draft_turbo.json` — Turbo LoRA で速い下書き（6 ステップ、CFG 1）。

動かす前に:

1. **カスタムノード**: このパッケージのほかに使うのは [comfyui-easy-use](https://github.com/yolain/ComfyUI-Easy-Use)（LoRA のスタック、切り替え）だけです。ワークフローを開いて、Manager の **Install Missing Custom Nodes** で見つかります。ComfyUI を再起動して、ワークフローを開き直します。
2. **モデルのファイル**: [circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima) の公式のファイル、ESRGAN 系の 4 倍の拡大モデル、（下書きだけ）Turbo LoRA。ファイル名・置き場所・入手先は [examples/README_ja.md](examples/README_ja.md#モデルのファイル) にあります。
3. **モデルの切り替え**: 最初は `1`（bf16、公式のファイル）です。任意で、より速くするには fp8 / int8 ConvRot のファイルを作り（作り方は [examples/README_ja.md](examples/README_ja.md#モデルのファイル)）、その読み込みノードのバイパスを外して（Ctrl+B）、切り替えを `0`（fp8）か `2`（int8 ConvRot。`anima_txt2img.json` だけ）にします。
4. **2 倍拡大**: 最初はオフです。`anima_txt2img.json` では、「Upscale method」を `1`（タイルなし）にするか、`0` のまま「USDU 2x」のバイパスを外します（Ctrl+B。Ultimate SD Upscale。VRAM が少ないときはこちらが速い）。下書きにあるのは Ultimate SD Upscale のノードだけです（Turbo との関係は [examples/README_ja.md](examples/README_ja.md)）。

## 実測の結果

断りがなければ RTX 4060 8GB（PCIe 4.0 x8、RAM 32GB）。「L4」は Colab の L4 の VM。画像の大きさは 896x1472（Wan は 832x480 の 1 フレーム、Z-Image / Lumina 2 は 1024x1024）。s/it は 1 ステップの秒数で、小さいほど速い。2 回目（ウォームアップ後）を計測。「VRAM の上限 N GB」は、補助のプロセスで残りの GPU メモリを埋めるか、PyTorch のプロセスごとのメモリの割合を設定して、ComfyUI と WeightFerry に同じ N GB だけを使わせた条件です。ComfyUI の起動引数は [コマンドラインの一覧](#comfyui-の起動引数)、「ビット一致」の意味は [確かめ方](#確かめ方) を参照。

| 条件 | WeightFerry | 比べた相手 | 出力 |
|---|---|---|---|
| Cosmos-Predict2 14B、fp8 | 6.56 s/it、ピーク VRAM **2.90GB** | ComfyUI 標準: 6.91 s/it、7.01GB | ビット一致 |
| Cosmos-Predict2 14B、int8 ConvRot（変換ツールで作る）、30 ステップ、CFG 4 | 5.30 s/it。bf16 との LPIPS 0.030 | scaled fp8: 6.89 s/it、LPIPS 0.33 | ビット一致（LoRA ありも） |
| Cosmos-Predict2 14B、bf16（ファイル 28.5GB）、RAM 32GB | 8.87 s/it（SSD 層） | ComfyUI 標準: 実行せず（重み 28.5GB、RAM 32GB） | ブロックごとにビット一致（各ブロックの最初の forward。生成全体は比べる相手がない） |
| Qwen-Image 20B | 6.21 s/it、ピーク VRAM 3.30GB | ComfyUI 標準: 6.54 s/it | ビット一致 |
| Anima 2B、VRAM の上限 2GB | 2.33 s/it | ComfyUI の Dynamic VRAM: 3.86 s/it（**1.65 倍**速い） | ビット一致 |
| Cosmos-Predict2 14B + LoRA、上限 4GB（L4） | 2.42 s/it | Dynamic VRAM: 3.78 s/it（1.56 倍） | ビット一致 |
| Wan 2.1 14B（fp8）/ Z-Image Turbo / Lumina 2、上限 4GB（L4） | 1.29 / 1.18 / 0.57 s/it | Dynamic VRAM: 1.21 / 1.26 / 0.55 s/it | ビット一致 |
| Wan 2.1 14B の GGUF Q8_0、上限 4GB | 2.35 s/it（L4: 1.46） | ComfyUI-GGUF 単体: 4.04 s/it、1.7 倍（L4: 4.23、2.9 倍） | ビット一致 |
| Anima 2B、モデルが収まる（上限なし）、ストリーミングを強制（`auto_config` ならここでは流さない） | ストリーミングなしの 0.94 倍の速さ、VRAM 3.70 → 3.04GB | ストリーミングなしの ComfyUI | ビット一致 |
| Qwen2.5-VL 7B のテキストエンコーダー、streamed LLM のエンジン（エンジン単体） | 約 1.7〜1.9 倍速い、VRAM +2.0GB | 同じトークンでの ComfyUI 自身のテキストエンコーダーのコード | ビット一致 |
| RAM が少ない: RAM の予算 2GB で SSD 層 | RAM だけのときの 0.91 倍の速さ、ページ固定のメモリは 660MB 少ない | 全ブロックを RAM | ビット一致 |
| Anima のタイルなし 2 倍拡大、GPU 4GB / 2GB | 1 枚 61.0 / 74.3 秒（`auto_config` のまま、手での設定なし） | 同じワークフローを上限なしで: 51.5 秒 | 上限なしと LPIPS 約 0.008（見た目は同じ） |
| Anima のブロックの融合（int8 ConvRot、CFG 5） | 1 ブロック 22.1 ms | ComfyUI のブロック: 30.3 ms（1.37 倍。交互に計測した中央値） | ビット一致 |
| Anima の GGUF Q5_1（ComfyUI-GGUF）、モデルが収まる | 1 本のカーネルでの展開で 1.67 s/it | ComfyUI-GGUF の展開: 2.37 s/it（1.42 倍。Q4_1 は 1.19 倍、Q8_0 は 1.03 倍） | ビット一致 |
| Anima の GGUF Q5_1 + LoRA、モデルが収まる / 上限 4GB（どちらも 1 本のカーネルでの展開） | LoRA の係数を固定 RAM から送って 1.68 / 0.97 s/it | 係数を通常の RAM から送る（ComfyUI-GGUF の置き方のまま）: 1.89 / 1.18 s/it（1.12 倍 / 1.22 倍） | ビット一致 |
| Wan 2.1 14B の GGUF Q4_K_M、上限 4GB | K-quant のカーネルで 1.91 s/it | ComfyUI-GGUF の K-quant の展開: 2.81 s/it（1.47 倍。同じプロセスで交互に計測） | ビット一致 |
| Anima 1792x2944 をトークンの行ごとに計算（タイルなしの 2 倍拡大）、上限 4GB | SageAttention の K / V の準備をブロックごとに 1 回で 2.39 s/it | 行の塊ごとに準備: 3.2 s/it（1.34 倍） | ビット一致 |

## このパッケージが全体に与える影響

読み込むだけで、自分のノード以外にも次の影響があります。

* PyTorch の TF32（行列積・cuDNN）をオン（Ampere / Ada で速い。厳密な float32 とは結果がわずかに変わる）。cuDNN の自動調整は ComfyUI に任せる（ComfyUI の `--fast autotune` でオン）。
* Windows: ComfyUI の実行中、システムのタイマーの分解能を 1ms に。
* `TRITON_CACHE_DIR` を `<ComfyUI のベースディレクトリ>/.cache/triton` に（自分で設定していれば、そちらを優先）。
* 28 ブロック用の Anima の LoRA を Anima 2.9B（40 ブロックの版）に当てるためのフック（`WEIGHTFERRY_LORA_REMAP=0` でオフ）。
* KSampler の一覧に、サンプラー `real` とスケジューラー `tors` を追加。
* ComfyUI の Cosmos-Predict2 / Anima の `Block`・`Attention`・`GPT2FeedForward` のクラスに、要素ごとの演算をまとめた処理を入れる（結果はビット単位で同じ。`WEIGHTFERRY_FUSED_BLOCK=0` で無効）。
* ComfyUI-GGUF が入っていれば、その展開（`ops.dequantize_tensor`・`GGMLLayer.get_weight`）を、重みごとに 1 本の Triton のカーネルに置き換える（Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_0 と K-quant の Q2_K〜Q6_K）。LoRA は comfy 自身の `torch.mm` のあと同じカーネルの中で足し、LoRA の係数は固定 RAM から送る。結果はビット単位で同じ（`WEIGHTFERRY_GGUF_TRITON=0` ですべて無効。細かいスイッチは [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md)）。LoRA の係数を固定 RAM に置いた時点から、外した LoRA の固定 RAM を返すため、ComfyUI の `free_memory`・`unload_all_models`・`soft_empty_cache` を包む。
* ストリーミングのノードの `sage_attention` / `sage_pv_fp16` は、アテンションを全体で差し替え、元に戻しません。
* VRAM の上限（`vram_budget_gb` または `WEIGHTFERRY_VRAM_BUDGET_GB`）を付けると、ComfyUI の予約（`--reserve-vram` で決まる量）と Dynamic VRAM の余裕を引き上げます。上限なしの WeightFerry のノードが動くと元に戻ります。

## 困ったとき

* **「... MB of this process' GPU memory is in shared GPU memory (system RAM) ...」**（Windows）: VRAM がシステムの RAM にあふれていて、生成がとても遅くなります。`auto_config` は次の読み込みからその分の VRAM を空けておきます。ほかの GPU を使うアプリを閉じるのも効きます。
* **「Anima fused block: Triton not available ...」**: ブロックの融合のうち Triton を使う部分（`mod`・`modq`・`gelu`・`rms`・`ln`・`tay`）がオフになり、ほかの部分はそのまま効きます（画像は同じで、少し遅いだけ）。使うには Triton（Windows では `triton-windows`）を入れてください。
* **Wan で「required rank 4 tensor to use channels_last format」**: `--force-channels-last` を付けずに ComfyUI を起動してください（Wan の 5 次元の重みでの ComfyUI 側の制限）。
* **例のワークフローで `anima-base-v1.0_fp8_scaled.safetensors` / `..._int8_convrot.safetensors` が見つからない**: ファイルを作るか（[例のワークフロー](#例のワークフロー) の 3）、その読み込みノードを再びバイパス（Ctrl+B）して切り替えを `1`（bf16）に戻してください。
* **ComfyUI 本来の動作に戻したい**: ストリーミングのノードをバイパス（Ctrl+B）。[全体への変更](#このパッケージが全体に与える影響) のうち、切れるのはブロックの融合（`WEIGHTFERRY_FUSED_BLOCK=0`）、GGUF の展開（`WEIGHTFERRY_GGUF_TRITON=0`）、LoRA の読み替え（`WEIGHTFERRY_LORA_REMAP=0`）です。TF32、1ms のタイマー、Triton のキャッシュの場所（`TRITON_CACHE_DIR` を自分で設定していなければ）、`real` / `tors` の追加は、パッケージを入れている間はそのままです。SageAttention の差し替えは、一度使うと ComfyUI を再起動するまで残ります。

## コマンドラインの一覧

### ComfyUI の起動引数

必須のものはありません。この README の RTX 4060 での計測は、次の ComfyUI の起動引数で行いました（Wan は `--force-channels-last` なし。[困ったとき](#困ったとき) を参照）:
`--cuda-malloc --force-channels-last --bf16-unet --bf16-vae --fp32-text-enc --use-sage-attention --enable-triton-backend`

### 変換ツール（[tools/](tools/)）

ComfyUI のフォルダで、ComfyUI の Python で実行します（例: `python custom_nodes/ComfyUI-WeightFerry/tools/convert_to_fp8.py models/diffusion_models/MODEL.safetensors --scaled`）。

| スクリプト | 引数 | 出力 |
|---|---|---|
| `convert_to_fp8.py SRC` | `--scaled`（テンソルごとのスケール付き、ComfyUI の comfy_quant 形式。おすすめ）、`--keep-vanishing`（`--scaled` なしのときだけ: fp8 で行がすべて 0 になるテンソルも残す）、`--out ファイル` | fp8 のモデル。weight dtype は `default` で読み込む。既定の名前は `<src>_fp8_scaled.safetensors`（`--scaled`）または `<src>_fp8_e4m3fn.safetensors` |
| `convert_quant.py SRC --like FP8_SCALED --format 形式` | `--format` は `int8_convrot`（おすすめ）・`int8`・`int8_channel`・`w4a4`。`--like` は同じモデルのスケール付き fp8（そこで量子化されている層を変換する）。`--out ファイル`。`--comfy-root フォルダ`（ComfyUI のコードが見つからないとき。デスクトップ版など） | ComfyUI 自身の量子化の形式。CUDA の GPU が必要。1 テンソルずつ書くので、変換後のモデルが RAM に収まらなくてもよい。既定の名前は `<src>_<形式>.safetensors` |
| `convert_to_gguf.py SRC` | `--type` は `Q8_0`（既定）・`Q5_1`・`Q5_0`・`Q4_1`・`Q4_0`・`BF16`。`--out ファイル` | ComfyUI-GGUF 用の GGUF（Anima / Cosmos-Predict2 の DiT。`pip install gguf` が必要）。既定の名前は `<src>-<type>.gguf`。「Unet Loader (GGUF)」で読み込み、その後ろにストリーミングのノードを置ける |

fp8 の変換はブロックごとの重みを連続した範囲に書くので、ストリーミングのノードがファイルから直接ブロックを読めます。

### 環境変数

`WEIGHTFERRY_*` の 75 個（どれも任意。調整・低 VRAM のスイッチ・デバッグ・既定でオンのものを切るスイッチ）。既定値と根拠の計測は [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md)（英語）。`ComfyUI/user/weightferry.env` に 1 行 1 つ `名前=値` で書いても効きます（本物の環境変数が優先。Comfy Desktop で便利）。

## 制限

* 速さは PCIe と RAM の帯域しだいです。
* モデルを小さくしたり、重みを変えたりはしません（tools/ の変換ツールは、実行したときにだけ変換します）。

## 確かめ方

ストリーミングのすべてのモード（ブロック単位 / チャンク単位、RAM / SSD 層、スケール付き fp8、LoRA）を、同じモデルを ComfyUI 標準で計算した結果と、同じプロセス・同じ設定で比べています。比べるのは、ブロックごとに使われた重み、各ブロックの最初の forward、生成全体で、すべてビット単位です。活性ストリーミングは、小さな Cosmos-Predict2 のテストでビット単位の一致を確かめています。Anima のタイルなし 2 倍拡大を上限 4GB / 2GB で通しで作ると（活性ストリーミングと TaylorSeer のキャッシュの RAM 置きが入る）、上限なしの画像と LPIPS 約 0.008 の差が出ます（見た目は同じ。原因はまだ突き止めていません）。仕組み上画像を変える TaylorSeer と CFG の早期停止は、並べた比較（PSNR / SSIM / LPIPS と目視）で設定を決めました。

## テスト

モデルのファイルなしで動く確認です（約 1 分、空き 2GB ほどの NVIDIA の GPU）。

```bash
python custom_nodes/ComfyUI-WeightFerry/tests/test_streaming.py
```

引数: `--comfy-root フォルダ`（ComfyUI のコードが見つからないとき、そのフォルダ。既定は `$COMFYUI_ROOT` か custom_nodes の 1 つ上）、`--legacy`（Dynamic VRAM ではなく ComfyUI の従来の ModelPatcher で。こちらはモデル全体を先に GPU に載せるので、SSD 層のケースは飛ばします）。小さな乱数の Cosmos-Predict2 の DiT をメモリ上に作り、ブロック単位・スロット数を減らした場合・常駐ブロックの自前管理・チャンク単位・SSD 層のそれぞれで、ComfyUI 標準のサンプリングとビット単位で同じ結果になるかを確かめます。

## 参考文献

* TaylorSeer — Liu ほか、ICCV 2025、[arXiv:2503.06923](https://arxiv.org/abs/2503.06923)
* ReAL サンプラー — Yin ほか、2026、[arXiv:2609.33202](https://arxiv.org/abs/2609.33202)
* TORS — Zhou ほか、ECCV 2026、[arXiv:2603.00763](https://arxiv.org/abs/2603.00763)。ここの `tors` スケジューラーは、論文の式に基づいてあらかじめ計算したシグマの表を使います（公式の配布物からの写しではありません）。表があるのは 4〜12・14・15・16・20・25・30 ステップで、それ以外のステップ数は 30 ステップの表から補間します。
* TGATE（試作）— [arXiv:2404.02747](https://arxiv.org/abs/2404.02747)
* ComfyUI、NVIDIA Cosmos、ComfyUI-Anima-TeaCache、TaylorSeer、ComfyUI-Anima-2.9B-loraPatch、ComfyUI_UltimateSDUpscale（Coyote-A の Ultimate SD Upscale のスクリプトを含む）、ComfyUI-GGUF のコードを元にした部分: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## ライセンス

パッケージ**全体**は **GNU GPL v3** で配布します（[LICENSE](LICENSE)）。GPL-3.0 のプロジェクトを元にしたファイルが入っているためです。その中で:

* ほかのプロジェクトを元にしたファイルは、元のライセンスのままです（ComfyUI / TaylorSeer / loraPatch / Ultimate SD Upscale〔ssitu・Coyote-A〕由来は GPL-3.0、NVIDIA Cosmos / ComfyUI-GGUF 由来は Apache-2.0。ComfyUI-Anima-TeaCache 由来の部分〔Apache-2.0〕は GPL-3.0 の TaylorSeer のファイルの中にある）。どのライセンスかは各ファイルの 1 行目に書いてあります。
* このプロジェクトのコード: **MIT-0** — それだけを取り出せば、何に使っても構いません。条件はありません。役に立ったら、このプロジェクトのことを書いてもらえるとうれしいですが、義務ではありません。
* 例のワークフロー（[examples/](examples/)）とロゴ（[assets/](assets/)）も **MIT-0** です。

詳しくは [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
