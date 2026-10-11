# 例のワークフロー

[English](README.md) | 日本語

Anima のワークフローが 2 つあります（公式の Anima の base モデル、空の LoRA スタック）。ComfyUI の画面にドラッグして開いてください。読み込みのノードは [circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima) のファイル名になっています（自分のモデルを選んでも構いません）。

| ファイル | 内容 |
|---|---|
| `anima_txt2img.json` | 本体（保存）→ 2 倍拡大（任意）→ 保存。切り替え: モデルの精度（「Model switch」。1 = bf16〔公式のファイル、既定〕、0 = fp8 / 2 = int8 ConvRot はファイルを作ってから）、拡大の方式（0 = Ultimate SD Upscale〔最初はバイパス = 拡大なし〕、1 = タイルなしで描き直し。切り替えの名前の「fast」は VRAM が足りるときの話で、VRAM が少ないときは Ultimate SD Upscale のほうが速い）。TaylorSeer と CFG の早期停止（タイトルが「⚡」で始まる）は Ctrl+B でオフ = 画質優先。 |
| `anima_draft_turbo.json` | 速い下書き: Turbo LoRA 0.8、6 ステップ、CFG 1、その後に色合わせ。Ultimate SD Upscale はバイパス（Ctrl+B で使う）。モデルの切り替え: 1 = bf16（既定）/ 0 = fp8。「⚡ Speed: Turbo」のグループの左上（上の段の右端）にある Int「⚡ Turbo: 0 = off / 1 = on」で、Turbo LoRA（すべての段）・本体のステップ数（6 / 16）・本体の CFG（1 / 5）がまとめて切り替わります。Ultimate SD Upscale は 16 ステップ・CFG 5 のままなので、使うときは Turbo を 0 にしてください。Turbo がオンのときは、⚡ の TaylorSeer（`first_enhance` 8 が 6 ステップより多い）と CFG の早期停止（CFG 1）は何もしません。 |

## プロンプト

プロンプトは、一般的なタグで描いたオリジナルの人物です。「Prompt」のノードを自分のプロンプトに書き換えてください。拡大も同じプロンプトを使います。負のプロンプトは別のノード（「Negative」）です。

## 必要なカスタムノード

このパッケージと ComfyUI 自身のノードのほかに、次の 1 つのパックを使います（Manager の「Install Missing Custom Nodes」で見つかります）。

| ノード | パック |
|---|---|
| `easy loraStack`、`easy loraStackApply`、`easy anythingIndexSwitch`、`easy imageColorMatch`（下書きだけ） | [comfyui-easy-use](https://github.com/yolain/ComfyUI-Easy-Use) |

## モデルのファイル

読み込みのノードは、公式の Anima（[circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima) の `split_files/`）のファイル名です。base モデルの fp8 / int8 版は任意です（このパッケージのツールで作ります）。その読み込みノードはバイパスした状態で保存してあるので、ファイルが無くても止まりません。

| 読み込み | ワークフローのファイル | 使うもの |
|---|---|---|
| UNET（index 1、bf16、既定） | `anima-base-v1.0.safetensors` | 公式の base モデル（`split_files/diffusion_models/`） |
| UNET（index 0、fp8。任意、バイパス中） | `anima-base-v1.0_fp8_scaled.safetensors` | base の fp8（スケール付き）版。ComfyUI のフォルダで `python custom_nodes/ComfyUI-WeightFerry/tools/convert_to_fp8.py models/diffusion_models/anima-base-v1.0.safetensors --scaled`（元のファイルの隣に書きます）。作ったら読み込みのバイパスを外し（Ctrl+B）、切り替えを 0 に |
| UNET（index 2、int8 ConvRot。任意、バイパス中。txt2img だけ） | `anima-base-v1.0_int8_convrot.safetensors` | ComfyUI の int8（ConvRot）形式の base。fp8 より bf16 に近く、このパッケージのブロックの融合と合わせると RTX 4060 では最も速い。先に fp8 を作ってから、ComfyUI のフォルダで `python custom_nodes/ComfyUI-WeightFerry/tools/convert_quant.py models/diffusion_models/anima-base-v1.0.safetensors --like models/diffusion_models/anima-base-v1.0_fp8_scaled.safetensors --format int8_convrot`。バイパスを外して切り替えを 2 に |
| テキストエンコーダー | `qwen_3_06b_base.safetensors`（種類 `qwen_image`） | Anima のテキストエンコーダー（Qwen3 0.6B、`split_files/text_encoders/`）。プロンプトは、この読み込みの重みを使う CLIP Text Encode (Streamed LLM) でエンコード |
| VAE | `qwen_image_vae.safetensors` | Qwen-Image の VAE（`split_files/vae/`） |
| 拡大モデル | `4x-UltraSharp.pth` | [Kim2091/UltraSharp](https://huggingface.co/Kim2091/UltraSharp)（CC BY-NC-SA 4.0）。ESRGAN 系の 4 倍のモデルなら何でも |
| Turbo LoRA（下書き） | `anima_turbo_4step_v3.safetensors` | [sorryhyun/anima-turbo-4step](https://huggingface.co/sorryhyun/anima-turbo-4step) の Anima 用 Turbo LoRA（ライセンスはモデルのページを参照）、またはほかの Anima 用の少ないステップ用の LoRA |

ファイルは `ComfyUI/models/` の `diffusion_models/`・`text_encoders/`・`vae/`・`upscale_models/`・`loras/`（Turbo LoRA）に置きます。

モデルと LoRA にはそれぞれのライセンスがあります。画像を公開する前に確認してください。

## ワークフローの設定

`anima_txt2img.json` の値です（下書きのワークフローは、上の表に書いたところが違います）。

* **サンプリング**: KSampler はすべて `real` / `tors`（「Shared sampler」経由）、CFG 5、16 ステップ。
* **CFG の早期停止**: `hard_stop`。本体 0.65、拡大 0.5。
* **TaylorSeer**: `adaptive`、`rel_l1_thresh` 0.06、`max_predict_steps` 3、`max_order` 1、`last_enhance` 1。`first_enhance` は本体 8、拡大 1（ノードの既定値は本体の README の [使い方](../README_ja.md#taylorseerweightferryspeed)）。下書きのワークフローでは、TaylorSeer は本体の段だけにあります。
* **DiT Layer Streaming**: `auto_config` オン、`sage_attention` オン（`sageattention` のパッケージが無ければ、ログに警告が 1 行出て ComfyUI のアテンションで動きます）。
* **Ultimate SD Upscale**（このパッケージの Ultimate SD Upscale+。最初はバイパス。Ctrl+B で使う）: 2 倍、denoise 0.38、16 ステップ。タイルと 2 つの機能の値は「USDU tile calculator」（`tile_factor` 4）から（本体の README の [拡大](../README_ja.md#拡大weightferryupscale)）。
* **タイルなし**（拡大の方式 = 1）: 本体の README の [拡大](../README_ja.md#拡大weightferryupscale) のつなぎ方で、lanczos 0.5（4 倍のモデルで 2 倍に）、denoise 0.38。「Upscale factor」のノードが効くのは Ultimate SD Upscale とタイルの計算だけです。この経路で倍率を変えるときは、lanczos の倍率と RoPE Scale の `factor` を変えてください。

## ライセンス

このフォルダのファイル（ワークフローと README）は **MIT-0** です（[../LICENSES/MIT-0.txt](../LICENSES/MIT-0.txt)）。何に使っても、複製・改変・再配布しても構いません。条件も表記の義務もありません。
