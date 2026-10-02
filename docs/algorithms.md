# unsloth-native algorithms

`AlgorithmSpec` は advantage、目的関数、ratio の参照元、behavior 補正、正規化を分離する。
`resolve_algorithm` が legacy 引数との矛盾を検出し、`skip_old_logprob_forward` を bool に解決する。

記号: `l` は学習者の logprob、`o` は更新直前の学習者、`s` は sampling policy、
`f` は KL reference、`m` は action mask、`T_i=Σ_t m_it`、`sg` は detach。
Tool token は attention に残し、loss と診断から除く。

| knob | 数式・意味 | Unsloth kwargs / 実現箇所 |
|---|---|---|
| `advantage_estimator=group_zscore` | `(reward−group_mean)/(std+1e-4)`、sample std | `dataset.AdvantageSpec` → advantages 上書き |
| `std_scope=group/batch` | group_zscore の std を group / batch の軌跡報酬から計算 | 同上。軌跡ごとに 1 票 |
| `group_center` | `reward−group_mean` | 同上 |
| `rloo` | `G/(G−1)·(reward−group_mean)` | 同上 |
| `policy_objective=grpo_clip` | `−min(rA,clip(r,1−epsilon_low,1+epsilon_high)A)` | `loss_type=dapo` |
| `policy_objective=cispo` | `−sg(min(r,epsilon_high))·A·l` | `loss_type=cispo` |
| `policy_objective=reinforce` | `−A·l`（選択した behavior 補正と KL は別途適用） | `loss_type=cispo`, old=None、Zoo `epsilon_high=max(1,epsilon_high)` |
| `ratio_reference=proximal` | `r=exp(l−o)` | old=scoring logprob、省略時 `sg(l)` |
| `ratio_reference=behavior` | scoring を省略、proximal r=1。behavior weight が比を担う | CISPO の behavior cap は `epsilon_high` |
| `behavior_correction=none` | weight=1 | `use_vllm=False`, `vllm_importance_sampling_correction=False` |
| `behavior_correction=truncate`, `behavior_cap` | `w=min(exp(o−s),cap)` | `vllm_importance_sampling_mode={token,sequence}_truncate`, cap と clip_max |
| `behavior_correction=mask`, `behavior_clip_min/max` | 範囲外の w を 0 にする。min=null は下限なし、max=null は cap | `vllm_importance_sampling_mode={token,sequence}_mask`, clip_min/max |
| `importance_sampling_level=sequence` | proximal r は `exp(Σm(l−o)/max(T,1))`、behavior w は **`exp(Σm(o−s))`** | `importance_sampling_level` と behavior mode の両方 |
| `loss_normalization=action_token` | 全 step の `D=Σ_i T_i` | scalar `num_items_in_batch=D` |
| `loss_normalization=sequence` | `D=行数`, `A_i←A_i/max(T_i,1)` | beta=0 は左記。beta!=0 は native grpo の行平均 / GAS |
| `loss_normalization=fixed_length`, `fixed_length` | `D=行数×fixed_length`、null は model.max_seq_length | 同上 |
| `epsilon`, `epsilon_high` | PPO の下側・上側幅。CISPO の high は幅でなく ratio の上限 | `epsilon_low/high`、config 構築後に再設定 |
| `delta` | PPO の第一項だけ `r←min(r,delta)` | `delta` |
| `beta` | `+beta·(exp(f−l)−(f−l)−1)`、behavior weight を掛けた後に加える | `beta`、reference forward は残る |
| `skip_old_logprob_forward` | auto: behavior、または dropout=0・1 iteration・reference 同期なし | scoring 中だけ `use_vllm=False`。old=None と sampling ありなら loss 内で old=sg(new) |

action_token / fixed_length と beta=0 の sequence は masked token loss の和を D で割る。
これらは ragged micro-batch も同じ D を共有する。
`mask_truncated_completions` は trainer.args の明示値を最優先する。未指定なら legacy
loss_type=dapo または loss_type 未指定で True、grpo/dr_grpo/bnpo で False。
config 構築後と begin_step で解決値を設定し、Unsloth の書き換えを打ち消す。
True の場合、renderer が length 終了と判定した行の numerator を 0 にし、
action_token 分母からその行の token を除く。sequence / fixed_length は全行数を維持する。
`grpo_clip+behavior` は拒否する。`group_zscore` / `group_center` / `rloo` は G<2 を学習対象から外す。
`sequence+beta!=0` は grpo_clip で許可し、CISPO / REINFORCE では拒否する。
Zoo の native grpo は目的関数も PPO clip に切り替えるため、後者に適用すると数式が変わる。
CISPO+behavior は truncate のみ。CISPO の epsilon_high は未指定なら behavior_cap を使う。
参照元にかかわらず epsilon_high<1 は拒否し、r=1 で clip が勾配を縮小しないことを保証する。
REINFORCE + proximal の behavior 補正は LoRA dropout=0 のみ許可する。Zoo の unit-ratio
経路では補正の old も current.detach() になり、dropout がある scoring old と分離できない。
これらの境界条件の設定意図は SPEC だけでは不明。REINFORCE と CISPO が r=1 で一致するのは
CISPO の epsilon_high>=1、同じ補正・KL・分母の場合である。

| legacy trainer.args | resolved algorithm |
|---|---|
| `loss_type=dapo`（既定） | grpo_clip + action_token |
| `loss_type=grpo` | grpo_clip + sequence |
| `loss_type=dr_grpo` | grpo_clip + fixed_length、長さは legacy max_completion_length |
| `loss_type=bnpo` | grpo_clip + action_token。1 micro-batch のときのみ元の BNPO と等価、警告 1 回 |
| `scale_rewards=group/True` | group_zscore + std_scope=group |
| `scale_rewards=none/False` | group_center |
| `scale_rewards=batch` | group_zscore + std_scope=batch |
| `epsilon`, `epsilon_high`, `delta`, `beta`, `importance_sampling_level` | 同じ意味で引き継ぐ |
| `vllm_importance_sampling_cap` | behavior_cap |

明示した algorithm と legacy が異なる値なら validation error。`trainer.args.loss_type=cispo`
ではなく `trainer.algorithm.policy_objective=cispo` を使う。

| algorithm | Unsloth | CPU / plain TRL |
|---|---|---|
| GRPO | zoo dapo。sequence + beta!=0 は zoo grpo | shim `_compute_loss` → `policy_gradient_loss` |
| RLOO + grpo_clip | RLOO advantage → zoo dapo。sequence + beta!=0 は zoo grpo | 同上 |
| REINFORCE | zoo cispo、r=1、high>=1 | 同上、`−A·l` |
| CISPO | zoo cispo | 同上、detached clipped ratio |

Unsloth の generated `compute_loss` と accumulated loss は、関数オブジェクトと globals を
trainer ごとに複製する。共有 module と wheel のソースは変更しない。custom autograd の
forward 境界で old を補い、元の backward を継承する。Gemma 4 memory path も同じ補正を行う。
診断は各 micro-batch の action token を連結して集計し、signed logprob 差は custom forward
境界で取得する（Zoo の返す abs(delta) から符号を推定しない）。

GPU 上の generated trainer、compiled custom autograd、PrefixGrouper、sequence packing は未検証。

## KL と legacy 分割の互換性

`sequence + beta!=0` では advantage を縮小せず、Zoo `loss_type=grpo` で
`mean_rows(Σ_t loss_it m_it / max(T_i,1)) / num_chunks` を計算する。
KL 項も行ごとに同じ分母で割る。等サイズ chunk が必要であり、planner の行数が不均一なら
`micro_batch_chunks(n_rows, max_rows)` による等分へ切り替えて警告する。
この fallback は group 不分割や token/group budget を上書きし得る。
`fixed_length + beta!=0` は D=全行数×L を使う dapo 経路のままでよい。
等サイズ K chunk なら native dr_grpo の分母 `(B_chunk×L)×K` と等しく、
policy gradient と KL の両方を B×L で割るため数学的に一致する。

legacy YAML（algorithm と batch の明示設定なし）は従来の等分と `num_chunks` API を使う。
特に legacy `loss_type=grpo` は beta=0 でも native grpo / TRL grpo の行平均 / GAS を保持する。
algorithm または batch を明示すると新 planner を使い、未指定 max_rows は
`per_device_train_batch_size` になる。planner は長さ降順の first-fit、または入力順の first-fit。
予算超過の単一 group は警告して単独に置く。全行を必ず一度ずつ含める。
legacy `begin_step(num_chunks=...)` も index 集合を検査し、末尾行が落ちる場合は ValueError。
行数より多い num_chunks と空 chunk は ValueError。

`build_batch` は既存の互換性のため zero-std group の行を保持してカウントする。
動的 sampling の drop 判定は `would_train_group` が行う。group 系は使える trajectory が
2 本以上かつ報酬が不均一の場合だけ True。
長さ超過は既存どおり診断であり、trajectory を除外しない。平均・sample std は各 trajectory
1 entry で計算し、複数 row に同じ advantage を broadcast する。
`num_zero_advantage_groups` は estimator 適用後に全 advantage が厳密に 0 の group 数。

`skip_old_logprob_forward=auto` は behavior 参照、または LoRA dropout=0・num_iterations=1・
reference snapshot 同期なしの場合に有効。beta!=0 の reference forward は残る。
Zoo の PrefixGrouper は `not use_vllm`、behavior correction は `use_vllm` で gate されるため
同じ loss forward では排他的。補正なし・skip 有効なら PrefixGrouper が候補になる。
sequence packing は use_vllm では gate されない。prefix_grouper_verified /
seq_packing_verified は model の検証フラグが None でない場合に True とする。
これは過去の shape 検証状態であり、今回の step で実行されたことを表さない。

off-policy 診断は既存 loss record の detach 済み tensor だけから step 終了時に集約し、
診断用 forward は行わない。action token 上で `lw=o_detached-s`（sequence level では
sequence 内の和）、`w_raw=exp(lw)`、`w` は loss が使った truncate/mask 後の weight とする。
`behavior_log_ratio_mean/std/abs_p99` と `behavior_ratio_raw_p50/p95/p99/max/mean` は
truncate 前、`importance_ratio_p50/p95/p99/max/mean` と
`importance_ratio_capped_fraction` は truncate/mask 後を表す。
`behavior_ess_relative=(Σw_raw)^2/(N Σw_raw^2)`、`behavior_ess_relative_used` は同式の
`w` 版である(いずれも log-space の logsumexp で計算し、sequence level の大きな log-weight
でも float32 で overflow しない)。`behavior_ratio_raw_p50/p95/p99/max` も float64 の
log-weight 上で quantile/max を求めてから exp し、`behavior_ratio_raw_mean` は float64 の
`exp(logsumexp(lw)-log(N))` で求める。このため raw 比の各指標は float32 の exp を経由せず、
float64 で表現可能な範囲では有限である。制限: `behavior_correction: none` のとき Unsloth の生成
`compute_loss` は fused loss に sampling logprob を渡さないため、fused 経路では
`behavior_*` と ESS が出ない(CPU shim と memory path では出る)。補正無しで raw 比を見たい
場合は `skip_old_logprob_forward: false` にすると `_sampling_stats` 経由で得られる。`proximal_ratio_identity` は skip、behavior reference、REINFORCE のように
proximal ratio が恒等的に 1 の step で 1、それ以外で 0。

各 loss record には `_prepare_inputs` が選んだ元 batch の row index を付け、
`learner_version_before-row.behavior_version` ごとに同じ診断を集約する。index mapping が
得られない Zoo/memory fallback は `off_policy_by_staleness={"all": ...}` の whole-step
bucket とし、`off_policy_by_staleness_partial=true` を付ける。
`num_loss_action_tokens` は loss を持つ forward 対象行の action token 数、
`num_forward_tokens` は実際に forward した行の `total_len` 合計、
`num_forward_padded_tokens` は trim 後の各 micro-batch の `rows×width` 合計、
`num_skipped_zero_loss_action_tokens` は zero-loss として forward しなかった行の action
token 数である。従来の batch 全体の `num_action_tokens` / `num_tokens` は変更しない。

## Adapter と Gemma 4

`export_adapter(version)` は async 用 AdapterSnapshot を model lock 内で作る。
`export_adapter_path(version)` は同期 VllmServerPolicy 用の Path を返す。
`adapter_export_root` は `<output_dir>/.lora-sync`。async policy / pool を作る統合者は
この property を `adapter_export_root` に渡す。snapshot のディスク書き出しは publisher が行う。
snapshot の所要時間は AdapterSnapshot.snapshot_s に保持し、報告済みの StepResult を変更しない。Gemma 4 は get_peft_model 直後に text-only mm_token_type_ids hook を装着し、
close で解除する。

診断は loss_diagnostics が detached logprob から必要な統計だけを計算し、loss 本体を再実行しない。
importance_ratio_capped_fraction は truncate では上限到達率、mask では下限未満または上限超過率。
生成した loss wrapper は trainer ごとに (AlgorithmSpec, base compute_loss) でキャッシュする。
_prepare_inputs は scoring に追加の autocast を適用しない。scalar 分母を全 chunk で共有し、
行 tensor と list を plan でスライスする。画像関連キーを含む入力は NotImplementedError。
