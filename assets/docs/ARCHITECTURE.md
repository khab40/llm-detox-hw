# Architecture

This diagram shows the implemented flow for the LLM detox homework:
data preparation, supervised and preference training, reward modeling,
PPO reward variants, and final evidence outputs.

```mermaid
flowchart TD
    HH["Anthropic/hh-rlhf<br/>harmless-base split"]
    Base["Qwen/Qwen2.5-0.5B<br/>base policy"]
    EvalPrompts["Held-out prompt slices<br/>mild_prefix<br/>direct_provocation<br/>rtp_challenging"]

    HH --> Parse["Parse prompt / chosen / rejected triples"]
    Parse --> Filter["Detoxify filter<br/>chosen toxicity <= 0.10<br/>rejected toxicity >= 0.50"]
    Filter --> SFTData["data/sft.jsonl<br/>prompt + benign response"]
    Filter --> DPOData["data/dpo.jsonl<br/>prompt + chosen + rejected"]

    Base --> SFT["SFT LoRA fine-tune<br/>src.detox_hw.train_sft"]
    SFTData --> SFT
    SFT --> SFTCkpt["checkpoints/sft"]
    SFTCkpt --> SFTMerge["Merge SFT LoRA into base<br/>src.detox_hw.merge_lora"]
    Base --> SFTMerge
    SFTMerge --> SFTMerged["checkpoints/sft_merged<br/>PPO actor/reference"]

    SFTCkpt --> DPO["DPO fine-tune<br/>tasks.task2_dpo_loss<br/>src.detox_hw.train_dpo"]
    DPOData --> DPO
    DPO --> DPOCkpt["checkpoints/dpo"]

    DPOData --> RM["Bradley-Terry reward model<br/>tasks.task4_bt_loss<br/>tasks.task5_reward_head"]
    Base --> RM
    RM --> RMCkpt["checkpoints/rm"]

    SFTMerged --> PPOInv["PPO via verl<br/>reward = inv:detoxify"]
    SFTMerged --> PPORM["PPO via verl<br/>reward = trained RM"]
    SFTMerged --> PPOCustom["PPO via verl<br/>reward = custom function"]
    RMCkpt --> PPORM
    Detoxify["unitary/toxic-bert<br/>via detoxify"] --> Filter
    Detoxify --> PPOInv
    CustomReward["tasks.task8_custom_reward<br/>saturated Detoxify<br/>RM blend<br/>anti-template gates"] --> PPOCustom

    PPOInv --> MergeInv["Merged PPO checkpoint<br/>ppo_inv_detoxify_merged"]
    PPORM --> MergeRM["Merged PPO checkpoint<br/>ppo_rm_merged"]
    PPOCustom --> MergeCustom["Merged PPO checkpoint<br/>ppo_custom_merged"]

    SFTCkpt --> Eval["Evaluation library<br/>sampled_eval<br/>greedy_eval<br/>worst_of_k_eyeball"]
    DPOCkpt --> Eval
    MergeInv --> Eval
    MergeRM --> Eval
    MergeCustom --> Eval
    EvalPrompts --> Eval
    Detoxify --> Eval
    Eval --> Reports["submissions/<br/>JSON metrics + writeups<br/>eyeballed completions"]
    Reports --> Zip["evidence/*.zip<br/>README-required submission files"]
```

## Planned Components

| Component | Responsibility |
|---|---|
| `data_prep.build_pairs` | Build filtered SFT and DPO datasets from hh-rlhf preference pairs. |
| `src.detox_hw.train_sft` | Train the benign-response LoRA adapter used as the first detox baseline. |
| `src.detox_hw.merge_lora` | Merge the SFT LoRA adapter into a regular HF checkpoint for verl PPO. |
| `tasks.task2_dpo_loss` and `src.detox_hw.train_dpo` | Implement and run DPO against chosen/rejected preference pairs. |
| `tasks.task4_bt_loss`, `tasks.task5_reward_head`, `src.detox_hw.train_rm` | Train a scalar reward model with Bradley-Terry preference loss. |
| `src.toxic_rl.verl_runner` and `src.toxic_rl.verl_reward` | Launch verl PPO and route reward variants. |
| `tasks.task8_custom_reward` | Combine saturated Detoxify, prompt relevance, RM blending, and anti-collapse penalties. |
| `src.detox_hw.eval_lib` | Evaluate greedy outputs, sampled support, and worst-of-k completions. |
| `scripts/ssh_*` and `scripts/tmux_run.sh` | Upload, run, retry, and collect experiments on the Nebius VM. |
| `submissions/` | Store metrics, diagnostic text, and task writeups. |

## Final Task 8 Observation

The final custom PPO checkpoint avoided the earlier malformed output
attractors (`溆`, prompt echoes, role labels, and web/user-agent
fragments). It still behaves conservatively, but the final K=16 support
rates were low:

| Slice | Support | Mean max | Mean std |
|---|---:|---:|---:|
| `mild_prefix` | 0.000 | 0.031 | 0.008 |
| `direct_provocation` | 0.000 | 0.009 | 0.002 |
| `rtp_challenging` | 0.060 | 0.085 | 0.021 |
