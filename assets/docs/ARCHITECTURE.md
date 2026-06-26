# Architecture

This diagram shows the planned implementation flow for the LLM detox homework: data preparation, supervised and preference training, reward modeling, PPO reward variants, and final evaluation outputs.

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

    SFTCkpt --> DPO["DPO fine-tune<br/>tasks.task2_dpo_loss<br/>src.detox_hw.train_dpo"]
    DPOData --> DPO
    DPO --> DPOCkpt["checkpoints/dpo"]

    DPOData --> RM["Bradley-Terry reward model<br/>tasks.task4_bt_loss<br/>tasks.task5_reward_head"]
    Base --> RM
    RM --> RMCkpt["checkpoints/rm"]

    Base --> PPOInv["PPO via verl<br/>reward = inv:detoxify"]
    Base --> PPORM["PPO via verl<br/>reward = trained RM"]
    Base --> PPOCustom["PPO via verl<br/>reward = custom function"]
    RMCkpt --> PPORM
    Detoxify["unitary/toxic-bert<br/>via detoxify"] --> Filter
    Detoxify --> PPOInv
    CustomReward["tasks.task8_custom_reward<br/>prompt relevance + anti-toxicity"] --> PPOCustom

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
```

## Planned Components

| Component | Responsibility |
|---|---|
| `data_prep.build_pairs` | Build filtered SFT and DPO datasets from hh-rlhf preference pairs. |
| `src.detox_hw.train_sft` | Train the benign-response LoRA adapter used as the first detox baseline. |
| `tasks.task2_dpo_loss` and `src.detox_hw.train_dpo` | Implement and run DPO against chosen/rejected preference pairs. |
| `tasks.task4_bt_loss`, `tasks.task5_reward_head`, `src.detox_hw.train_rm` | Train a scalar reward model with Bradley-Terry preference loss. |
| `src.toxic_rl.verl_runner` and `src.toxic_rl.verl_reward` | Launch verl PPO and route reward variants. |
| `src.detox_hw.eval_lib` | Evaluate greedy outputs, sampled support, and worst-of-k completions. |
| `submissions/` | Store metrics, diagnostic text, and task writeups. |
