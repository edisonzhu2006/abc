# When humans intervene and hand back: ABC paper, YAM DAgger data, and the HIL literature

Compiled 2026-10-30 for the Astra-as-corrector DAgger work in `agent_dagger/`.

## 1. ABC paper (Section 5.3, Appendix F)

- Intervene: "intervene when it enters a state where it is struggling". No definition, no timer, no failure taxonomy.
- Hand back: "At any time the operator can press another button to transition back to the policy."
- Handback mechanics: the last few teleop actions become the RTC prefix so the policy picks the right mode (retry a flap vs move on). Caveat in Appendix H.2: prefix conditioning can suppress visual responsiveness; shorter prefixes helped (prefix 1: 8/10 vs prefix 4: 5/10 on t-shirt folding).
- Volume: rounds of 1 to 1.5 h; intervention share 30% in round 1, 15% in round 2. Training mix 80:10:10 (previous data : current interventions : rest of current episodes).
- Logged in the released recorder (`deploy/robot/gym/dagger_rollout.py`): `intervention_start` / `intervention_end` events with timestamp, controller state, episode, checkpoint. No reason, no note. The box-folding DAgger recordings are not released.

## 2. Measured from public YAM HG-DAgger data (YOLO2431, LeRobot format, per-frame `is_intervention`)

Task: bimanual key unlock on YAM arms, 30 fps. Computed from the parquet files only.

| | key-unlock-v3 r1 | key-unlock r3 |
|---|---|---|
| episodes | 495 | 200 |
| episode length, median | 39.7 s | 38.5 s |
| interventions per episode, mean / median / max | 1.54 / 2 / 3 | 1.61 / 2 / 3 |
| histogram of count 0,1,2,3 | 12, 214, 261, 8 | 5, 75, 113, 7 |
| episodes with no intervention | 2% | 2% |
| intervention length, median (p10 to p90) | 7.2 s (3.0 to 12.7) | 7.7 s (4.3 to 11.7) |
| first intervention starts, median | 10.8 s (27% into episode) | 8.5 s (23%) |
| policy gap between interventions, median | 10.7 s | 12.3 s |
| policy time after last handback, median (p90) | 6.6 s (8.2) | 4.3 s (6.4) |
| episodes ending inside an intervention | 21 of 483 | 9 of 195 |
| overall intervention share of frames | 28% | 33% |

Intervention start times are bimodal. In r1 the starts by episode decile are
[4, 86, 166, 37, 15, 136, 238, 66, 8, 4]: one cluster at 10 to 30% of the episode and one at 50 to 80%. The task has two bottleneck sub-steps and the operator intervenes once at each. Interventions are short (about 7 s), the operator hands back as soon as the sub-step is through, and the policy finishes the episode on its own in the last few seconds. Essentially nobody intervenes in the first or last 10%.

Nine datasets in the same family span 21% to 66% intervention share depending on task and round.

## 3. HG-DAgger (Kelly et al. 2019, arXiv 1810.02890)

- Take control: "the novice policy is rolled out until the expert observes that the novice has entered an unsafe region of the state space."
- Return: "Once in a safe region, control is returned to the novice."
- Only expert-controlled steps are labeled and added to the dataset. Novice steps produce no labels.
- "Doubt" = l2 norm of the ensemble output covariance diagonal, logged at each intervention; threshold tau = mean of the final 25% of logged doubts. Used to evaluate the trained policy, not to gate at test time.
- No intervention frequency or duration reported. 10,000 BC labels, then 2,000 expert labels per epoch for 5 epochs.

## 4. Sirius (Liu et al. 2023, arXiv 2211.08416)

- Intervene: "When the human detects undesirable robot behavior (e.g., gear getting stuck), the human intervenes."
- Hand back: "When the human judges that the robot can continue the task, the human passes control back to the robot."
- Human reaction time measured at about 22 s, about 15 robot actions; they drop a pre-intervention segment of that size from the policy data because it is probably already failing.
- Intervention ratio, frequency and length all decrease over rounds. Values only shown as plots.

## 5. RaC (Hu et al. 2025, arXiv 2509.07953)

- Intervene: "human operators intervene when failure appears imminent".
- Rule 1, recover then correct: every intervention first rewinds the robot to a familiar in-distribution state, then performs the correction that pushes the current sub-task forward. A SAM2 gripper-visitation heatmap from the round-0 demos is overlaid on the overhead camera to show where to recover to.
- Rule 2, terminate after intervention: the episode ends when the intervention ends. There is no handback at all. Rationale: post-intervention states are a human-plus-policy mixture that does not match the policy's own distribution on later sub-tasks.
- Data: only human frames from intervened episodes are trained on; fully autonomous successes are kept whole. Whole episode length is charged to the budget.
- Rounds: K = 10 for lid sealing, 9 for takeout box packing; 5 h total for shirt hanging reached 78% vs 89 h for ALOHA Unleashed.
- Without Rule 1, HG-DAgger-style data skews heavily to correction frames with few recovery frames; DROID has recovery in 3.68% of episodes.
- Test-time scaling: success rate rises linearly with the number of recovery segments the policy performs at deployment.
- Interface: VR controller clutch; side button held = human, released = policy. Relative pose deltas from clutch engagement, no posture alignment.

## 6. pi*0.6 / RECAP (Physical Intelligence 2025, arXiv 2511.14759)

- Corrections "serve more to fix large mistakes and overcome challenges with exploration, and do not by themselves provide for optimal supervision". Intervening "is a disruptive event, and even expert human operators cannot guarantee a consistent quality of interventions".
- All correction segments are forced to the positive advantage indicator; the rest of the episode is labeled by the learned value function. Whole episodes, autonomous and corrected, go into the dataset.
- Volume per iteration: box assembly 600 autonomous + 360 corrected episodes; diverse laundry 450 eval + 287 correction episodes, later about 1000 autonomous + 280 + 378 correction; cafe 414 autonomous + 429 correction. Two iterations on box assembly. Tasks run 5 to 15 min.
- No handback rule is stated.

## 7. DAgger (Ross et al. 2011, arXiv 1011.0686)

- The expert never takes control. The learner (or a beta-mixture of expert and learner) drives; the expert labels every visited state; datasets are aggregated across iterations.
- Behavior cloning error compounds as O(u T^2) in the horizon; DAgger reduces it to O(u T N) with N iterations of no-regret learning.

## 8. What this implies for the agent corrector

- Nobody writes down a handback criterion beyond "the robot can continue". The measurable human behavior on YAM is: intervene at the sub-step that is failing, take about 7 s, hand back as soon as that sub-step is through, 1 to 2 interventions per 40 s episode.
- RaC's recover-then-correct structure is the only protocol with evidence behind it. The corrector prompt already does correction; it does not ask for an explicit recovery segment first. RaC also argues for ending the episode after the intervention rather than handing back, which conflicts with the ABC and LeRobot protocols; for a short single-sub-task episode like bottles-in-bin the difference is small.
- Sirius's reaction-time trimming suggests the policy frames right before an intervention are low quality. The ABC 80:10:10 mix keeps them; RaC and HG-DAgger drop them.
- The ABC handback seeds the RTC prefix with the operator's last actions. `dagger_collect.py` seeds it by tiling the current state, which drops the mode-selection effect and should be revisited.
