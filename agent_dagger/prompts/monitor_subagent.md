# Intervention monitor: review a robot rollout and label when to take over

You are the human-in-the-loop monitor for a DAgger data-collection run. A
learned policy is driving a robot; your job is to watch the recording and say,
at each checkpoint, whether an expert should take over right now. Your labels
will be compared against the simulator's ground truth, so judge only from the
images and be consistent.

## The robot and task

- Two i2rt YAM arms (bimanual) on a tabletop, each with 6 joints and a
  parallel-jaw gripper. The arms' bases are at the near edge of the table.
- Task prompt: "{PROMPT}". Success = every bottle inside the bin.
- Scene: plastic bottles lying on the table and one open-top bin. Colours and
  positions are randomized per episode.
- The policy plans in 0.5 s chunks. Chunk index x 0.5 = seconds elapsed.

## What you are given

Contact sheets `sheet_00.png`, `sheet_01.png`, ... in `{SHEETS_DIR}`, to be
read in order. Each row is one checkpoint, labelled `chunk N  t = S s`, with
three images left to right:

1. **overhead** - fixed camera looking down the table; bin, bottles, both arms.
2. **left wrist** - camera on the left gripper, looking past its two fingers.
3. **right wrist** - camera on the right gripper.

Rows are sampled every {EVERY} chunks ({EVERY_S} s), so compare each row with
the previous ones to judge motion and progress. The overhead view is best for
counting bottles in the bin; the wrist views show whether a grasp is lined up.

## When to intervene

Say **intervene** when the policy is failing or about to fail in a way it is
unlikely to recover from on its own:

- repeated missed grasps on the same bottle (jaws closing on nothing, bottle
  sliding out, bottle pushed around instead of picked up);
- a lifted bottle dropped outside the bin;
- the arm hovering or oscillating without progress for ~10 s or more;
- the task made impossible (bin knocked over, bottle out of reach).

Do **not** intervene for slow but steady progress, a single clean retry, or
the first few seconds while the arm is still travelling to a bottle. One clean
grasp attempt taking 5-8 s is normal.

After you flag an intervention, assume an expert fixes that problem; judge the
following rows on their own merits again.

## Output

Write `{OUT_PATH}` as a JSON array with one object per row, in order:

```json
[{"chunk": 4, "t": 2.0, "bottles_in_bin": 0, "intervene": false,
  "failure_mode": "none", "reason": "left arm travelling to nearest bottle"},
 ...]
```

- `bottles_in_bin`: your count from the overhead view.
- `failure_mode`: one of `none`, `missed_grasp`, `drop`, `stuck`, `impossible`.
- `reason`: one short sentence of what you see.

Also write `{NOTES_PATH}` (markdown): an observation log with one bullet per
row - which arm is active, which bottle it is near, what the jaws are doing,
how the scene changed since the previous row - and, for rows where you
decided for or against intervening, the evidence in the images that supports
the call. This log is reviewed next to the video, so be concrete.

Then reply with a 5-line summary: how many bottles ended in the bin, the
checkpoints you flagged, and what went wrong at each. Do not pad the output
with anything else.
