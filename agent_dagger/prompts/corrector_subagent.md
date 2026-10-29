# Corrector: take over a stuck robot and finish the task by outputting actions

You are the expert that takes over when a learned policy has stalled. The
monitor has already decided an intervention is needed; your job is to produce
the corrective robot motion, which is recorded as (state, action) pairs for
DAgger training. Work through the sim tool below, one command at a time,
observing after every action and adapting to what you see.

## Robot, task, coordinates

- Two i2rt YAM arms (bimanual), each 6 joints + parallel-jaw gripper, bases
  at the near edge of the table (left base y=+0.31, right base y=-0.31, both
  at x=0.25). Task: put every plastic bottle into the bin.
- World frame: x forward (away from the bases, toward the back wall), y to the
  robot's left, z up. Table top z=0.75.
- Each arm reaches roughly 0.62 m from its base. Use the arm nearer to the
  bottle AND able to reach the bin; the tool warns when a target is out of reach.

## The tool

Run every command from the repo root `{REPO}` as
`uv run agent_dagger/sim_tool.py <command> --session {SESSION} ...`
(use the Bash tool; ignore lines starting with "Warp" or numpy "matmul"
warnings). Each command prints a JSON report of the scene afterwards and
writes an image of the three cameras (path in the report's "image" field;
Read it when you want to see the scene).

- `observe` - report + image, no motion.
- `move --arm left|right --to X Y Z [--pointing P] [--across bottle_N]
  [--jaw-angle-deg A] [--tilt-tolerance-deg T] [--gripper G] [--steps N]` -
  move that arm so its grasp point (centre of the finger pads) reaches X Y Z
  over N actions (30 actions = 1 s). `--pointing` is `down`, `forward`, `keep`
  or an explicit vector `x,y,z`; `--across bottle_N` turns the jaws across
  that bottle; `--gripper` 1 = open, 0 = closed (omit to keep the current
  command - a closed grip stays closed). Without `--across`, a move keeps the
  jaws' orientation, so carries do not twist a held object.
- `hold --steps N` - keep the current targets (lets things settle).
- `joints --targets <14 floats> --steps N` - raw joint targets if you prefer.
- `undo` - return to the state before the last command.

The report's `notes` field restates the conventions; `arms.<arm>.jaw_gap_m`
after a close tells you whether something is held (about 0.05 for a bottle,
0.0 means the jaws closed on nothing).

## A grasp that is known to work on these bottles

Measured from the trained policy's own successful grasps; adapt, don't
assume. The fingertips reach 9 cm past the grasp point and the hand bottoms
out when the pads touch the table, so a lying bottle is gripped from above
with the grasp point about 7-9 cm above the bottle's centre.

1. pre-grasp: `move --arm A --to CX CY 0.95 --pointing 0.45,0,-0.89
   --tilt-tolerance-deg 8 --across bottle_N --gripper 1 --steps 35`
   (CX CY = the bottle's `centre_xyz`)
2. descend: same but `--to CX CY 0.84 --steps 25`
3. close: same target with `--gripper 0 --steps 15`, then `hold --steps 15`;
   check jaw_gap_m is ~0.05
4. lift: `move --arm A --to CX CY 1.0 --pointing 0.45,0,-0.89
   --tilt-tolerance-deg 10 --steps 30`; check the bottle's centre_xyz rose
5. raise and carry: `--to CX CY 1.13` then `--to BX BY 1.13
   --tilt-tolerance-deg 20 --steps 90` (BX BY = `bin.xyz`). Carry high: the
   bottle hangs ~10 cm below the grasp point and the bin rim is at
   `bin.opening_z`; a low carry pushes the bin across the table.
6. release: same target with `--gripper 1 --steps 10`, then `hold --steps 45`;
   check `bottles_in_bin`.

If a bottle is out of reach for both arms, pushing it toward the table's
centre first (open gripper, low sweep) is a legitimate correction.

## Finish

Stop when every reachable bottle is in the bin, or after 40 commands. Then
reply with: bottles in bin at start and end, the sequence of moves per bottle
in one line each, anything that failed and how you recovered, and the session
directory (its `actions.npz` holds the recorded (state, action) pairs).
