# 4D Control Mapping Research and Transform Specification
Date: 2026-02-12
Status: Draft v0.1 (research + proposed normative spec)

## 1. Objective
Define a control mapping and transform math spec for first-person 4D navigation that:
- Preserves fast, low-error muscle memory.
- Minimizes mode confusion while still exposing genuine 4D motion.
- Is mathematically explicit enough to prevent long-term acclimation to a flawed scheme.

This document is scoped to the interactive explorer in `crates/game` and references current behavior in:
- `crates/game/src/input.rs`
- `crates/game/src/camera.rs`
- `crates/game/src/main.rs`

## 2. Current Engine Baseline (Observed)

### 2.1 Control schemes and rotation pairs
Current runtime behavior has three control schemes:
- `UPRIGHT` (`ControlScheme::IntuitiveUpright`)
- `LEG-SIDE`
- `LEG-SCRL`

Current rotation pairs:
- `Standard`: horizontal -> `Yaw`, vertical -> `Pitch`
- `FourD`: horizontal -> `XwAngle`, vertical -> `ZwAngle`
- `DoubleRotation`: horizontal -> `Yaw`, vertical -> `YwDeviation`

Mode activation:
- In `UPRIGHT`, holding mouse back button enables `FourD`; release returns to `Standard`.
- In legacy schemes, side buttons and optional wheel cycling can latch/switch pairs.

### 2.2 Movement semantics
Input axes from `InputState::movement_axes()`:
- `forward = W - S`
- `strafe = D - A`
- `vertical = Space - Shift`
- `w_axis = E - Q`

Two movement implementations:
- `apply_movement(...)` (legacy-style basis use, no input normalization).
- `apply_movement_upright(...)` (normalized input, center-forward + hidden-side decomposition).

`UPRIGHT` movement defines:
- `center_forward = normalize(view_z + view_w)`
- `side_w = normalize(view_w - view_z)`

This is the strongest existing mapping in the codebase for preserving FPS intuition while exposing 4D translation.

### 2.3 Existing correctness checks
Camera tests currently verify:
- Look-direction consistency with inverse view rotation.
- Mouse-right sign and mouse-up sign in baseline orientation.
- Upright constraints preventing Y inversion.
- Upright forward movement matching center look direction.
- Upright diagonal normalization.

All camera tests passed on 2026-02-12 via:
- `cargo test -p game camera::tests -- --nocapture`

## 3. External Research Findings (Condensed)

### 3.1 Humans can learn 4D spatial tasks, but training cost is real
- Aflalo & Graziano (2008) found participants could improve 4D maze path integration, with practice often spanning repeated sessions.
- Their logs reported typical practice on the order of minutes-to-hours per day over weeks.
- Implication: early control mapping decisions are high-impact because adaptation cost is non-trivial.

### 3.2 4D intuition can form, but can be fragile
- Ambinder et al. (2009) reported observers could use both 3D projection cues and fourth-dimensional information in judgments.
- They also describe this representation as limited/short-lived.
- Implication: the UI should reinforce stable invariants and avoid frequent remapping of command semantics.

### 3.3 Familiar 3D interfaces are still the best entry point
- Igarashi & Sawada (2023) explicitly frame 4D interaction as an extension of familiar 3D operations.
- Their findings include user confusion specifically when fourth-direction operation was required.
- Implication: preserve 3D-first defaults and expose 4D operations as deliberate, clearly signaled extensions.

### 3.4 Mode design matters: hold-to-modify beats hidden latching
- Sellen, Kurtenbach, and Buxton (1992) found user-maintained mode states reduced mode errors better than system-maintained/latching states.
- Implication: prefer hold-modifier mode access for alternate rotation pairs.

### 3.5 High-DOF control benefits from fine, direct motor channels
- Zhai, Milgram, and Buxton (1996) found that for 6-DOF tasks, designs leveraging finger participation improved completion times.
- Implication: use mouse (continuous, fine control) for rotational DOFs and keep keyboard roles simple and consistent.

## 4. Design Requirements

### 4.1 Human factors requirements
- `R1`: Primary controls must behave like a conventional FPS in common play.
- `R2`: 4D controls must be available instantly, without menu/context switching.
- `R3`: Mode state must be continuously legible in HUD and preferably user-maintained.
- `R4`: Command meaning must be stable across sessions (avoid remap-by-context).

### 4.2 Mathematical requirements
- `M1`: Orientation representation must remain orthonormal (`SO(4)` for 4x4 spatial block).
- `M2`: View/world transform conventions must be explicit (no hidden transpose ambiguities).
- `M3`: Translation speed must be isotropic across multi-key combinations.
- `M4`: Upright mode must keep world-up stable and prevent Y inversion in primary operation.
- `M5`: Angle update rules must avoid unbounded drift for periodic states.

## 5. Recommended Control Mapping (Normative)

### 5.1 Default policy
- Default scheme: `UPRIGHT`.
- Default pair: `Standard` (yaw/pitch).
- Alternate 4D pair access: hold mouse-back button (`user-maintained`).
- Legacy schemes remain optional compatibility paths, not the default.

### 5.2 Translation mapping
- `W/S`: `+/- center_forward`
- `A/D`: `-/+ right`
- `Q/E`: `-/+ hidden_side_w`
- `Space/Shift`: `+/- up` in fly mode
- In gravity mode: ignore `vertical` locomotion input, process jump/physics separately

Where:
- `center_forward = normalize(view_z + view_w)`
- `hidden_side_w = normalize(view_w - view_z)`

Rationale:
- Keeps `WASD` semantics familiar.
- Makes Q/E the dedicated "hidden-dimension strafe" pair.
- Avoids overloading forward with arbitrary Z/W ratio changes.

### 5.3 Rotation mapping
- Standard pair: horizontal mouse `dx` -> yaw.
- Standard pair: vertical mouse `dy` -> pitch (clamped).
- FourD pair (while modifier held): horizontal mouse `dx` -> `xw`.
- FourD pair (while modifier held): vertical mouse `dy` -> `zw`.

Operational rule:
- No persistent latch in primary path.
- Release modifier returns to `Standard`.

### 5.4 Mapping alternatives considered
- Always-on 4D rotation mix (all planes active): rejected due high interference and higher adaptation burden.
- Scroll/latch cycle for primary usage: rejected as default because mode-memory cost is high.
- Separate dedicated 4D "camera state": rejected as default due context splitting and weaker muscle-memory continuity.

## 6. Transform Math Specification (Normative)

### 6.1 Coordinate conventions
- World point: `p_w in R^4`
- Camera position: `c in R^4`
- View basis in world coordinates: columns of `B = [R U Z W]`, with `B in SO(4)`
- Homogeneous vectors use 5D form `[x y z w 1]^T`

### 6.2 Plane rotation primitive
Define `G(i, j, theta)` on an identity matrix with:
- `G[i,i] = cos(theta)`
- `G[i,j] = sin(theta)`
- `G[j,i] = -sin(theta)`
- `G[j,j] = cos(theta)`

This matches `rotation_matrix_one_angle(...)` in `src/matrix_operations.rs`.

### 6.3 Orientation composition
For existing code compatibility:
- Standard view rotation:  
  `R_std = G(Y,W,yw) * G(Z,W,zw) * G(X,W,xw) * G(Z,Y,pitch) * G(X,Z,yaw)`
- Upright view rotation:  
  `R_upr = G(Z,Y,pitch) * G(X,Z,yaw + yaw0) * G(Z,W,zw + zw0) * G(X,W,xw + xw0)`
- Current offsets: `yaw0 = 0`, `zw0 = +pi/4`, `xw0 = -pi/2`.

Vector application order is right-to-left.

### 6.4 View transform
Use:
- `d_w = R_view^T * d_v` for pure directions
- `x_v = R_view * (x_w - c)` for points

Homogeneous view matrix:
- `V = [ R_view  -R_view*c ; 0 0 0 0 1 ]`

### 6.5 Look ray definition
Center look in view coordinates:
- `l_v = (0, 0, 1/sqrt(2), 1/sqrt(2))`

World look:
- `l_w = normalize(R_view^T * l_v)`

### 6.6 Translation integration
Let input tuple be `(f, s, v, q)` from forward/strafe/vertical/w-axis controls.

Build wish direction:
- `wish = f*center_forward + s*right + q*hidden_side_w + v_eff*up`
- `v_eff = v` in fly mode, else `0`

Then:
- if gravity mode, project to world XZW hyperplane (`wish.y = 0`) before normalization
- normalize and scale by `speed * dt` with analog-preserving magnitude cap
- `c_next = c + step`

### 6.7 Constraints
- Pitch clamp: `pitch in [-81deg, +81deg]`
- Upright constraint: force `yw = 0` and enforce pitch clamp.
- Angle wrapping: wrap all periodic angles (`yaw`, `xw`, `zw`, and recommended `yw`) to `(-pi, pi]`.

Note:
- Current code wraps `yaw`, `xw`, `zw`, but not `yw`.
- Recommended change: wrap `yw` as well to avoid long-session precision drift.

## 7. Verification and Regression Test Plan

### 7.1 Algebraic invariants
- `R_view^T * R_view = I` within tolerance.
- `det(R_view_spatial) ~= +1`.
- `dot(center_forward, hidden_side_w) ~= 0`.
- `||center_forward|| = ||hidden_side_w|| = 1`.

### 7.2 Behavioral invariants
- Mouse-right in standard pair increases look direction along right axis.
- In upright mode, sweep `xw` through full range and verify yaw sign never inverts.
- Forward key in upright moves along center look direction.
- Multi-key diagonals do not exceed base speed.
- Gravity walking speed is invariant to pitch.

### 7.3 Mode invariants
- Modifier-held alternate pair engages only while pressed.
- HUD label always reflects active pair and scheme.
- Releasing modifier returns to primary pair within one frame.

### 7.4 Candidate additional unit tests
- `view_world_roundtrip_direction()`
- `upright_hidden_side_is_orthogonal_to_forward()`
- `all_periodic_angles_are_wrapped()`
- `legacy_and_upright_movement_speed_caps()`

## 8. Risk Register
- `Risk-A`: Angle-order regression when touching composition order.
- `Risk-B`: Silent transpose/sign error in world-view conversions.
- `Risk-C`: Mode confusion if default path reintroduces latching.
- `Risk-D`: Drift or precision loss if periodic angles are left unbounded.
- `Risk-E`: Gravity mode speed distortion if projection occurs after normalization.

## 9. Recommended Next Implementation Steps
1. Promote this spec to "enforced" by adding the missing invariant tests.
2. Wrap `yw_deviation` as a periodic angle.
3. Keep `UPRIGHT + hold-modifier FourD` as default and treat legacy paths as compatibility only.
4. Add a one-screen "control primer" overlay for first launch to reduce onboarding time.

## 10. References
- Aflalo TN, Graziano MSA. Four-Dimensional Spatial Reasoning in Humans. Journal of Experimental Psychology: Human Perception and Performance, 2008.  
  Link: https://grazianolab.princeton.edu/document/98
- Ambinder MS, Wang RF, Crowell JA, Francis GK, Brinkmann P. Human four-dimensional spatial intuition in virtual reality. Psychonomic Bulletin & Review, 2009.  
  Link: https://pubmed.ncbi.nlm.nih.gov/19815783/
- Igarashi H, Sawada H. 4D Exploring System for Intuitive Understanding of 4D Space by Extending Familiar 3D Interfaces. ICAT-EGVE, 2023.  
  Link: https://diglib.eg.org/bitstreams/4b4b661f-6224-4c09-9c3b-292ac11b833c/download
- Sellen AJ, Kurtenbach GP, Buxton WAS. The Prevention of Mode Errors Through Sensory Feedback. Human-Computer Interaction, 1992.  
  Link: https://www.billbuxton.com/ModeErrorsHCI.pdf
- Zhai S, Milgram P, Buxton W. The Influence of Muscle Groups on Performance of Multiple Degree-of-Freedom Input. CHI 1996.  
  Link: https://www.billbuxton.com/homunculus.pdf
