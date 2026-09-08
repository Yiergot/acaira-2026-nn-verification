--------------------------------------------------------------------------------
-- EXERCISE VERSION of specs/triage.vcl
--
-- Everything from the original specification is here and unchanged, so this
-- file typechecks and every original property can still be verified with it.
-- Four exercises are appended at the end; search for "EXERCISE" and "TODO".
-- Solutions are in the notebook (collapsed "Solutions" section).
--------------------------------------------------------------------------------

type Patient = Tensor Real [7]

-- Outputs: one score per urgency class. The network "advises" the class with
-- the highest score.
type Scores = Tensor Real [3]

-- Names for the input positions ...
rr    = 0   -- respiration rate, breaths per minute
spo2  = 1   -- oxygen saturation, %
sbp   = 2   -- systolic blood pressure, mmHg
pulse = 3   -- heart rate, beats per minute
temp  = 4   -- temperature, degrees Celsius
o2    = 5   -- 1 if the patient is on supplemental oxygen, else 0
cvpu  = 6   -- 1 if new confusion / not alert (C, V, P or U), else 0

-- ... and for the output positions.
low    = 0
medium = 1
high   = 2

-- The network itself. Its implementation (an ONNX file) is supplied on the
-- command line, so the same specification can be checked against any model.
@network
triage : Patient -> Scores

--------------------------------------------------------------------------------
-- The region we care about: physiologically plausible measurements.

validPatient : Patient -> Bool
validPatient x =
  4  <= x ! rr    <= 60  and
  50 <= x ! spo2  <= 100 and
  40 <= x ! sbp   <= 250 and
  20 <= x ! pulse <= 200 and
  30 <= x ! temp  <= 43  and
  0  <= x ! o2    <= 1   and
  0  <= x ! cvpu  <= 1

-- The two flags are really booleans. Verifiers only understand real numbers,
-- so we pin them to exact values inside each property.
alertOnAir : Patient -> Bool
alertOnAir x = x ! o2 == 0 and x ! cvpu == 0

-- "The network advises class c" = c has the strictly highest score.
advises : Patient -> Index 3 -> Bool
advises x c = forall j . j != c => triage x ! c > triage x ! j

--------------------------------------------------------------------------------
-- Hard-safety properties: things that must hold for EVERY valid patient.
-- Each mirrors a NEWS2 rule that a single parameter scoring 3 points is,
-- on its own, a trigger for urgent clinical review — never "low urgency".

-- SpO2 of 91% or below scores 3 points. Never low.
@property
hypoxiaNeverLow : Bool
hypoxiaNeverLow = forall x .
  validPatient x and alertOnAir x and x ! spo2 <= 91 =>
    not (advises x low)

-- Same rule for a patient who is already on oxygen (which itself adds 2 points).
@property
hypoxiaOnOxygenNeverLow : Bool
hypoxiaOnOxygenNeverLow = forall x .
  validPatient x and x ! o2 == 1 and x ! cvpu == 0 and x ! spo2 <= 91 =>
    not (advises x low)

-- Systolic blood pressure of 90 mmHg or below scores 3 points. Never low.
@property
shockNeverLow : Bool
shockNeverLow = forall x .
  validPatient x and alertOnAir x and x ! sbp <= 90 =>
    not (advises x low)

-- A patient who is not alert scores 3 points. Never low.
@property
notAlertNeverLow : Bool
notAlertNeverLow = forall x .
  validPatient x and x ! o2 == 0 and x ! cvpu == 1 =>
    not (advises x low)

--------------------------------------------------------------------------------
-- Over-triage matters too (it costs beds and attention). A patient whose
-- vitals sit comfortably inside every 0-point range must be advised "low".
-- The margins (e.g. 13-19 rather than 12-20) are a *specification decision*:
-- how close to a clinical threshold do we demand the network be exactly right?

normalVitals : Patient -> Bool
normalVitals x =
  13   <= x ! rr    <= 19   and
  97   <= x ! spo2  <= 100  and
  115  <= x ! sbp   <= 210  and
  55   <= x ! pulse <= 88   and
  36.3 <= x ! temp  <= 37.8

@property
normalVitalsAlwaysLow : Bool
normalVitalsAlwaysLow = forall x .
  validPatient x and alertOnAir x and normalVitals x =>
    advises x low

--------------------------------------------------------------------------------
-- Robustness to measurement noise around real (here: synthetic) patients.
-- Two observation charts that differ by less than measurement error should
-- not receive different triage categories.

@parameter
epsRR    : Real   -- breaths per minute
@parameter
epsSpo2  : Real   -- percentage points
@parameter
epsSbp   : Real   -- mmHg
@parameter
epsPulse : Real   -- beats per minute
@parameter
epsTemp  : Real   -- degrees Celsius

@parameter(infer=True)
n : Nat

@dataset
patients : Vector Patient n

@dataset
labels : Vector (Index 3) n

withinNoise : Patient -> Bool
withinNoise d =
  -epsRR    <= d ! rr    <= epsRR    and
  -epsSpo2  <= d ! spo2  <= epsSpo2  and
  -epsSbp   <= d ! sbp   <= epsSbp   and
  -epsPulse <= d ! pulse <= epsPulse and
  -epsTemp  <= d ! temp  <= epsTemp  and
  d ! o2 == 0 and d ! cvpu == 0

robustAround : Patient -> Index 3 -> Bool
robustAround x c = forall d .
  withinNoise d and validPatient (x + d) =>
    advises (x + d) c

@property
noiseRobust : Vector Bool n
noiseRobust = foreach i . robustAround (patients ! i) (labels ! i)


--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
-- EXERCISES
--
-- Each exercise is a property whose *region* is deliberately wrong or missing.
-- As written, every exercise property typechecks and can be run; that way you
-- can see what the verifier says before and after your fix. Run them with
--
--   verify("specs/triage-exercises.vcl", "models/triage-v3.onnx",
--          properties=["temperatureExtremesNeverLow"])
--
-- and read the counterexample with show_patient(...) in the notebook.
--------------------------------------------------------------------------------

-- EXERCISE 1 — temperature extremes.
-- NEWS2 Scale 1 scores temperature as follows (degrees Celsius):
--   <= 35.0 : 3 points     35.1-36.0 : 1     36.1-38.0 : 0     38.1-39.0 : 1     >= 39.1 : 2
-- Only ONE of the two extremes scores 3 points on its own, so only one of them
-- is a "never low" trigger. State that rule as a property.
--
-- TODO: replace the placeholder region below (which currently covers EVERY
-- valid temperature, so the property is false for a perfectly normal patient)
-- with the correct NEWS2 threshold.
-- Hint: look at `shockNeverLow` above; the temperature version has the same shape.
-- Hint: the input is real-valued; the labelling rule truncates 35.09 to 35.0,
--       so the rule changes at 35.1 and the threshold `<= 35` sits a full
--       recording unit (0.1 degrees) inside the 3-point region.
@property
temperatureExtremesNeverLow : Bool
temperatureExtremesNeverLow = forall x .
  validPatient x and alertOnAir x and x ! temp <= 43 =>     -- TODO: 43 is a placeholder
    not (advises x low)

-- EXERCISE 2 — hypertensive crisis.
-- Systolic blood pressure of 220 mmHg or ABOVE also scores 3 points (the
-- upper end of the SBP scale; 111-219 scores 0). So far the specification
-- only covers the lower end (`shockNeverLow`, SBP <= 90).
--
-- TODO: replace the placeholder lower bound (40 = the whole valid range) with
-- the NEWS2 threshold, then verify on v1, v2 and v3. Does every model pass?
-- If one fails, use show_patient on the counterexample: is the patient's
-- charted SBP really >= 220, or is the counterexample sitting exactly on the
-- threshold where the model's learned boundary is a fraction of a unit off?
@property
hypertensiveCrisisNeverLow : Bool
hypertensiveCrisisNeverLow = forall x .
  validPatient x and alertOnAir x and x ! sbp >= 40 =>      -- TODO: 40 is a placeholder
    not (advises x low)

-- EXERCISE 3 — tighter (and looser) noise boxes.
-- No spec change is needed for this one: the epsilon values are @parameters,
-- passed on the command line (`-p epsSpo2:1` etc.). In the notebook, sweep
-- the noise scale for v3 and answer:
--   (a) At what scale do the two BOUNDARY patients (rows 10 and 11 of
--       data/triage-eval-patients.csv) first fail? Why is that not a bug?
--   (b) What is the largest scale at which all ten NON-boundary patients are
--       still robust on v3? And on v2?
--   (c) Optional: write `spo2Robust` below so that ONLY SpO2 varies (all other
--       components of d pinned to 0). Sweep epsSpo2 = 1, 2, 3, 4 on v3 to read
--       off, patient by patient, how far each one is from a decision boundary.
--
-- TODO (optional, part c): complete withinSpo2Noise. As written it lets
-- nothing vary at all, so spo2Robust is trivially true for every patient.
withinSpo2Noise : Patient -> Bool
withinSpo2Noise d =
  d ! rr == 0 and
  d ! spo2 == 0 and            -- TODO: allow -epsSpo2 <= d ! spo2 <= epsSpo2
  d ! sbp == 0 and
  d ! pulse == 0 and
  d ! temp == 0 and
  d ! o2 == 0 and d ! cvpu == 0

spo2RobustAround : Patient -> Index 3 -> Bool
spo2RobustAround x c = forall d .
  withinSpo2Noise d and validPatient (x + d) =>
    advises (x + d) c

@property
spo2Robust : Vector Bool n
spo2Robust = foreach i . spo2RobustAround (patients ! i) (labels ! i)

-- EXERCISE 4 — the relaxation trap.
-- `hypoxiaNeverLow` pins the oxygen flag to 0 and `hypoxiaOnOxygenNeverLow`
-- pins it to 1. It is tempting to merge them into one property by letting the
-- flag range over 0 <= o2 <= 1, as below. That is what many people write first.
--
-- TODO: run this property as written on triage-v3 (for which BOTH pinned
-- versions are proved). Look at the value of o2 in the counterexample.
-- Is that a patient who can exist? What went wrong, and what is the fix?
-- Hint: verifiers reason over real numbers; a "boolean" input is only
--       boolean if the specification says so, one value at a time.
@property
hypoxiaNeverLowAnyOxygen : Bool
hypoxiaNeverLowAnyOxygen = forall x .
  validPatient x and 0 <= x ! o2 <= 1 and x ! cvpu == 0 and x ! spo2 <= 91 =>   -- TODO: fix
    not (advises x low)
