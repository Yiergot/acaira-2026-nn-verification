--------------------------------------------------------------------------------
-- Specification for a (synthetic, educational) neural triage network
--
-- The network takes seven clinical measurements and returns three scores:
-- low / medium / high urgency. The training labels follow the NHS NEWS2
-- early-warning bands (Royal College of Physicians, 2017; Scale 1 only).
-- Nothing here is a clinical tool. The point is to show how clinical
-- hard limits become *properties* that hold for every patient in a region,
-- not just for the patients we happened to test on.
--------------------------------------------------------------------------------

-- Inputs: a vector of 7 real numbers, in clinical units (no normalisation
-- here — it is folded into the network's first layer).
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
