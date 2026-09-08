--------------------------------------------------------------------------------
-- Fairness as a verifiable property (stretch exercise)
--
-- A loan-approval network f takes one applicant and returns two scores
-- (approve, deny). We would like to state:
--
--     for all applicants a, b that differ ONLY in the protected attribute,
--     f approves a  <=>  f approves b            ("counterfactual fairness")
--
-- That property applies the network twice, which query-based verifiers such
-- as Marabou cannot express. The trick: build a "twin" network that is two
-- weight-shared copies of f side by side (models/loan-twin-*.onnx). It takes
-- a pair (a ++ b) as ONE input of 12 numbers and returns (f a ++ f b) as ONE
-- output of 4 numbers. Now the fairness property is an ordinary property.
-- (Idea from Athavale et al., CAV 2024, "two-safety" properties.)
--------------------------------------------------------------------------------

type Pair   = Tensor Real [12]   -- applicant A (0..5) followed by applicant B (6..11)
type Scores = Tensor Real [4]    -- denyA, approveA, denyB, approveB

-- Applicant A occupies positions 0..5, applicant B positions 6..11.
incomeA   = 0   -- thousand pounds per year
debtA     = 1   -- debt-to-income ratio, 0..1
historyA  = 2   -- years of credit history
defaultsA = 3   -- number of past defaults
ageA      = 4   -- years
protA     = 5   -- protected attribute, 0 or 1

incomeB   = 6
debtB     = 7
historyB  = 8
defaultsB = 9
ageB      = 10
protB     = 11

denyA    = 0
approveA = 1
denyB    = 2
approveB = 3

@network
twin : Pair -> Scores

-- Plausible applicants (the same box for A and B).
validA : Pair -> Bool
validA p =
  10 <= p ! incomeA   <= 200 and
  0  <= p ! debtA     <= 1   and
  0  <= p ! historyA  <= 40  and
  0  <= p ! defaultsA <= 5   and
  18 <= p ! ageA      <= 90

validB : Pair -> Bool
validB p =
  10 <= p ! incomeB   <= 200 and
  0  <= p ! debtB     <= 1   and
  0  <= p ! historyB  <= 40  and
  0  <= p ! defaultsB <= 5   and
  18 <= p ! ageB      <= 90

-- A and B are identical except that A has protected = 0 and B has protected = 1.
sameExceptProtected : Pair -> Bool
sameExceptProtected p =
  p ! incomeA   == p ! incomeB   and
  p ! debtA     == p ! debtB     and
  p ! historyA  == p ! historyB  and
  p ! defaultsA == p ! defaultsB and
  p ! ageA      == p ! ageB      and
  p ! protA == 0 and p ! protB == 1

-- Verifiers work with closed sets: a strict "score1 > score2" is relaxed to
-- ">=", so a pair sitting exactly on the decision boundary would be reported
-- as a spurious counterexample. We therefore ask: if A is approved with at
-- least `margin` to spare, then B must be approved too.
@parameter
margin : Real

approvesClearlyA : Pair -> Bool
approvesClearlyA p = twin p ! approveA >= twin p ! denyA + margin

approvesClearlyB : Pair -> Bool
approvesClearlyB p = twin p ! approveB >= twin p ! denyB + margin

approvesA : Pair -> Bool
approvesA p = twin p ! approveA >= twin p ! denyA

approvesB : Pair -> Bool
approvesB p = twin p ! approveB >= twin p ! denyB

-- Two one-directional properties (each compiles to a single query).
-- A counterexample is a *pair* of applicants: same numbers, different flag,
-- different decision.
@property
fairAtoB : Bool
fairAtoB = forall p .
  validA p and validB p and sameExceptProtected p =>
    (approvesClearlyA p => approvesB p)

@property
fairBtoA : Bool
fairBtoA = forall p .
  validA p and validB p and sameExceptProtected p =>
    (approvesClearlyB p => approvesA p)
