--------------------------------------------------------------------------------
-- Loan-approval network: ordinary (single-application) properties
--------------------------------------------------------------------------------

type Applicant = Tensor Real [6]
type Scores    = Tensor Real [2]

income   = 0   -- thousand pounds per year
debt     = 1   -- debt-to-income ratio, 0..1
history  = 2   -- years of credit history
defaults = 3   -- number of past defaults
age      = 4   -- years
prot     = 5   -- protected attribute, 0 or 1

deny    = 0   -- output index 0
approve = 1   -- output index 1 (the training label 1 meant "approved")

@network
loan : Applicant -> Scores

validApplicant : Applicant -> Bool
validApplicant a =
  10 <= a ! income   <= 200 and
  0  <= a ! debt     <= 1   and
  0  <= a ! history  <= 40  and
  0  <= a ! defaults <= 5   and
  18 <= a ! age      <= 90  and
  0  <= a ! prot     <= 1

approves : Applicant -> Bool
approves a = loan a ! approve > loan a ! deny

-- A region property: a plainly strong applicant is approved whatever the
-- protected attribute is (stated separately for each value of the flag).
strong : Applicant -> Bool
strong a =
  a ! income >= 90 and a ! debt <= 0.25 and a ! history >= 8 and a ! defaults == 0

@property
strongApplicantsApproved0 : Bool
strongApplicantsApproved0 = forall a . validApplicant a and strong a and a ! prot == 0 => approves a

@property
strongApplicantsApproved1 : Bool
strongApplicantsApproved1 = forall a . validApplicant a and strong a and a ! prot == 1 => approves a

-- Local robustness around real (synthetic) applicants: a small change in the
-- reported income must not flip the decision.
@parameter
epsIncome : Real   -- thousand pounds

@parameter(infer=True)
n : Nat

@dataset
applicants : Vector Applicant n

@dataset
decisions : Vector (Index 2) n

decides : Applicant -> Index 2 -> Bool
decides a c = forall j . j != c => loan a ! c > loan a ! j

robustAround : Applicant -> Index 2 -> Bool
robustAround a c = forall d .
  -epsIncome <= d ! income <= epsIncome and
  d ! debt == 0 and d ! history == 0 and d ! defaults == 0 and d ! age == 0 and d ! prot == 0 and
  validApplicant (a + d) =>
    decides (a + d) c

@property
incomeRobust : Vector Bool n
incomeRobust = foreach i . robustAround (applicants ! i) (decisions ! i)
