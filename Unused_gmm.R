sink("Result_gmm.txt", split = TRUE)
suppressPackageStartupMessages({
  library(dplyr); library(readxl); library(plm)
})

panel <- read_excel("panel_unused.xlsx") |>
  transmute(
    fyr, lafCd,
    unused_rate,
    fiscal_rate   = fin_indep_rate,
    log_grdp_pc   = log(r_grdp_pc),
    log_perInc_pc = log(r_perInc_pc),
    maletofemale,
    unemp,
    ln_budget,
    subsidy_rate,
    yearend_spend_rate,
    rapid_exec_rate,
    admin_exp_rate,
    event_exp_rate,
    female_staff_ratio = staff_female / staff_total,
    liberal_d = as.numeric(liberal_gov == 1),
    capital_d = as.numeric(region %in% c("서울", "인천", "경기"))
  ) |>
  filter(unused_rate >= 0, unused_rate <= 1) |>
  arrange(lafCd, fyr)

pdat <- pdata.frame(panel, index = c("lafCd", "fyr"))

X   <- c("fiscal_rate", "log_grdp_pc", "log_perInc_pc", "maletofemale", "unemp",
         "ln_budget", "subsidy_rate", "yearend_spend_rate", "rapid_exec_rate",
         "admin_exp_rate", "event_exp_rate", "female_staff_ratio",
         "liberal_d", "capital_d")
rhs <- paste(X, collapse = " + ")
fml <- as.formula(sprintf(
  "unused_rate ~ lag(unused_rate, 1) + %s | lag(unused_rate, 2:4) | %s", rhs, rhs))

sgmm <- pgmm(fml, data = pdat,
             effect = "twoways", model = "twosteps",
             transformation = "ld", collapse = TRUE)

print(summary(sgmm, robust = TRUE))
cat("\n[AR(1) p<0.05]"); print(mtest(sgmm, order = 1, vcov = vcovHC))
cat("\n[AR(2) p>0.05]"); print(mtest(sgmm, order = 2, vcov = vcovHC))
cat("\n[Sargan p>0.05]"); print(sargan(sgmm))
sink()
