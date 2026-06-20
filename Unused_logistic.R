suppressPackageStartupMessages({
  library(dplyr); library(purrr); library(readr); library(readxl)
  library(tidymodels)
  library(sandwich); library(lmtest)
})

PANEL_FILE <- "panel_unused.xlsx"
OUT_DIR    <- "."
TOPS <- c(0.25)

panel_wide <- read_excel(PANEL_FILE)
cat(sprintf("[Panel] %d rows × %d cols (%d-%d)\n",
            nrow(panel_wide), ncol(panel_wide),
            min(panel_wide$fyr), max(panel_wide$fyr)))
cat(sprintf("[Missing] liberal_gov NA %d (mode-imputed in recipe)\n",
            sum(is.na(panel_wide$liberal_gov))))

set.seed(1227)
test_munis <- sample(unique(panel_wide$lafCd),
                     size = round(0.2 * n_distinct(panel_wide$lafCd)))
panel_wide <- panel_wide |>
  mutate(split = if_else(lafCd %in% test_munis, "test", "train"))
cat(sprintf("[Split] municipality-grouped 80/20 — train %d rows(%d) / test %d rows(%d)\n",
            sum(panel_wide$split == "train"),
            n_distinct(panel_wide$lafCd) - length(test_munis),
            sum(panel_wide$split == "test"), length(test_munis)))

build_panel <- function(topk) {
  panel_wide |>
    transmute(
      fyr, lafCd, region = factor(region), split,
      capital = factor(if_else(region %in% c("서울", "인천", "경기"), "capital", "noncapital"),
                       levels = c("noncapital", "capital")),
      unused_rate,
      unused_rate_lag1,
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
      liberal_gov   = factor(liberal_gov, levels = c(0, 1), labels = c("conserv", "liberal")),
      covid = factor(if_else(fyr %in% 2020:2022, "covid", "normal"),
                     levels = c("normal", "covid")),
      pre_election = factor(if_else(fyr %in% c(2017, 2021), "pre_elec", "other"),
                            levels = c("other", "pre_elec"))
    ) |>
    filter(between(unused_rate, 0, 1)) |>
    group_by(fyr) |>
    mutate(y = factor(if_else(unused_rate >= quantile(unused_rate, 1 - topk, na.rm = TRUE),
                              "high", "low"),
                      levels = c("low", "high"))) |>
    ungroup()
}

PREDICTORS <- c("fiscal_rate", "unused_rate_lag1",
                "log_grdp_pc", "log_perInc_pc", "maletofemale",
                "unemp", "ln_budget", "subsidy_rate", "yearend_spend_rate",
                "rapid_exec_rate",
                "admin_exp_rate", "event_exp_rate", "female_staff_ratio",
                "liberal_gov", "covid", "pre_election", "capital")

make_rec_lin <- function(tr, fml) {
  recipe(fml, data = tr) |>
    step_impute_mean(all_numeric_predictors()) |>
    step_impute_mode(all_nominal_predictors()) |>
    step_dummy(liberal_gov, covid, pre_election, capital) |>
    step_zv(all_predictors()) |>
    step_lincomb(all_numeric_predictors()) |>
    step_normalize(all_numeric_predictors())
}

confusion_metrics <- function(truth, prob, thr) {
  pred <- factor(if_else(prob >= thr, "high", "low"), levels = c("high", "low"))
  TP <- sum(pred=="high" & truth=="high"); FP <- sum(pred=="high" & truth=="low")
  FN <- sum(pred=="low"  & truth=="high"); TN <- sum(pred=="low"  & truth=="low")
  rec  <- TP/(TP+FN); spec <- TN/(TN+FP)
  prec <- if (TP+FP > 0) TP/(TP+FP) else NA_real_
  acc  <- (TP+TN)/length(truth)
  f1   <- if (!is.na(prec) && prec+rec > 0) 2*prec*rec/(prec+rec) else NA_real_
  c(Accuracy=acc, Recall=rec, Specificity=spec, Precision=prec, F1=f1)
}

perf_rows <- function(fit_w, tr, te, topk, nm, prev) {
  eval_set <- function(df, setname) {
    prob <- predict(fit_w, df, type = "prob")$.pred_high
    auc  <- yardstick::roc_auc_vec(df$y, prob, event_level = "second")
    m05  <- confusion_metrics(df$y, prob, thr = 0.5)
    madj <- confusion_metrics(df$y, prob, thr = quantile(prob, 1 - prev))
    tibble(top_k = sprintf("%d%%", topk*100), set = setname, model = nm,
           Accuracy = m05["Accuracy"], Recall = m05["Recall"], Specificity = m05["Specificity"],
           Precision = m05["Precision"], F1 = m05["F1"],
           Recall_adj = madj["Recall"], Precision_adj = madj["Precision"], F1_adj = madj["F1"],
           ROC_AUC = auc, n_pos = sum(df$y == "high"))
  }
  bind_rows(eval_set(tr, "train"), eval_set(te, "test"))
}

fit_block <- function(topk) {
  pan <- build_panel(topk)
  tr  <- pan |> filter(split == "train"); te <- pan |> filter(split == "test")
  fml <- as.formula(paste("y ~", paste(PREDICTORS, collapse = " + ")))
  rec <- make_rec_lin(tr, fml)
  wf    <- workflow() |> add_recipe(rec) |> add_model(logistic_reg() |> set_engine("glm"))
  fit_w <- fit(wf, tr)
  list(topk = topk, tr = tr, te = te, fit_w = fit_w)
}

blocks <- map(TOPS, fit_block)

perf <- map_dfr(blocks, function(b)
  perf_rows(b$fit_w, b$tr, b$te, b$topk, "Logistic", mean(b$tr$y == "high")))
perf_print <- perf |>
  mutate(set = factor(set, levels = c("train", "test")),
         across(where(is.numeric), ~round(., 3))) |>
  arrange(top_k, set)

cat("\n=== Logistic binary classification performance (*_adj = prevalence-adjusted threshold) ===\n\n")
print(as.data.frame(perf_print), row.names = FALSE)
write_csv(perf_print, file.path(OUT_DIR, "summary_logistic_classification.csv"))

coefs <- map_dfr(blocks, function(b) {
  glm_fit <- extract_fit_engine(b$fit_w)
  cl_id   <- b$tr$lafCd[as.integer(rownames(model.frame(glm_fit)))]
  cl_vcov <- sandwich::vcovCL(glm_fit, cluster = cl_id, type = "HC1")
  ct      <- lmtest::coeftest(glm_fit, vcov. = cl_vcov)
  tibble(top_k     = sprintf("%d%%", b$topk*100),
         term      = rownames(ct),
         estimate  = ct[, "Estimate"],
         std.error = ct[, "Std. Error"],
         statistic = ct[, "z value"],
         p.value   = ct[, "Pr(>|z|)"],
         odds_ratio = exp(ct[, "Estimate"]))
}) |>
  arrange(top_k, desc(abs(estimate)))

cat("\n=== Logistic standardized coefficients / odds ratios (SE = lafCd cluster-robust; positive coef = higher prob of high unused) ===\n\n")
print(as.data.frame(coefs |> mutate(across(where(is.numeric), ~round(., 4)))), row.names = FALSE)
write_csv(coefs, file.path(OUT_DIR, "summary_logistic_coef.csv"))

vif_tbl <- map_dfr(blocks, function(b) {
  glm_fit <- extract_fit_engine(b$fit_w)
  v <- car::vif(glm_fit)
  if (is.matrix(v)) {
    tibble(top_k = sprintf("%d%%", b$topk*100),
           term  = rownames(v),
           GVIF  = v[, "GVIF"],
           Df    = v[, "Df"],
           VIF   = v[, "GVIF^(1/(2*Df))"]^2)
  } else {
    tibble(top_k = sprintf("%d%%", b$topk*100),
           term  = names(v), GVIF = as.numeric(v), Df = 1, VIF = as.numeric(v))
  }
}) |>
  arrange(top_k, desc(VIF))

print(as.data.frame(vif_tbl |> mutate(across(where(is.numeric), ~round(., 3)))), row.names = FALSE)
write_csv(vif_tbl, file.path(OUT_DIR, "summary_logistic_vif.csv"))

cat("\n[Done] saved summary_logistic_classification.csv / summary_logistic_coef.csv / summary_logistic_vif.csv\n")
