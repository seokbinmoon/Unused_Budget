sink("Result_20.txt", split = TRUE)
suppressPackageStartupMessages({
  library(dplyr); library(tidyr); library(purrr); library(readxl)
  library(tidymodels); library(xgboost)
})

PANEL_FILE <- "panel_unused.xlsx"
PLOT_DIR   <- "plots"
TOPK       <- 0.20   # top 20%
GRID       <- 50

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
                      levels = c("high", "low"))) |>
    ungroup()
}

PREDICTORS <- c("fiscal_rate", "unused_rate_lag1",
                "log_grdp_pc", "log_perInc_pc", "maletofemale",
                "unemp", "ln_budget", "subsidy_rate", "yearend_spend_rate",
                "rapid_exec_rate",
                "admin_exp_rate", "event_exp_rate", "female_staff_ratio",
                "liberal_gov", "covid", "pre_election", "capital")


desc_dat <- build_panel(TOPK) |> select(unused_rate, all_of(PREDICTORS))

desc_num <- desc_dat |>
  select(where(is.numeric)) |>
  pivot_longer(everything(), names_to = "variable", values_to = "value") |>
  group_by(variable) |>
  summarise(
    n      = sum(!is.na(value)),
    n_miss = sum(is.na(value)),
    mean   = mean(value, na.rm = TRUE),
    sd     = sd(value,   na.rm = TRUE),
    min    = min(value,  na.rm = TRUE),
    p25    = quantile(value, 0.25, na.rm = TRUE),
    median = median(value, na.rm = TRUE),
    p75    = quantile(value, 0.75, na.rm = TRUE),
    max    = max(value,  na.rm = TRUE),
    .groups = "drop"
  ) |>
  mutate(across(where(is.numeric), ~round(., 3))) |>
  arrange(variable)

desc_cat <- desc_dat |>
  select(where(is.factor)) |>
  pivot_longer(everything(), names_to = "variable", values_to = "level",
               values_transform = list(level = as.character)) |>
  count(variable, level, name = "n") |>
  group_by(variable) |>
  mutate(prop = round(n / sum(n), 3)) |>
  ungroup() |>
  arrange(variable, level)

cat("\n=== Descriptive statistics — numeric ===\n\n");     print(as.data.frame(desc_num), row.names = FALSE)
cat("\n=== Descriptive statistics — categorical ===\n\n"); print(as.data.frame(desc_cat), row.names = FALSE)


pan <- build_panel(TOPK)
tr  <- pan |> filter(split == "train")
te  <- pan |> filter(split == "test")
fml <- as.formula(paste("y ~", paste(PREDICTORS, collapse = " + ")))

rec <- recipe(fml, data = tr) |>
  step_impute_mean(all_numeric_predictors()) |>
  step_impute_mode(all_nominal_predictors()) |>
  step_dummy(liberal_gov, covid, pre_election, capital, one_hot = TRUE) |>
  step_zv(all_predictors())

set.seed(1227)
folds <- group_vfold_cv(tr, group = lafCd, v = 5)

xgb_spec <- boost_tree(trees = 500, tree_depth = tune(), learn_rate = tune(),
                       min_n = tune(), loss_reduction = tune()) |>
  set_mode("classification") |>
  set_engine("xgboost", seed = 1227, nthread = 1)

wf <- workflow() |> add_recipe(rec) |> add_model(xgb_spec)
set.seed(1227)
tg <- tune_grid(wf, folds, grid = GRID,
                metrics = metric_set(pr_auc, roc_auc), control = control_grid())
fit_w <- finalize_workflow(wf, select_best(tg, metric = "pr_auc")) |> fit(tr)


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

prev <- mean(tr$y == "high")
eval_set <- function(df, setname) {
  prob <- predict(fit_w, df, type = "prob")$.pred_high
  m05  <- confusion_metrics(df$y, prob, thr = 0.5)
  madj <- confusion_metrics(df$y, prob, thr = quantile(prob, 1 - prev))
  tibble(top_k = sprintf("%d%%", TOPK*100), set = setname, model = "XGBoost",
         Accuracy = m05["Accuracy"], Recall = m05["Recall"], Specificity = m05["Specificity"],
         Precision = m05["Precision"], F1 = m05["F1"],
         Recall_adj = madj["Recall"], Precision_adj = madj["Precision"], F1_adj = madj["F1"],
         ROC_AUC = yardstick::roc_auc_vec(df$y, prob, event_level = "first"),
         PR_AUC  = yardstick::pr_auc_vec(df$y, prob, event_level = "first"),
         n_pos = sum(df$y == "high"))
}
perf <- bind_rows(eval_set(tr, "train"), eval_set(te, "test")) |>
  mutate(set = factor(set, levels = c("train", "test")),
         across(where(is.numeric), ~round(., 3)))

cat("\n=== XGBoost binary classification performance (municipality-grouped 80/20; *_adj = prevalence-adjusted threshold; tuning=pr_auc) ===\n\n")
print(as.data.frame(perf), row.names = FALSE)


collapse_dummy <- function(nm) sub("^(region|covid|liberal_gov|pre_election|capital)_.*", "\\1", nm)

booster <- extract_fit_engine(fit_w)
X       <- bake(extract_recipe(fit_w), new_data = tr, all_predictors(), composition = "matrix")
contrib <- predict(booster, X, predcontrib = TRUE)
contrib <- contrib[, -ncol(contrib), drop = FALSE]
colnames(contrib) <- colnames(X)

shap_imp <- tibble(variable = colnames(contrib), value = colMeans(abs(contrib))) |>
  mutate(variable = collapse_dummy(variable)) |>
  group_by(variable) |> summarise(value = sum(value), .groups = "drop") |>
  mutate(top_k = sprintf("%d%%", TOPK*100), rel = round(value / sum(value), 4)) |>
  arrange(desc(rel))

cat("\n=== XGBoost SHAP importance (rel = mean(|SHAP|), sums to 1) ===\n\n")
print(as.data.frame(shap_imp), row.names = FALSE)


dir.create(PLOT_DIR, showWarnings = FALSE)
KFONT <- "AppleGothic"
tk    <- sprintf("%d%%", TOPK*100)
vars  <- colnames(X)
ord   <- names(sort(colMeans(abs(contrib)), decreasing = TRUE))
shap_long <- map_dfr(vars, function(v) {
  fv <- X[, v]; fr <- range(fv, na.rm = TRUE)
  tibble(variable    = v,
         shap        = contrib[, v],
         feat_scaled = if (diff(fr) > 0) (fv - fr[1]) / diff(fr) else 0.5)
}) |>
  mutate(variable = factor(variable, levels = rev(ord)))

p <- ggplot(shap_long, aes(x = shap, y = variable, color = feat_scaled)) +
  geom_vline(xintercept = 0, linewidth = 0.3, color = "grey60") +
  geom_jitter(height = 0.2, width = 0, alpha = 0.45, size = 1) +
  scale_color_gradient(low = "#3B4CC0", high = "#B40426",
                       breaks = c(0, 1), labels = c("Low", "High"), name = "Feature value") +
  labs(title = sprintf("XGBoost SHAP summary — top %s", tk),
       subtitle = "point = municipality-year obs · x>0 pushes toward high (top unused) probability",
       x = "SHAP value (contribution to high prob)", y = NULL) +
  theme_minimal(base_size = 12, base_family = KFONT) +
  theme(panel.grid.major.y = element_blank())
ggsave(file.path(PLOT_DIR, sprintf("xgb_shap_%s.png", gsub("%", "pct", tk))),
       p, width = 7, height = 5, dpi = 300)

sink()
