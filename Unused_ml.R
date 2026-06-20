suppressPackageStartupMessages({
  library(dplyr); library(tidyr); library(purrr); library(readr); library(readxl)
  library(tidymodels); library(ranger); library(xgboost)
  library(bonsai); library(lightgbm)
})

PANEL_FILE <- "panel_unused.xlsx"
OUT_DIR    <- "."
TOPS <- c(0.25)
GRID <- c(RF = 50, XGBoost = 50, LightGBM = 50)

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

desc_dat <- build_panel(max(TOPS)) |> select(unused_rate, all_of(PREDICTORS))

desc_num <- desc_dat |>
  select(where(is.numeric)) |>
  pivot_longer(everything(), names_to = "variable", values_to = "value") |>
  group_by(variable) |>
  summarise(
    n        = sum(!is.na(value)),
    n_miss   = sum(is.na(value)),
    mean     = mean(value, na.rm = TRUE),
    sd       = sd(value,   na.rm = TRUE),
    min      = min(value,  na.rm = TRUE),
    p25      = quantile(value, 0.25, na.rm = TRUE),
    median   = median(value, na.rm = TRUE),
    p75      = quantile(value, 0.75, na.rm = TRUE),
    max      = max(value,  na.rm = TRUE),
    .groups  = "drop"
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

cat("\n=== Descriptive statistics — numeric ===\n\n");      print(as.data.frame(desc_num), row.names = FALSE)
cat("\n=== Descriptive statistics — categorical ===\n\n");  print(as.data.frame(desc_cat), row.names = FALSE)
write_csv(desc_num, file.path(OUT_DIR, "summary_descriptives_numeric.csv"))
write_csv(desc_cat, file.path(OUT_DIR, "summary_descriptives_categorical.csv"))

make_rec_tree <- function(tr, fml) {
  recipe(fml, data = tr) |>
    step_impute_mean(all_numeric_predictors()) |>
    step_impute_mode(all_nominal_predictors()) |>
    step_dummy(liberal_gov, covid, pre_election, capital, one_hot = TRUE) |>
    step_zv(all_predictors())
}

make_group_folds <- function(df) {
  set.seed(1227)
  rsample::group_vfold_cv(df, group = lafCd, v = 5)
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
    auc  <- yardstick::roc_auc_vec(df$y, prob, event_level = "first")
    prauc <- yardstick::pr_auc_vec(df$y, prob, event_level = "first")
    m05  <- confusion_metrics(df$y, prob, thr = 0.5)
    madj <- confusion_metrics(df$y, prob, thr = quantile(prob, 1 - prev))
    tibble(top_k = sprintf("%d%%", topk*100), set = setname, model = nm,
           Accuracy = m05["Accuracy"], Recall = m05["Recall"], Specificity = m05["Specificity"],
           Precision = m05["Precision"], F1 = m05["F1"],
           Recall_adj = madj["Recall"], Precision_adj = madj["Precision"], F1_adj = madj["F1"],
           ROC_AUC = auc, PR_AUC = prauc, n_pos = sum(df$y == "high"))
  }
  bind_rows(eval_set(tr, "train"), eval_set(te, "test"))
}

collapse_dummy <- function(nm) sub("^(region|covid|liberal_gov|pre_election|capital)_.*", "\\1", nm)

ml_specs <- function() list(
  RF       = rand_forest(trees = 500, mtry = tune(), min_n = tune()) |>
               set_mode("classification") |>
               set_engine("ranger", importance = "impurity", seed = 1227),
  XGBoost  = boost_tree(trees = 500, tree_depth = tune(), learn_rate = tune(),
                        min_n = tune(), loss_reduction = tune()) |>
               set_mode("classification") |>
               set_engine("xgboost", seed = 1227, nthread = 1),
  LightGBM = boost_tree(trees = 500, tree_depth = tune(), learn_rate = tune(),
                        min_n = tune(), loss_reduction = tune()) |>
               set_mode("classification") |>
               set_engine("lightgbm", seed = 1227, num_threads = 1)
)

fit_block <- function(topk) {
  pan <- build_panel(topk)
  tr  <- pan |> filter(split == "train"); te <- pan |> filter(split == "test")
  fml <- as.formula(paste("y ~", paste(PREDICTORS, collapse = " + ")))
  rec <- make_rec_tree(tr, fml)
  folds <- make_group_folds(tr); mset <- metric_set(pr_auc, roc_auc)
  fits <- imap(ml_specs(), function(sp, nm) {
    wf <- workflow() |> add_recipe(rec) |> add_model(sp)
    set.seed(1227)
    g  <- if (nm %in% names(GRID)) GRID[[nm]] else 50L
    tg <- tune_grid(wf, folds, grid = g, metrics = mset, control = control_grid())
    finalize_workflow(wf, select_best(tg, metric = "pr_auc")) |> fit(tr)
  })
  list(topk = topk, tr = tr, te = te, fits = fits)
}

blocks <- map(TOPS, fit_block)

perf <- map_dfr(blocks, function(b) {
  prev <- mean(b$tr$y == "high")
  map_dfr(names(b$fits), ~ perf_rows(b$fits[[.x]], b$tr, b$te, b$topk, .x, prev))
})
perf_print <- perf |>
  mutate(set = factor(set, levels = c("train", "test")),
         across(where(is.numeric), ~round(., 3))) |>
  arrange(top_k, set, desc(F1_adj))

cat("\n=== RF·XGBoost·LightGBM binary classification performance (municipality-grouped 80/20; *_adj = prevalence-adjusted threshold; tuning=pr_auc) ===\n\n")
print(as.data.frame(perf_print), row.names = FALSE)
write_csv(perf_print, file.path(OUT_DIR, "summary_ml_classification.csv"))

mdi <- map_dfr(blocks, function(b) {
  imp <- extract_fit_engine(b$fits[["RF"]])$variable.importance
  tibble(variable = names(imp), value = as.numeric(imp)) |>
    mutate(variable = collapse_dummy(variable)) |>
    group_by(variable) |> summarise(value = sum(value), .groups = "drop") |>
    mutate(top_k = sprintf("%d%%", b$topk*100), method = "RF_MDI",
           rel = value / sum(value))
})

shap_contrib <- function(fit_w, newdata) {
  booster <- extract_fit_engine(fit_w)
  X <- bake(extract_recipe(fit_w), new_data = newdata,
            all_predictors(), composition = "matrix")
  contrib <- if (inherits(booster, "lgb.Booster")) {
    -predict(booster, X, type = "contrib")
  } else {
    predict(booster, X, predcontrib = TRUE)
  }
  contrib <- contrib[, -ncol(contrib), drop = FALSE]
  colnames(contrib) <- colnames(X)
  list(X = X, contrib = contrib)
}

SHAP_MODELS <- tibble::tribble(
  ~fit,        ~method,     ~label,     ~file,
  "XGBoost",   "XGB_SHAP",  "XGBoost",  "xgb_shap",
  "LightGBM",  "LGBM_SHAP", "LightGBM", "lgbm_shap"
)

shap <- map_dfr(blocks, function(b) {
  pmap_dfr(SHAP_MODELS, function(fit, method, label, file) {
    ma <- colMeans(abs(shap_contrib(b$fits[[fit]], b$tr)$contrib))
    tibble(variable = names(ma), value = as.numeric(ma)) |>
      mutate(variable = collapse_dummy(variable)) |>
      group_by(variable) |> summarise(value = sum(value), .groups = "drop") |>
      mutate(top_k = sprintf("%d%%", b$topk*100), method = method,
             rel = value / sum(value))
  })
})

importance_all <- bind_rows(mdi, shap) |>
  mutate(rel = round(rel, 4)) |>
  arrange(method, top_k, desc(rel))

cat("\n=== Variable importance (RF_MDI / XGB·LGBM_SHAP; rel = sums to 1 within method·topk) ===\n\n")
print(as.data.frame(importance_all), row.names = FALSE)
write_csv(importance_all, file.path(OUT_DIR, "summary_ml_importance.csv"))

PLOT_DIR <- file.path(OUT_DIR, "plots")
dir.create(PLOT_DIR, showWarnings = FALSE)
fname <- function(x) gsub("%", "pct", x)

KFONT <- "AppleGothic"
update_geom_defaults("text", list(family = KFONT))

for (tk in unique(mdi$top_k)) {
  d <- mdi |> filter(top_k == tk)
  p <- ggplot(d, aes(x = rel, y = reorder(variable, rel))) +
    geom_col(fill = "#4C72B0") +
    geom_text(aes(label = sprintf("%.1f%%", rel * 100)), hjust = -0.15, size = 3) +
    scale_x_continuous(labels = scales::percent_format(accuracy = 1),
                       expand = expansion(mult = c(0, 0.12))) +
    labs(title = sprintf("Random Forest variable importance (MDI) — top %s", tk),
         subtitle = "Mean Decrease in Impurity, relative within method·threshold (sums to 1)",
         x = "Relative importance", y = NULL) +
    theme_minimal(base_size = 12, base_family = KFONT) +
    theme(panel.grid.major.y = element_blank())
  ggsave(file.path(PLOT_DIR, sprintf("rf_mdi_%s.png", fname(tk))),
         p, width = 7, height = 5, dpi = 300)
}

for (b in blocks) {
  tk <- sprintf("%d%%", b$topk * 100)
  pwalk(SHAP_MODELS, function(fit, method, label, file) {
    sc      <- shap_contrib(b$fits[[fit]], b$tr)
    X       <- sc$X; contrib <- sc$contrib
    vars    <- colnames(X)
    ord     <- names(sort(colMeans(abs(contrib[, vars, drop = FALSE])), decreasing = TRUE))
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
      labs(title = sprintf("%s SHAP summary — top %s", label, tk),
           subtitle = "point = municipality-year obs · x>0 pushes toward high (top unused) probability",
           x = "SHAP value (contribution to high prob)", y = NULL) +
      theme_minimal(base_size = 12, base_family = KFONT) +
      theme(panel.grid.major.y = element_blank())
    ggsave(file.path(PLOT_DIR, sprintf("%s_%s.png", file, fname(tk))),
           p, width = 7, height = 5, dpi = 300)
  })
}
