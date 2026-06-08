library(randomForest)
library(caret)
library(dplyr)
library(VIM)
library(vip)
library(xgboost)
library(rpart)


dane_stroke = read.csv2("zaj3_Stroke.csv", dec = ".")

dane_stroke$work_type = as.factor(dane_stroke$work_type)
dane_stroke$smoking_status = as.factor(dane_stroke$smoking_status)
dane_stroke$gender = as.factor(dane_stroke$gender)
dane_stroke$age = as.numeric(dane_stroke$age)
dane_stroke$avg_glucose_level = as.numeric(gsub(",", ".",dane_stroke$avg_glucose_level))
dane_stroke$bmi = as.numeric(gsub(",", ".",dane_stroke$bmi))
dane_stroke$hypertension = as.factor(dane_stroke$hypertension)
dane_stroke$heart_disease = as.factor(dane_stroke$heart_disease)
dane_stroke$ever_married = as.factor(dane_stroke$ever_married)
dane_stroke$Residence_type = as.factor(dane_stroke$Residence_type)
dane_stroke$stroke = as.factor(dane_stroke$stroke)

#braki danych uzupełnione knn
set.seed(123)
dane_stroke_final = kNN(dane_stroke, k = 5, imp_var = FALSE)

summary(dane_stroke_final)
summary(dane_stroke)


#Podział na zbiór uczący i testowy
set.seed(123)
index = sample(nrow(dane_stroke_final), size = 0.2*nrow(dane_stroke_final), replace = F)
testowy = dane_stroke_final[index,]
uczacy = dane_stroke_final[-index,]


p = ncol(uczacy) - 1
predictors = setdiff(names(uczacy), "stroke")

# funkcja do metryk
calc_metrics = function(true, pred, positive_level = "1") {
  cm = caret::confusionMatrix(pred, true, positive = positive_level)
  out = c(
    Accuracy = as.numeric(cm$overall["Accuracy"]),
    Sensitivity = as.numeric(cm$byClass["Sensitivity"]),
    Specificity = as.numeric(cm$byClass["Specificity"])
  )
  return(out)
}

ntree_list = c(10, 20, 30, 40, 50, 100, 200)
mtry_list = c(2, 3, 5, 7, 9, p)
nodesize_list = c(1, 5, 10)


#Testowanie
set.seed(123)
results <- data.frame()

for (nt in ntree_list) {
  for (mt in mtry_list) {
    for (ns in nodesize_list) {
      
      cat("Trenuję model ntree =", nt,
          "mtry =", mt,
          "nodesize =", ns, "\n")
      
      rf = randomForest(
        stroke ~ .,
        data = uczacy,
        ntree = nt,
        mtry = mt,
        nodesize = ns,
        importance = TRUE
      )
      
      # predykcje
      pred_train = predict(rf, uczacy)
      pred_test = predict(rf, testowy)
      
      # metryki
      met_train = calc_metrics(uczacy$stroke, pred_train)
      met_test = calc_metrics(testowy$stroke, pred_test)
      
      # zapis
      results = rbind(results, data.frame(
        ntree = nt,
        mtry = mt,
        nodesize = ns,
        Accuracy_train = met_train["Accuracy"],
        Sensitivity_train = met_train["Sensitivity"],
        Specificity_train = met_train["Specificity"],
        Accuracy_test  = met_test["Accuracy"],
        Sensitivity_test = met_test["Sensitivity"],
        Specificity_test = met_test["Specificity"]
      ))
    }
  }
}

#Najlepsze modele
#według dokładności
results %>% arrange(desc(Accuracy_test)) %>% head(3)

#według średniej z czułości i specyficzności
results %>%
  mutate(mean_SS = (Sensitivity_test + Specificity_test) / 2) %>%
  arrange(desc(mean_SS)) %>%
  head(3)

#Najlepszy dla ntree=20, mtry=3, nodesize=10
rf_best = randomForest(
  stroke ~ .,
  data = uczacy,
  ntree = 20,
  mtry = 3,
  nodesize = 10,
  importance = TRUE
)

pred_uczacy = predict(rf_best, uczacy)
pred_testowy = predict(rf_best, testowy)

#Ocena ważności zmiennych
importance(rf_best)
varImpPlot(rf_best)
vip(rf_best)
#Zmienna gender, heart_disease, ever_married i Residence_type negatywnie wpływają na model, największy wpływ ma wiek



#Boosting

calc_metrics_boost = function(true, prob, positive_level = "1", cutoff = 0.5) {
  # zamiana na 0/1 na factor
  pred = ifelse(prob >= cutoff, 1, 0)
  pred_factor = factor(pred, levels = c(0,1))
  true_factor = factor(true, levels = c(0,1))
  
  cm = caret::confusionMatrix(pred_factor, true_factor, positive = positive_level)
  
  out = c(
    Accuracy = as.numeric(cm$overall["Accuracy"]),
    Sensitivity = as.numeric(cm$byClass["Sensitivity"]),
    Specificity = as.numeric(cm$byClass["Specificity"])
  )
  return(out)
}

# Matryce dla xgboost
train_matrix = model.matrix(stroke ~ ., uczacy)[, -1]
test_matrix = model.matrix(stroke ~ ., testowy)[, -1]

train_label = as.numeric(as.character(uczacy$stroke))
test_label = as.numeric(as.character(testowy$stroke))

dtrain = xgb.DMatrix(data = train_matrix, label = train_label)
dtest = xgb.DMatrix(data = test_matrix,  label = test_label)

#Siatka hiperparametrów
grid = expand.grid(
  eta = c(0.01, 0.05, 0.1),
  max_depth = c(2, 3, 5),
  subsample = c(0.5, 0.7, 1.0),
  colsample_bytree = c(0.5, 0.7, 1.0),
  nrounds = c(100)#, 200, 500
)

#zapewniamy powtarzalność wyników
set.seed(123)

results_boost <- data.frame(
  eta = numeric(),
  max_depth = numeric(),
  subsample = numeric(),
  colsample_bytree = numeric(),
  nrounds = numeric(),
  Accuracy_train = numeric(),
  Sensitivity_train = numeric(),
  Specificity_train = numeric(),
  Accuracy_test = numeric(),
  Sensitivity_test = numeric(),
  Specificity_test = numeric(),
  stringsAsFactors = FALSE
)

for(i in 1:nrow(grid)) {
  pars = grid[i, ]
  
  cat("Trenuję model", i, "z", nrow(grid), "\n")
  
  model = xgb.train(
    params = list(
      objective = "binary:logistic",
      eval_metric = "logloss",
      eta = pars$eta,
      max_depth = pars$max_depth,
      subsample = pars$subsample,
      colsample_bytree = pars$colsample_bytree,
      seed = 123
    ),
    data = dtrain,
    nrounds = pars$nrounds,
    verbose = 0
  )
  
  # predykcje
  prob_train = predict(model, dtrain)
  prob_test = predict(model, dtest)
  
  met_train = calc_metrics_boost(train_label, prob_train)
  met_test = calc_metrics_boost(test_label, prob_test)
  
  results_boost = rbind(results_boost, data.frame(
    eta = pars$eta,
    max_depth = pars$max_depth,
    subsample = pars$subsample,
    colsample_bytree = pars$colsample_bytree,
    nrounds = pars$nrounds,
    Accuracy_train = met_train["Accuracy"],
    Sensitivity_train = met_train["Sensitivity"],
    Specificity_train = met_train["Specificity"],
    Accuracy_test = met_test["Accuracy"],
    Sensitivity_test = met_test["Sensitivity"],
    Specificity_test = met_test["Specificity"]
  ))
}

#Najwyższa dokładność
results_boost %>% arrange(desc(Accuracy_test)) %>% head(3)

#Najwyższa średnia czułości i specyficzności
results_boost %>%
  mutate(SS_mean = (Sensitivity_test + Specificity_test)/2) %>%
  arrange(desc(SS_mean)) %>% head(3)

#Najlepsze wyniki dla boostingu: eta=0.01, max_depth=2, subsample=1, colsample_bytree=1, nrounds=100
best_params = list(
  objective = "binary:logistic",
  eval_metric = "logloss",
  eta = 0.01,
  max_depth = 2,
  subsample = 1,
  colsample_bytree = 1
)

best_nrounds = 100

best_model_boost = xgb.train(
  params = best_params,
  data = dtrain,
  nrounds = best_nrounds,
  verbose = 1
)

prob_train_boost = predict(best_model_boost, dtrain)
prob_test_boost = predict(best_model_boost, dtest)

metrics_train = calc_metrics_boost(train_label, prob_train_boost)
metrics_test = calc_metrics_boost(test_label, prob_test_boost)

metrics_train
metrics_test

vip(best_model_boost)


#Porównanie
pred_test_boost_class = ifelse(prob_test_boost >= 0.5, 1, 0)

#Drzewo decyzyjne
best_model_dt = rpart(stroke ~.,
               data = uczacy,
               method = "class",
               control = rpart.control(maxdepth = 4))
#Bagging
best_bagging <- randomForest(
  formula = stroke ~ .,
  data = uczacy,
  ntree = 10,
  mtry = 10,
  sampsize = floor(nrow(uczacy) * 0.5),
  replace = TRUE
)
#Las losowy
confusionMatrix(pred_testowy, testowy$stroke, positive = "1")
#Boosting
confusionMatrix(factor(pred_test_boost_class, levels=c(0,1)), factor(testowy$stroke, levels = c(0,1)), positive = "1")
#Znacznie lepsza czułość w modelu boosting


#Tabela
prob_tree= predict(best_model_dt, newdata = testowy, type = "prob")[,2]
prob_bagging = predict(best_bagging, newdata = testowy, type = "prob")[,2]
prob_rf = predict(rf_best, newdata = testowy, type = "prob")[,2]
prob_boost = predict(best_model_boost, newdata = dtest)

metrics_tree = calc_metrics_boost(testowy$stroke, prob_tree)
metrics_bagging = calc_metrics_boost(testowy$stroke, prob_bagging)
metrics_rf = calc_metrics_boost(testowy$stroke, prob_rf)
metrics_boost = calc_metrics_boost(testowy$stroke, prob_boost)

results_summary = bind_rows(
  data.frame(Method = "Decision Tree", t(metrics_tree)),
  data.frame(Method = "Bagging", t(metrics_bagging)),
  data.frame(Method = "Random Forest", t(metrics_rf)),
  data.frame(Method = "Boosting", t(metrics_boost))
)

results_summary
