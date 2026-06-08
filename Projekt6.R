library(caret)
library(dplyr)
library(VIM)
library(vip)
library(e1071)
library(ggplot2)
library(tidyr)


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

#metryki
calc_metrics = function(true, pred, set_name) {
  cm = caret::confusionMatrix(pred, true, positive = "1")
  
  data.frame(
    set = set_name,
    Accuracy = cm$overall["Accuracy"],
    Sensitivity = cm$byClass["Sensitivity"],
    Specificity = cm$byClass["Specificity"]
  )
}


#Algorytm SVM
#=============
#ZADANIE 1A
#jądro liniowe
grid_C = C = c(0.01, 0.1, 0.3, 0.5, 0.8, 1, 3, 10)

results_linear <- data.frame()

for(cval in grid_C) {
  model = svm(stroke ~ ., data = uczacy,
               kernel = "linear", cost = cval)
  
  pred_test = predict(model, testowy)
  pred_train = predict(model, uczacy)
  
  metrics_test <- calc_metrics(testowy$stroke, pred_test, set_name = "test")
  metrics_train <- calc_metrics(uczacy$stroke, pred_train, set_name = "train")
  
  results_linear <- rbind(results_linear,
                          cbind(C = cval, metrics_train),
                          cbind(C = cval, metrics_test))
}

results_linear %>% 
  filter(set == "test") %>%
  mutate(średnia = rowMeans(cbind(Sensitivity, Specificity)))%>%
  arrange(desc(średnia)) %>%
  head(5)

results_linear %>% 
  filter(set == "test") %>%
  arrange(desc(Sensitivity))

results_linear %>% 
  filter(set == "test") %>%
  arrange(desc(Accuracy))

#Wykres
results_long_lin = results_linear %>%
  pivot_longer(cols = starts_with(c("Accuracy", "Sensitivity", "Specificity")),
               names_to = "metric",
               values_to = "value")

# Rysujemy osobno dla train/test (facet) i oddzielnie dla metric
ggplot(results_long_lin, aes(x = C, y = value, color = set)) +
  geom_line() + 
  geom_point() +
  facet_wrap(~metric, scales = "fixed") +
  labs(title = "SVM - jądro liniowe, parametr C",
       x = "Parametr C",
       y = "Wartość metryki",
       color = "Zbiór") +
  theme_minimal()

#==============
#ZADANIE 1B
#jądro wielomianowe
grid_poly <- expand.grid(
  C = c(0.1, 0.5, 1),
  degree = c(2, 3, 4),
  gamma = c(0.01, 0.1, 1),
  coef0 = c(0, 1)
)

results_poly <- data.frame()

for (i in 1:nrow(grid_poly)) {
  pars <- grid_poly[i, ]
  
  model <- svm(stroke ~ ., data = uczacy,
               kernel = "polynomial",
               cost = pars$C,
               degree = pars$degree,
               gamma = pars$gamma,
               coef0 = pars$coef0)
  
  pred_test = predict(model, testowy)
  pred_train = predict(model, uczacy)
  
  metrics_test <- calc_metrics(testowy$stroke, pred_test, set_name = "test")
  metrics_train <- calc_metrics(uczacy$stroke, pred_train, set_name = "train")
  
  results_poly <- rbind(results_poly, 
                        cbind(pars, metrics_train),
                        cbind(pars, metrics_test))
}

results_poly %>% 
  filter(set == "test") %>%
  mutate(średnia = rowMeans(cbind(Sensitivity, Specificity)))%>%
  arrange(desc(średnia)) %>%
  head(5)

results_poly %>% 
  filter(set == "test") %>%
  arrange(desc(Sensitivity))%>%
  head(5)

results_poly %>% 
  filter(set == "test") %>%
  arrange(desc(Accuracy)) %>%
  head(5)

#Wykres Dokładności (zbiór treningowy)
heatmap_poly_acc_train = results_poly %>%
  filter(set == "train") %>%
  group_by(C, degree, gamma, coef0) %>%
  summarise(Accuracy = mean(Accuracy), .groups = "drop")

ggplot(heatmap_poly_acc_train, aes(x = factor(gamma), 
                             y = factor(degree), 
                             fill = Accuracy)) +
  geom_tile() +
  geom_text(aes(label = round(Accuracy, 3)), color = "white") +
  scale_fill_viridis_c() +
  labs(
    title = "Heatmap Dokładność: Degree, Gamma, C, Coef0",
    x = "gamma",
    y = "degree",
    fill = "średnia dokładność"
  ) +
  facet_wrap(coef0 ~ C, scales = "fixed") +
  theme_minimal()

#Wykres Dokładności (zbiór testowy)
heatmap_poly_acc_test = results_poly %>%
  filter(set == "test") %>%
  group_by(C, degree, gamma, coef0) %>%
  summarise(Accuracy = mean(Accuracy), .groups = "drop")

ggplot(heatmap_poly_acc_test, aes(x = factor(gamma), 
                         y = factor(degree), 
                         fill = Accuracy)) +
  geom_tile() +
  geom_text(aes(label = round(Accuracy, 3)), color = "white") +
  scale_fill_viridis_c() +
  labs(
    title = "Heatmap Dokładność: Degree, Gamma, C, Coef0",
    x = "gamma",
    y = "degree",
    fill = "średnia dokładność"
  ) +
  facet_wrap(coef0 ~ C, scales = "fixed") +
  theme_minimal()


#Wykres Czułości (zbiór treningowy)
heatmap_poly_sen_train = results_poly %>%
  filter(set == "train") %>%
  group_by(C, degree, gamma, coef0) %>%
  summarise(Sensitivity = mean(Sensitivity), .groups = "drop")

ggplot(heatmap_poly_sen_train, aes(x = factor(gamma), 
                             y = factor(degree), 
                             fill = Sensitivity)) +
  geom_tile() +
  geom_text(aes(label = round(Sensitivity, 3)), color = "white") +
  scale_fill_viridis_c() +
  labs(
    title = "Heatmap Czułość: Degree, Gamma, C, Coef0",
    x = "gamma",
    y = "degree",
    fill = "średnia czułość"
  ) +
  facet_wrap(coef0 ~ C, scales = "fixed") +
  theme_minimal()


#Wykres Czułości (zbiór testowy)
heatmap_poly_sen_test = results_poly %>%
  filter(set == "test") %>%
  group_by(C, degree, gamma, coef0) %>%
  summarise(Sensitivity = mean(Sensitivity), .groups = "drop")

ggplot(heatmap_poly_sen_test, aes(x = factor(gamma), 
                             y = factor(degree), 
                             fill = Sensitivity)) +
  geom_tile() +
  geom_text(aes(label = round(Sensitivity, 3)), color = "white") +
  scale_fill_viridis_c() +
  labs(
    title = "Heatmap Czułość: Degree, Gamma, C, Coef0",
    x = "gamma",
    y = "degree",
    fill = "średnia czułość"
  ) +
  facet_wrap(coef0 ~ C, scales = "fixed") +
  theme_minimal()


#================
#ZADANIE 1C
#jądro radialne
grid_rbf <- expand.grid(
  C = c(0.1, 0.3, 0.5, 1, 3, 10),
  gamma = c(0.001, 0.01, 0.1, 1)
)

results_rbf <- data.frame()

for (i in 1:nrow(grid_rbf)) {
  pars <- grid_rbf[i, ]
  
  model <- svm(stroke ~ ., data = uczacy,
               kernel = "radial",
               cost = pars$C,
               gamma = pars$gamma)
  
  pred_test = predict(model, testowy)
  pred_train = predict(model, uczacy)
  
  metrics_test <- calc_metrics(testowy$stroke, pred_test, set_name = "test")
  metrics_train <- calc_metrics(uczacy$stroke, pred_train, set_name = "train")
  
  results_rbf <- rbind(results_rbf, 
                       cbind(pars, metrics_train),
                       cbind(pars, metrics_test))
}

results_rbf %>% 
  filter(set == "test") %>%
  mutate(średnia = rowMeans(cbind(Sensitivity, Specificity)))%>%
  arrange(desc(średnia)) %>%
  head(5)

results_rbf %>% 
  filter(set == "test") %>%
  arrange(desc(Accuracy)) %>%
  head(5)

results_rbf %>% 
  filter(set == "test") %>%
  arrange(desc(Sensitivity)) %>%
  head(10)

#Wykres
results_rbf_long = results_rbf %>%
  pivot_longer(
    cols = c(Accuracy, Sensitivity, Specificity),
    names_to = "metric",
    values_to = "value"
  )

ggplot(results_rbf_long,
       aes(x = C, y = value, color = set)) +
  geom_line() +
  geom_point() +
  facet_grid(metric ~ gamma, scales = "fixed") +
  labs(
    title = "SVM – jądro radialne, parametr C i gamma",
    x = "Parametr C",
    y = "Wartość metryki",
    color = "Zbiór"
  ) +
  theme_minimal()


#===============
#ZADANIE 2

model_log = glm(stroke ~ ., data = uczacy, family = binomial)

# predykcje prawdopodobieństw
pred_prob_test = predict(model_log, testowy, type = "response")

# konwersja na klasy (próg 0.5, bo zbiór jest zbalansowany po 50%)
pred_test = factor(ifelse(pred_prob_test  > 0.5, 1, 0))

#Obliczenie miar klasyfikacji
metrics_reg_log = calc_metrics(testowy$stroke, pred_test, set_name="test")


#Porównywanie
get_best_svm = function(results_svm, uczacy, testowy, kernel_name, criterion="Accuracy") {
  if(criterion == "Accuracy") {
    best_row = results_svm %>%
      filter(set == "test") %>%
      arrange(desc(Accuracy)) %>%
      slice(1)
  } else if(criterion == "Mean") {
    best_row = results_svm %>%
      filter(set == "test") %>%
      mutate(Mean = (Sensitivity + Specificity)/2) %>%
      arrange(desc(Mean)) %>%
      slice(1)
  }
  
  # dopasowanie modelu SVM
  if(kernel_name == "linear") {
    model = svm(stroke ~ ., data = uczacy,
                kernel="linear", cost=best_row$C)
  } else if(kernel_name == "polynomial") {
    model = svm(stroke ~ ., data = uczacy,
                kernel="polynomial",
                cost=best_row$C,
                degree=best_row$degree,
                gamma=best_row$gamma,
                coef0=best_row$coef0)
  } else if(kernel_name == "radial") {
    model = svm(stroke ~ ., data = uczacy,
                kernel="radial",
                cost=best_row$C,
                gamma=best_row$gamma)
  }
  
  # predykcje
  pred = predict(model, newdata = testowy)
  
  metrics = calc_metrics(testowy$stroke, pred, "test")
  return(metrics)
}

#dokładność
metrics_svm_linear_acc = get_best_svm(results_linear, uczacy, testowy, "linear", criterion="Accuracy")
metrics_svm_poly_acc = get_best_svm(results_poly, uczacy, testowy, "polynomial", criterion="Accuracy")
metrics_svm_rbf_acc = get_best_svm(results_rbf, uczacy, testowy, "radial", criterion="Accuracy")

#średnia czułości i specyficzności
metrics_svm_linear_cs = get_best_svm(results_linear, uczacy, testowy, "linear", criterion="Mean")
metrics_svm_poly_cs = get_best_svm(results_poly, uczacy, testowy, "polynomial", criterion="Mean")
metrics_svm_rbf_cs = get_best_svm(results_rbf, uczacy, testowy, "radial", criterion="Mean")


#Tabelka dokładność
results_summary_acc = bind_rows(
  cbind(Method = "SVM_linear", metrics_svm_linear_acc[, c("Accuracy","Sensitivity","Specificity")]),
  cbind(Method = "SVM_poly",   metrics_svm_poly_acc[, c("Accuracy","Sensitivity","Specificity")]),
  cbind(Method = "SVM_rbf",    metrics_svm_rbf_acc[, c("Accuracy","Sensitivity","Specificity")]),
  cbind(Method = "Logistic",   metrics_reg_log[, c("Accuracy","Sensitivity","Specificity")])
)
rownames(results_summary_acc) = NULL
results_summary_acc

#Tabelka średnia czułość i specyficzność
results_summary_cs = bind_rows(
  cbind(Method = "SVM_linear", metrics_svm_linear_cs[, c("Accuracy","Sensitivity","Specificity")]),
  cbind(Method = "SVM_poly",   metrics_svm_poly_cs[, c("Accuracy","Sensitivity","Specificity")]),
  cbind(Method = "SVM_rbf",    metrics_svm_rbf_cs[, c("Accuracy","Sensitivity","Specificity")]),
  cbind(Method = "Logistic",   metrics_reg_log[, c("Accuracy","Sensitivity","Specificity")])
)
rownames(results_summary_cs) = NULL
results_summary_cs
