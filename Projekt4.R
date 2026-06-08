library(rpart)
library(rattle)
library(caret)
library(vip)
library(VIM)
library(randomForest)
library(dplyr)
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


#Model drzewa decyzyjnego
set.seed(123)
index = sample(nrow(dane_stroke_final), size = 0.2*nrow(dane_stroke_final), replace = F)
testowy = dane_stroke_final[index,]
uczacy = dane_stroke_final[-index,]
model = rpart(stroke ~.,
              data = uczacy,
              method = "class")
model

#Wykres
fancyRpartPlot(model, sub = NULL)


#predykcja na zbiorze uczącym i testowym
y_pred_uczacy = predict(model, newdata = uczacy, type = "class")
y_pred_test = predict(model, newdata = testowy, type = "class")
#uczący
confusionMatrix(y_pred_uczacy, uczacy$stroke, positive = "1")
#testowy
confusionMatrix(y_pred_test, testowy$stroke, positive = "1")

#Model ze zmienioną maksymalną głębokością
model2 = rpart(stroke ~.,
               data = uczacy,
               method = "class",
               control = rpart.control(maxdepth = 4))
fancyRpartPlot(model2, sub = NULL)

#Ważność zmiennych
vip(model2)

#pakiet caret i drzewa decyzyjne
model3 = caret::train(survived ~.,
                      data = uczacy,
                      method = "rpart",
                      tuneGrid = expand.grid(cp = seq(0.02, 0.2, 0.02)))


#===================
#Przy sprawdzaniu jakie parametry są lepsze dobrze jest zrobić w pętli (kombinacje kilku naraz mogą pokazać jakie faktycznie najlepsze)

#domyślnie indeks Giniego


#===============
#Bagging

# Parametry eksperymentu
p = ncol(uczacy) - 1  # liczba predyktorów (zakładamy, że stroke jest jedną z kolumn)
predictors = setdiff(names(uczacy), "stroke")

ntree_list = c(10, 50, 100, 200, 500)
# sampsize jako ułamek lub liczba — użyjemy proporcji względem nrow(uczacy)
sampsize_prop = c(0.5, 0.7, 0.9, 1.0)

# Metryki dla danej predykcji
calc_metrics = function(true, pred, positive_level = "1") {
  cm = caret::confusionMatrix(pred, true, positive = positive_level)
  out = c(Accuracy = as.numeric(cm$overall["Accuracy"]),
          Sensitivity = as.numeric(cm$byClass["Sensitivity"]),
          Specificity = as.numeric(cm$byClass["Specificity"]))
  return(out)
}

set.seed(123)
results = data.frame()
for(nt in ntree_list) {
  for(sp in sampsize_prop) {
    cat("Trenuję: ntree =", nt, "sampsize_prop =", sp, "\n")
    sampsize_n = floor(nrow(uczacy) * sp)
    if(sampsize_n < 1) next
    
    rf = randomForest(
      formula = stroke ~ .,
      data = uczacy,
      ntree = nt,
      mtry = p,
      sampsize = sampsize_n,
      replace = TRUE
    )
    
    # Predykcje - na zbiorze uczącym i testowym
    pred_train = predict(rf, newdata = uczacy, type = "response")
    pred_test = predict(rf, newdata = testowy, type = "response")
    
    # Metryki
    met_train = calc_metrics(uczacy$stroke, pred_train, positive_level = "1")
    met_test  = calc_metrics(testowy$stroke, pred_test, positive_level = "1")
    
    # Zapisywanie wyniku
    results = rbind(results, data.frame(
      ntree = nt,
      sampsize_prop = sp,
      sampsize_n = sampsize_n,
      Accuracy_train = met_train["Accuracy"],
      Sensitivity_train = met_train["Sensitivity"],
      Specificity_train = met_train["Specificity"],
      Accuracy_test  = met_test["Accuracy"],
      Sensitivity_test = met_test["Sensitivity"],
      Specificity_test = met_test["Specificity"]
    ))
  }
}

results = results %>% arrange(ntree, sampsize_prop)
print(results)

#Jak liczba drzew wpływa na metryki (dla każdej sampsize_prop)
results_long = results %>%
  pivot_longer(cols = starts_with(c("Accuracy", "Sensitivity", "Specificity")),
               names_to = c("metric", "set"),
               names_sep = "_",
               values_to = "value")

# Rysujemy osobno dla train/test (facet) i oddzielnie dla metric
ggplot(results_long, aes(x = ntree, y = value, color = as.factor(sampsize_prop))) +
  geom_line() + geom_point() +
  facet_grid(metric ~ set, scales = "free_y") +
  labs(title = "Wpływ liczby drzew (ntree) i rozmiaru próbki (sampsize_prop)",
       x = "Liczba drzew (ntree)",
       color = "sampsize_prop") +
  theme_minimal()

# --- Wybór najlepszego modelu
# Kryterium: najwyższa Sensitivity_test (priorytet wykrywania '1'), potem Accuracy_test
best_idx = which.max(results$Sensitivity_test + 1e-6 * results$Accuracy_test)
best_row = results[best_idx, ]
cat("Najlepsze parametry wg (Sensitivity_test, potem Accuracy_test):\n")
print(best_row)

