library(stats)
library(e1071)
library(dplyr)
library(mice)
library(caret)
library(pROC)
library(tidyr)
library(ggplot2)

#============================
#Wczytanie i obróbka danych
#============================
dane_stroke = read.csv2("stroke.csv", dec = ".")
head(dane_stroke)

md.pattern(dane_stroke)
dane_stroke = na.omit(dane_stroke)
md.pattern(dane_stroke, rotate.names = TRUE)

dane_stroke$gender = ifelse(dane_stroke$gender=="Male", 1, 0) # Male to 1
dane_stroke$ever_married = ifelse(dane_stroke$ever_married=="Yes", 1, 0)
dane_stroke$Residence_type = ifelse(dane_stroke$Residence_type=="Urban", 1, 0) #Urban to 1

dane_stroke$work_type = as.factor(dane_stroke$work_type)
dane_stroke$smoking_status = as.factor(dane_stroke$smoking_status)
dane_stroke$gender = as.factor(dane_stroke$gender)
dane_stroke$hypertension = as.factor(dane_stroke$hypertension)
dane_stroke$heart_disease = as.factor(dane_stroke$heart_disease)
dane_stroke$ever_married = as.factor(dane_stroke$ever_married)
dane_stroke$Residence_type = as.factor(dane_stroke$Residence_type)
dane_stroke$stroke = as.factor(dane_stroke$stroke)

summary(dane_stroke)

#balansowanie zbioru
stroke_0 = dane_stroke[dane_stroke$stroke == 0, ]
stroke_1 = dane_stroke[dane_stroke$stroke == 1, ]

set.seed(123)
stroke_0_sample = stroke_0[sample(1:nrow(stroke_0), nrow(stroke_1)), ]

#cały zbiór zbalansowany
dane_stroke_final = rbind(stroke_0_sample, stroke_1)
summary(dane_stroke_final)

boxplot(dane_stroke_final[c("age", "avg_glucose_level", "bmi")], xlab="Cały zbiór zbalansowany")

#zbiór z usuniętymi outlierami

num_cols <- c("age", "avg_glucose_level", "bmi")

remove_outliers_iqr <- function(data, column) {
  Q1 <- quantile(data[[column]], 0.25, na.rm = TRUE)
  Q3 <- quantile(data[[column]], 0.75, na.rm = TRUE)
  IQR_value <- Q3 - Q1
  
  lower_bound <- Q1 - 1.5 * IQR_value
  upper_bound <- Q3 + 1.5 * IQR_value
  
  before <- nrow(data)
  data_clean <- data %>%
    filter(.data[[column]] >= lower_bound & .data[[column]] <= upper_bound)
  after <- nrow(data_clean)
  
  cat(column, ": usunięto", before - after, "obserwacji\n")
  
  return(data_clean)
}

dane_stroke_final2 <- dane_stroke_final
for (col in num_cols) {
  dane_stroke_final2 <- remove_outliers_iqr(dane_stroke_final2, col)
}

boxplot(dane_stroke_final2[c("age", "avg_glucose_level", "bmi")], xlab="Bez wartości odstających")

#Podział na zbiór uczący i testowy
set.seed(123)
index = sample(length(dane_stroke_final$gender), 85) #około 20% wszystkich obserwacji (418)
testowy = dane_stroke_final[index,]
uczacy = dane_stroke_final[-index,]

summary(uczacy) #Brak wartości dla work_type Never_worked
summary(testowy) #Brak wartości dla work_type Never_worked


#Podział na zbiór uczący i testowy bez wartości odstających
set.seed(123)
index2 = sample(length(dane_stroke_final2$gender), 77) #około 20% wszystkich obserwacji (383)
testowy2 = dane_stroke_final2[index,]
uczacy2 = dane_stroke_final2[-index,]


#====================
# ZADANIE 1
#====================

#Regresja logistyczna
#model na całym zbalansowanym zbiorze
glm_model = glm(stroke ~ ., data = uczacy, family = "binomial")
summary(glm_model)

#Predykcja na zbiorze uczącym
pred_uczacy = predict(glm_model, newdata = uczacy, type = "response")
prog_odciecia = 0.5
klasy_uczacy = ifelse(pred_uczacy >= prog_odciecia, 1, 0)
confusionMatrix(factor(klasy_uczacy), factor(uczacy$stroke), positive = "1")

roc_obj_u <- roc(uczacy$stroke, pred_uczacy)
auc(roc_obj_u)
plot(roc_obj_u, main = "Krzywa ROC - model regresji logistycznej")

#Predykcja na zbiorze testowym
pred_testowy <- predict(glm_model, newdata = testowy, type = "response")
klasy_testowy <- ifelse(pred_testowy >= prog_odciecia, 1, 0)
confusionMatrix(factor(klasy_testowy), factor(testowy$stroke), positive = "1")

roc_obj_t <- roc(testowy$stroke, pred_testowy)
auc(roc_obj_t)
plot(roc_obj_t, main = "Krzywa ROC - model regresji logistycznej")



#model bez wartości odstających
glm_model2 = glm(stroke ~ ., data = uczacy2, family = "binomial")
summary(glm_model2)

#Predykcja na zbiorze uczącym v2
pred_uczacy2 = predict(glm_model2, newdata = uczacy2, type = "response")
prog_odciecia2 = 0.5
klasy_uczacy2 = ifelse(pred_uczacy2 >= prog_odciecia2, 1, 0)
head(klasy_uczacy2, 10)
confusionMatrix(factor(klasy_uczacy2), factor(uczacy2$stroke), positive = "1")

roc_obj_u2 <- roc(uczacy2$stroke, pred_uczacy2)
auc(roc_obj_u2)
plot(roc_obj_u2, main = "Krzywa ROC - model regresji logistycznej")

#Predykcja na zbiorze testowym v2
pred_testowy2 <- predict(glm_model2, newdata = testowy2, type = "response")
klasy_testowy2 <- ifelse(pred_testowy2 >= prog_odciecia2, 1, 0)
confusionMatrix(factor(klasy_testowy2), factor(testowy2$stroke), positive = "1")

roc_obj_t2 <- roc(testowy2$stroke, pred_testowy2)
auc(roc_obj_t2)
plot(roc_obj_t2, main = "Krzywa ROC - model regresji logistycznej")



#==========================
# ZADANIE NR 2
#==========================

#BAYESA
#test normalności dla każdej ze zmiennych liczbowych
shapiro.test(dane_stroke_final[dane_stroke_final$stroke == "1", "age"])
shapiro.test(dane_stroke_final[dane_stroke_final$stroke == "0", "age"])

shapiro.test(dane_stroke_final[dane_stroke_final$stroke == "1", "avg_glucose_level"])
shapiro.test(dane_stroke_final[dane_stroke_final$stroke == "0", "avg_glucose_level"])

shapiro.test(dane_stroke_final[dane_stroke_final$stroke == "1", "bmi"])
shapiro.test(dane_stroke_final[dane_stroke_final$stroke == "0", "bmi"])



#model na podstawie naiwnego klasyfikatora Bayesa
nb_model = naiveBayes(stroke ~., data = dane_stroke_final)
nb_model

#Predykcja na zbiorze uczącym
pred_nb_uczacy <- predict(nb_model, newdata = uczacy, type="raw")
klasy_uczacy_nb <- ifelse(pred_nb_uczacy[, "1"] >= prog_odciecia, 1, 0)
confusionMatrix(factor(klasy_uczacy_nb, levels = c(0,1)), factor(uczacy$stroke, levels = c(0,1)), positive = "1")

#Predykcja na zbiorze testowym
pred_nb_test <- predict(nb_model, newdata = testowy, type="raw")
klasy_testowy_nb <- ifelse(pred_nb_test[,"1"] >= prog_odciecia, 1, 0)
confusionMatrix(factor(klasy_testowy_nb, levels = c(0,1)), factor(testowy$stroke, levels = c(0,1)), positive = "1")



#model na podstawie naiwnego klasyfikatora Bayesa bez outlierów
nb_model2 = naiveBayes(stroke ~., data = dane_stroke_final2)
nb_model2

#Predykcja na zbiorze uczącym v2
pred_nb_uczacy2 <- predict(nb_model2, newdata = uczacy2, type="raw")
klasy_uczacy_nb2 <- ifelse(pred_nb_uczacy2[, "1"] >= prog_odciecia2, 1, 0)
confusionMatrix(factor(klasy_uczacy_nb2, levels = c(0,1)), factor(uczacy2$stroke, levels = c(0,1)), positive = "1")

#Predykcja na zbiorze testowym v2
pred_nb_test2 <- predict(nb_model2, newdata = testowy2, type="raw")
klasy_testowy_nb2 <- ifelse(pred_nb_test2[,"1"] >= prog_odciecia2, 1, 0)
confusionMatrix(factor(klasy_testowy_nb2, levels = c(0,1)), factor(testowy2$stroke, levels = c(0,1)), positive = "1")



#Porównanie wyników z regresją logistyczną
cm_logit <- confusionMatrix(factor(klasy_testowy, levels = c(0,1)), factor(testowy$stroke, levels = c(0,1)), positive = "1")
roc_logit <- roc(testowy$stroke, pred_testowy)

cm_logit2 <- confusionMatrix(factor(klasy_testowy2, levels = c(0,1)), factor(testowy2$stroke, levels = c(0,1)), positive = "1")
roc_logit2 <- roc(testowy2$stroke, pred_testowy2)

cm_nb <- confusionMatrix(factor(klasy_testowy_nb, levels = c(0,1)), factor(testowy$stroke, levels = c(0,1)), positive = "1")
roc_nb <- roc(testowy$stroke, pred_nb_test[, "1"])

cm_nb2 <- confusionMatrix(factor(klasy_testowy_nb2, levels = c(0,1)), factor(testowy2$stroke, levels = c(0,1)), positive = "1")
roc_nb2 <- roc(testowy2$stroke, pred_nb_test2[, "1"])

#tabelka z porównaniem 4 modeli
df_porownanie2 = data.frame(
  Model = c("Logit", "Logit w/o outliers", "Naive Bayes", "Naive Bayes w/o outliers"),
  Accuracy = c(cm_logit$overall["Accuracy"], cm_logit2$overall["Accuracy"], cm_nb$overall["Accuracy"], cm_nb2$overall["Accuracy"]),
  Sensitivity = c(cm_logit$byClass["Sensitivity"], cm_logit2$byClass["Sensitivity"], cm_nb$byClass["Sensitivity"], cm_nb2$byClass["Sensitivity"]),
  Specificity = c(cm_logit$byClass["Specificity"], cm_logit2$byClass["Specificity"], cm_nb$byClass["Specificity"], cm_nb2$byClass["Specificity"]),
  Precision = c(cm_logit$byClass["Precision"], cm_logit2$byClass["Precision"], cm_nb$byClass["Precision"], cm_nb2$byClass["Precision"]),
  F1 = c(cm_logit$byClass["F1"], cm_logit2$byClass["F1"], cm_nb$byClass["F1"], cm_nb2$byClass["F1"]),
  AUC = c(auc(roc_logit), auc(roc_logit2), auc(roc_nb), auc(roc_nb2))
)


#===========================
# ZADANIE 3
#===========================
#Jak zmiana progu odcięcia wpływa na wyniki
evaluate_cutoff <- function(prog, prob, y_true) {
  pred <- factor(ifelse(prob >= prog, 1, 0), levels = c(0,1))
  y_true <- factor(y_true, levels = c(0,1))
  cm <- confusionMatrix(pred, y_true, positive = "1")
  return(data.frame(
    cutoff = prog,
    Accuracy = cm$overall["Accuracy"],
    Sensitivity = cm$byClass["Sensitivity"],
    Specificity = cm$byClass["Specificity"]
  ))
}

#progi odcięcia
cutoffs <- seq(0.1, 0.9, by = 0.05)

results_logit <- do.call(rbind, lapply(cutoffs, function(c) {
  evaluate_cutoff(c, pred_testowy, testowy$stroke)
}))

results_logit2 <- do.call(rbind, lapply(cutoffs, function(c) {
  evaluate_cutoff(c, pred_testowy2, testowy2$stroke)
}))

results_nb <- do.call(rbind, lapply(cutoffs, function(c) {
  evaluate_cutoff(c, pred_nb_test[, "1"], testowy$stroke)
}))

results_nb2 <- do.call(rbind, lapply(cutoffs, function(c) {
  evaluate_cutoff(c, pred_nb_test2[, "1"], testowy2$stroke)
}))


results_logit$model <- "Logit"
results_logit2$model <- "Logit w/o outliers"
results_nb$model <- "Naive Bayes"
results_nb2$model <- "Naive Bayes w/o outliers"

# Połączenie wszystkich wyników
results_all <- bind_rows(results_logit, results_logit2, results_nb, results_nb2)

# Przekształcenie do formatu long dla ggplot
results_long <- results_all %>%
  pivot_longer(cols = c("Accuracy", "Sensitivity", "Specificity"),
               names_to = "Metric",
               values_to = "Value")

ggplot(results_long, aes(x = cutoff, y = Value, color = model)) +
  geom_line(size = 1) +
  geom_point() +
  facet_wrap(~ Metric, scales = "free_y") +
  labs(
    title = "Zmiana miar klasyfikacji w zależności od progu odcięcia",
    x = "Próg odcięcia",
    y = "Wartość miary",
    color = "Model"
  ) +
  theme_minimal() +
  theme(
    strip.text = element_text(size = 12, face = "bold"),
    legend.position = "bottom"
  )



#Wykresy krzywej ROC dla przedstawionych 4 modeli
plot(roc_logit, col="blue", main="ROC: Logit vs Logit w/o outliers vs Naive Bayes")
lines(roc_logit2, col="green")
lines(roc_nb, col="red")
lines(roc_nb2, col = "orange")
legend("bottomright", legend=c("Logit", "Logit w/o outliers", "Naive Bayes", "Naive Bayes w/o outliers"), col=c("blue", "green", "red", "orange"), lwd=2)



