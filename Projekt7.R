library(dplyr)
library(mice)
library(caret)
library(e1071)
library(ROSE)


dane_stroke = read.csv2("stroke.csv", dec = ".")

summary(dane_stroke)

md.pattern(dane_stroke, rotate.names = TRUE)
dane_stroke = na.omit(dane_stroke)
md.pattern(dane_stroke, rotate.names = TRUE)

dane_stroke$work_type = as.factor(dane_stroke$work_type)
dane_stroke$smoking_status = as.factor(dane_stroke$smoking_status)
dane_stroke$gender = as.factor(dane_stroke$gender)
dane_stroke$hypertension = as.factor(dane_stroke$hypertension)
dane_stroke$heart_disease = as.factor(dane_stroke$heart_disease)
dane_stroke$ever_married = as.factor(dane_stroke$ever_married)
dane_stroke$Residence_type = as.factor(dane_stroke$Residence_type)
dane_stroke$stroke = as.factor(dane_stroke$stroke)

summary(dane_stroke)

#Poziom niezbalansowania zbioru
prop.table(table(dane_stroke$stroke))


#funkcja licząca miary klasyfikacji
calc_metrics <- function(true, pred, model_name, set_name) {
  cm <- confusionMatrix(pred, true, positive = "1")
  
  data.frame(
    Model = model_name,
    Set = set_name,
    Accuracy = round(cm$overall["Accuracy"], 5),
    Sensitivity = round(cm$byClass["Sensitivity"], 5),
    Specificity = round(cm$byClass["Specificity"], 5)
  )
}


#Podział na zbiór uczący i testowy
set.seed(123)
index = sample(nrow(dane_stroke), size = 0.2*nrow(dane_stroke), replace = F)
testowy = dane_stroke[index,]
uczacy = dane_stroke[-index,]


results = data.frame()

#Model svm jądro radialne niezbalansowany
model_unb = svm(stroke ~ ., data = uczacy, kernel = "radial")

pred_uczacy_unb = predict(model_unb, newdata = uczacy)
pred_testowy_unb = predict(model_unb, newdata = testowy)

results <- rbind(results,
                 calc_metrics(uczacy$stroke, pred_uczacy_unb, "SVM_unbalanced", "Uczący"),
                 calc_metrics(testowy$stroke, pred_testowy_unb, "SVM_unbalanced", "Testowy")
)

#=============
#UNDERSAMPLING
#=============
#Model zbalansowany undersampling
#losowy undersampling
set.seed(123)
dane_b_under = ovun.sample(stroke ~., data = uczacy, method = "under")$data
nrow(dane_b_under)
prop.table(table(dane_b_under$stroke))

model_b_under = svm(stroke ~ ., data = dane_b_under, kernel = "radial")

pred_uczacy_under = predict(model_b_under, newdata = dane_b_under)
pred_testowy_under = predict(model_b_under, newdata = testowy)

results <- rbind(results,
                 calc_metrics(dane_b_under$stroke, pred_uczacy_under, "SVM_undersampling", "Uczący"),
                 calc_metrics(testowy$stroke, pred_testowy_under, "SVM_undersampling", "Testowy")
)

#==================================
#OVERSAMPLING
#==================================
#Model zbalansowany - oversampling
#losowy oversampling
set.seed(123)
dane_b_over = ovun.sample(stroke ~., data = uczacy, method = "over")$data
nrow(dane_b_over_los)
prop.table(table(dane_b_over$stroke))

model_b_over = svm(stroke ~ ., data = dane_b_over, kernel = "radial")

pred_uczacy_over = predict(model_b_over, newdata = dane_b_over)
pred_testowy_over = predict(model_b_over, newdata = testowy)

results <- rbind(results,
                 calc_metrics(dane_b_over$stroke, pred_uczacy_over, "SVM_oversampling", "Uczący"),
                 calc_metrics(testowy$stroke, pred_testowy_over, "SVM_oversampling", "Testowy")
)

rownames(results) = NULL
results


#==========
#Zadanie 2
#==========
set.seed(123)
prop_pos = c(0.1, 0.2, 0.3, 0.4)

results_sim = data.frame()

pos_all = dane_stroke %>% filter(stroke == "1")
neg_all = dane_stroke %>% filter(stroke == "0")

for (p in prop_pos) {
  N = nrow(dane_stroke)
  n_pos = round(N * p)
  n_neg = N - n_pos
  
  pos_sample = pos_all %>% sample_n(n_pos, replace = TRUE)
  neg_sample = neg_all %>% sample_n(n_neg, replace = TRUE)
  
  data_sim = rbind(pos_sample, neg_sample)
  
  index = sample(nrow(data_sim), size = 0.2 * nrow(data_sim))
  test_sim = data_sim[index, ]
  train_sim = data_sim[-index, ]
  
  model_unb = svm(stroke ~ ., data = train_sim, kernel = "radial")
  
  pred_train_unb = predict(model_unb, train_sim)
  pred_test_unb  = predict(model_unb, test_sim)
  
  tmp = rbind(
    calc_metrics(train_sim$stroke, pred_train_unb, "Unbalanced", "Uczący"),
    calc_metrics(test_sim$stroke, pred_test_unb, "Unbalanced", "Testowy")
  )
  tmp$prop = p
  results_sim = rbind(results_sim, tmp)
  
  set.seed(123)
  train_under = ovun.sample(stroke ~ ., data = train_sim, method = "under")$data
  
  model_under = svm(stroke ~ ., data = train_under, kernel = "radial")
  
  pred_train_under = predict(model_under, train_under)
  pred_test_under = predict(model_under, test_sim)
  
  tmp = rbind(
    calc_metrics(train_under$stroke, pred_train_under, "Undersampling", "Uczący"),
    calc_metrics(test_sim$stroke, pred_test_under, "Undersampling", "Testowy")
  )
  tmp$prop = p
  results_sim = rbind(results_sim, tmp)
  
  set.seed(123)
  train_over = ovun.sample(stroke ~ ., data = train_sim, method = "over")$data
  
  model_over = svm(stroke ~ ., data = train_over, kernel = "radial")
  
  pred_train_over = predict(model_over, train_over)
  pred_test_over = predict(model_over, test_sim)
  
  tmp = rbind(
    calc_metrics(train_over$stroke, pred_train_over, "Oversampling", "Uczący"),
    calc_metrics(test_sim$stroke, pred_test_over, "Oversampling", "Testowy")
  )
  tmp$prop = p
  results_sim = rbind(results_sim, tmp)
}

rownames(results_sim) = NULL
results_sim %>% select(prop, Model, Set, Accuracy, Sensitivity, Specificity)


#Na zbiorach uczących (do porównania z wynikami testowych)
results_sim %>%
  select(prop, Model, Set, Accuracy, Sensitivity, Specificity) %>%
  filter(Set=="Uczący")

#Klasyfikacja według czułości
results_sim %>%
  arrange(desc(Sensitivity)) %>% 
  select(prop, Model, Set, Accuracy, Sensitivity, Specificity) %>%
  filter(Set == "Testowy")
