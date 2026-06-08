install.packages("class")
install.packages("kknn")
install.packages("mice")
install.packages("caret")
install.packages("future.apply")
library(future.apply)
install.packages("recipes", dependencies = TRUE)
library(recipes)
library(caret)
library(class)
library(kknn)
library(mice)


dane_stroke = read.csv2("stroke.csv", dec = ".")
head(dane_stroke)

md.pattern(dane_stroke)
dane_stroke = na.omit(dane_stroke)
md.pattern(dane_stroke)

dane_stroke$gender = ifelse(dane_stroke$gender=="Male", 1, 0) # Male to 1
dane_stroke$ever_married = ifelse(dane_stroke$ever_married=="Yes", 1, 0)
dane_stroke$Residence_type = ifelse(dane_stroke$Residence_type=="Urban", 1, 0) #Urban to 1

dane_stroke$work_type = model.matrix(~ dane_stroke$work_type - 1)
dane_stroke$smoking_status = model.matrix(~ dane_stroke$smoking_status - 1)
dane_stroke$stroke = as.factor(dane_stroke$stroke)

summary(dane_stroke)

#balansowanie zbioru
stroke_0 = dane_stroke[dane_stroke$stroke == 0, ]
stroke_1 = dane_stroke[dane_stroke$stroke == 1, ]

set.seed(416669)
stroke_0_sample = stroke_0[sample(1:nrow(stroke_0), nrow(stroke_1)), ]

dane_stroke_final = rbind(stroke_0_sample, stroke_1)
summary(dane_stroke_final)


# zad 2
set.seed(416669)
index = sample(length(dane_stroke_final$gender), 85)
testowy = dane_stroke_final[index,]
uczacy = dane_stroke_final[-index,]

#standaryzacja
stand = c(2, 8, 9)
uczacy[,stand] = scale(uczacy[,stand])
testowy[,stand] = scale(testowy[,stand])



#================================
k_v = 11
#Metoda KNN
# Predykcja na zbiorze uczącym
knn.uczacy = knn(train = uczacy[, -11],
                 test = uczacy[, -11],
                 cl = uczacy[, 11],
                 k = k_v)

#macierz błędów
table(knn.uczacy, uczacy$stroke)

# Predykcja na zbiorze testowym
knn.testowy = knn(train = uczacy[, -11],
                  test = testowy[, -11],
                  cl = uczacy[, 11],
                  k = k_v)

caret::confusionMatrix(knn.testowy, testowy$stroke, positive = "1")

#Które k wybrać
k_values <- seq(1, 31, 2)
accuracy <- numeric(length(k_values))
for(i in seq_along(k_values)){
  preds <- knn(cl = uczacy[, 11],
               train = uczacy[, -11],
               test = testowy[, -11],
               k = k_values[i])
  
  accuracy[i] <- mean(preds == testowy$stroke)
}
plot(k_values, accuracy, type = "b", 
     xlab = "Liczba sąsiadów k", 
     ylab = "Dokładność",
     main = "Wpływ k na dokładność kNN",
     col = "blue", pch = 19)

#Cross-validation
set.seed(416669)
train_control = trainControl(method = "cv", number = 10)
k_grid = expand.grid(k = k_values)

knn_cv = train(
  x = dane_stroke[,-11],
  y = dane_stroke[,11],
  method = "knn",
  trControl = train_control,
  tuneGrid = k_grid
)

plot(knn_cv)


#=========================
#Metoda KKNN
k_vk = 9
# Predykcja na zbiorze uczącym
kknn.uczacy = kknn(formula = uczacy[, 11]~.,
                   train = uczacy[, -11],
                   test = uczacy[, -11],
                   k = k_vk,
                   distance = 1,
                   kernel = "rectangular")
kknn.uczacy.wyniki = fitted(kknn.uczacy)

#macierz błędów
table(kknn.uczacy.wyniki, uczacy$stroke)

# Prognoza na zbiorze testowym
kknn.testowy = kknn(formula = uczacy[, 11]~.,
                    train = uczacy[, -11],
                    test = testowy[, -11],
                    k = k_v,
                    distance = 1,
                    kernel = "rectangular")
kknn.testowy.wyniki = fitted(kknn.testowy)

#Macierz błędów z miarami jakości klasyfikacji
caret::confusionMatrix(kknn.testowy.wyniki, testowy$stroke, positive = "1")


#Które k wybrać dla kknn
for(i in seq_along(k_values)){
  model <- kknn(formula = uczacy[, 11] ~ .,
                train = uczacy[, -11],
                test = testowy[, -11],
                k = k_values[i],
                distance = 1)
  
  preds <- fitted(model)
  accuracy[i] <- mean(preds == testowy$stroke)
}
plot(k_values, accuracy, type = "b", 
     xlab = "Liczba sąsiadów k", 
     ylab = "Dokładność",
     main = "Wpływ k na dokładność kNN",
     col = "blue", pch = 19)

k_values_kknn <- data.frame(kmax = k_values,
                            distance = 1,
                            kernel = "optimal")

kknn_cv <- train(
  stroke ~ ., 
  data = dane_stroke_final,
  method = "kknn",
  trControl = train_control,
  tuneGrid = k_values_kknn
)

plot(kknn_cv)
