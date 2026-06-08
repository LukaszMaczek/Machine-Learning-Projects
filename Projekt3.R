library(stats)
library(mice)
library(ggplot2)
library(dplyr)
library(VIM)
library(caret)
library(e1071)
library(kknn)
library(class)
library(tidyr)


#Wczytanie i obróbka danych
dane_stroke = read.csv2("zaj3_Stroke.csv", dec = ".")
dane_heart = read.csv2("zaj3_Heart.csv", dec = ".")

md.pattern(dane_stroke, rotate.names = T) # łącznie 154 braki danych w age, heart_disease, work_type, avg_glucose_level, bmi
md.pattern(dane_heart, rotate.names = T) # łącznie 198 braków w oldpeak, exerciseAngina,
# Restig ECG, Age, Cholesterol

sum(is.na(dane_stroke))
sum(is.na(dane_heart))

#dane stroke

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

str(dane_stroke)
summary(dane_stroke)

#statystyki opisowe
summary(dane_stroke[,c(2, 4, 6, 8, 9)])


#wykresy gęstości
#wiek
plot(density(dane_stroke$age, na.rm = T), 
     main = "Wykres gęstości",
     xlab = "Wiek")
grid()

#poziom glukozy
plot(density(dane_stroke$avg_glucose_level, na.rm = T), 
     main = "Wykres gęstości",
     xlab = "Średni poziom glukozy")
grid()

#bmi
plot(density(dane_stroke$bmi, na.rm = T), 
     main = "Wykres gęstości",
     xlab = "BMI")
grid()


#dane heart

dane_heart$Sex = as.factor(dane_heart$Sex)
dane_heart$ChestPainType = as.factor(dane_heart$ChestPainType)
dane_heart$FastingBS = as.factor(dane_heart$FastingBS)
dane_heart$RestingECG = as.factor(dane_heart$RestingECG)
dane_heart$ExerciseAngina = as.factor(dane_heart$ExerciseAngina)
dane_heart$Oldpeak = as.numeric(gsub(",", ".", dane_heart$Oldpeak))
dane_heart$ST_Slope = as.factor(dane_heart$ST_Slope)
dane_heart$HeartDisease = as.factor(dane_heart$HeartDisease)

str(dane_heart)
summary(dane_heart)

#statystyki opisowe
summary(dane_heart[,c(1,5,7,9,10)])


#wykresy gęstości
#wiek
plot(density(dane_heart$Age, na.rm = T), 
     main = "Wykres gęstości",
     xlab = "Wiek")
grid()

#Cholesterol
plot(density(dane_heart$Cholesterol, na.rm = T), 
     main = "Wykres gęstości",
     xlab = "Cholesterol")
grid()

#oldpeak
plot(density(dane_heart$Oldpeak, na.rm = T), 
     main = "Wykres gęstości",
     xlab = "Oldpeak")
grid()



#DZIAŁANIE Z BRAKAMI DANYCH
#usuwamy
dane_stroke_omit = na.omit(dane_stroke)
dane_heart_omit = na.omit(dane_heart)

#stroke
#statystyki opisowe
summary(dane_stroke_omit[,c(2, 4, 6, 8, 9)])


#wykresy gęstości
#wiek
plot(density(dane_stroke_omit$age, na.rm = T), 
     main = "Wykres gęstości",
     xlab = "Wiek")
grid()

#poziom glukozy
plot(density(dane_stroke_omit$avg_glucose_level, na.rm = T), 
     main = "Wykres gęstości",
     xlab = "Średni poziom glukozy")
grid()

#bmi
plot(density(dane_stroke_omit$bmi, na.rm = T), 
     main = "Wykres gęstości",
     xlab = "BMI")
grid()

#Heart
#statystyki opisowe
summary(dane_heart_omit[,c(1,5,7,9,10)])


#wykresy gęstości
#wiek
plot(density(dane_heart_omit$Age, na.rm = T), 
     main = "Wykres gęstości",
     xlab = "Wiek")
grid()

#Cholesterol
plot(density(dane_heart_omit$Cholesterol, na.rm = T), 
     main = "Wykres gęstości",
     xlab = "Cholesterol")
grid()

#oldpeak
plot(density(dane_heart_omit$Oldpeak, na.rm = T), 
     main = "Wykres gęstości",
     xlab = "Oldpeak")
grid()



#(do kategorycznych zawsze moda)
#wstawiamy średnią
moda <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

dane_stroke_mean = dane_stroke
dane_stroke_mean$age[is.na(dane_stroke_mean$age)] = mean(dane_stroke_mean$age, na.rm = T)
dane_stroke_mean$heart_disease[is.na(dane_stroke_mean$heart_disease)] = moda(dane_stroke_mean$heart_disease)
dane_stroke_mean$work_type[is.na(dane_stroke_mean$work_type)] = moda(dane_stroke_mean$work_type)
dane_stroke_mean$avg_glucose_level[is.na(dane_stroke_mean$avg_glucose_level)] = mean(dane_stroke_mean$avg_glucose_level, na.rm=T)
dane_stroke_mean$bmi[is.na(dane_stroke_mean$bmi)] = mean(dane_stroke_mean$bmi, na.rm=T)


dane_heart_mean = dane_heart
dane_heart_mean$Age[is.na(dane_heart_mean$Age)] = mean(dane_heart_mean$Age, na.rm = T)
dane_heart_mean$Cholesterol[is.na(dane_heart_mean$Cholesterol)] = mean(dane_heart_mean$Cholesterol, na.rm = T)
dane_heart_mean$Oldpeak[is.na(dane_heart_mean$Oldpeak)] = mean(dane_heart_mean$Oldpeak, na.rm = T)
dane_heart_mean$RestingECG[is.na(dane_heart_mean$RestingECG)] = moda(dane_heart_mean$RestingECG)
dane_heart_mean$ExerciseAngina[is.na(dane_heart_mean$ExerciseAngina)] = moda(dane_heart_mean$ExerciseAngina)


#Stroke
summary(dane_stroke_mean[,c(2, 4, 6, 8, 9)])


#wykresy gęstości
#wiek
plot(density(dane_stroke_mean$age, na.rm = T), 
     main = "Wykres gęstości",
     xlab = "Wiek")
grid()

#poziom glukozy
plot(density(dane_stroke_mean$avg_glucose_level, na.rm = T), 
     main = "Wykres gęstości",
     xlab = "Średni poziom glukozy")
grid()

#bmi
plot(density(dane_stroke_mean$bmi, na.rm = T), 
     main = "Wykres gęstości",
     xlab = "BMI")
grid()

#Heart
#statystyki opisowe
summary(dane_heart_mean[,c(1,5,7,9,10)])


#wykresy gęstości
#wiek
plot(density(dane_heart_mean$Age, na.rm = T), 
     main = "Wykres gęstości",
     xlab = "Wiek")
grid()

#Cholesterol
plot(density(dane_heart_mean$Cholesterol, na.rm = T), 
     main = "Wykres gęstości",
     xlab = "Cholesterol")
grid()

#oldpeak
plot(density(dane_heart_mean$Oldpeak, na.rm = T), 
     main = "Wykres gęstości",
     xlab = "Oldpeak")
grid()

#==================
#wstawiamy medianę
dane_stroke_med = dane_stroke
dane_stroke_med$age[is.na(dane_stroke_med$age)] = median(dane_stroke_med$age, na.rm = T)
dane_stroke_med$heart_disease[is.na(dane_stroke_med$heart_disease)] = moda(dane_stroke_med$heart_disease)
dane_stroke_med$work_type[is.na(dane_stroke_med$work_type)] = moda(dane_stroke_med$work_type)
dane_stroke_med$avg_glucose_level[is.na(dane_stroke_med$avg_glucose_level)] = median(dane_stroke_med$avg_glucose_level, na.rm=T)
dane_stroke_med$bmi[is.na(dane_stroke_med$bmi)] = median(dane_stroke_med$bmi, na.rm=T)


dane_heart_med = dane_heart
dane_heart_med$Age[is.na(dane_heart_med$Age)] = median(dane_heart_med$Age, na.rm = T)
dane_heart_med$Cholesterol[is.na(dane_heart_med$Cholesterol)] = median(dane_heart_med$Cholesterol, na.rm = T)
dane_heart_med$Oldpeak[is.na(dane_heart_med$Oldpeak)] = median(dane_heart_med$Oldpeak, na.rm = T)
dane_heart_med$RestingECG[is.na(dane_heart_med$RestingECG)] = moda(dane_heart_med$RestingECG)
dane_heart_med$ExerciseAngina[is.na(dane_heart_med$ExerciseAngina)] = moda(dane_heart_med$ExerciseAngina)


#Stroke
summary(dane_stroke_med[,c(2, 4, 6, 8, 9)])


#wykresy gęstości
#wiek
plot(density(dane_stroke_med$age, na.rm = T), 
     main = "Wykres gęstości",
     xlab = "Wiek")
grid()

#poziom glukozy
plot(density(dane_stroke_med$avg_glucose_level, na.rm = T), 
     main = "Wykres gęstości",
     xlab = "Średni poziom glukozy")
grid()

#bmi
plot(density(dane_stroke_med$bmi, na.rm = T), 
     main = "Wykres gęstości",
     xlab = "BMI")
grid()

#Heart
#statystyki opisowe
summary(dane_heart_med[,c(1,5,7,9,10)])


#wykresy gęstości
#wiek
plot(density(dane_heart_med$Age, na.rm = T), 
     main = "Wykres gęstości",
     xlab = "Wiek")
grid()

#Cholesterol
plot(density(dane_heart_med$Cholesterol, na.rm = T), 
     main = "Wykres gęstości",
     xlab = "Cholesterol")
grid()

#oldpeak
plot(density(dane_heart_med$Oldpeak, na.rm = T), 
     main = "Wykres gęstości",
     xlab = "Oldpeak")
grid()


#==============
#na podstawie knn
dane_stroke_knn = kNN(dane_stroke, k = 5)

dane_heart_knn = kNN(dane_heart, k = 5)


#Stroke
summary(dane_stroke_knn[,c(2, 4, 6, 8, 9)])


#wykresy gęstości
#wiek
plot(density(dane_stroke_knn$age, na.rm = T), 
     main = "Wykres gęstości",
     xlab = "Wiek")
grid()

#poziom glukozy
plot(density(dane_stroke_knn$avg_glucose_level, na.rm = T), 
     main = "Wykres gęstości",
     xlab = "Średni poziom glukozy")
grid()

#bmi
plot(density(dane_stroke_knn$bmi, na.rm = T), 
     main = "Wykres gęstości",
     xlab = "BMI")
grid()

#Heart
#statystyki opisowe
summary(dane_heart_knn[,c(1,5,7,9,10)])


#wykresy gęstości
#wiek
plot(density(dane_heart_knn$Age, na.rm = T), 
     main = "Wykres gęstości",
     xlab = "Wiek")
grid()

#Cholesterol
plot(density(dane_heart_knn$Cholesterol, na.rm = T), 
     main = "Wykres gęstości",
     xlab = "Cholesterol")
grid()

#oldpeak
plot(density(dane_heart_knn$Oldpeak, na.rm = T), 
     main = "Wykres gęstości",
     xlab = "Oldpeak")
grid()

par(mfrow = c(2, 3))  # 1 wiersz, 2 kolumny
plot(density(dane_heart$Age, na.rm = T), 
     main = "Z NA",
     xlab = "Wiek")
grid()
plot(density(dane_heart_omit$Age, na.rm = T), 
     main = "Usunięte NA",
     xlab = "Wiek")
grid()
plot(density(dane_heart_mean$Age, na.rm = T), 
     main = "Średnia",
     xlab = "Wiek")
grid()
plot(density(dane_heart_med$Age, na.rm = T), 
     main = "Mediana",
     xlab = "Wiek")
grid()
plot(density(dane_heart_knn$Age, na.rm = T), 
     main = "Knn",
     xlab = "Wiek")
grid()


#STROKE
#BMI
par(mfrow = c(2, 3))

plot(density(dane_stroke$bmi, na.rm = TRUE), 
     main = "Oryginalne", xlab = "BMI", col = "blue")

plot(density(dane_stroke_omit$bmi, na.rm = TRUE), 
     main = "Usunięte braki", xlab = "BMI", col = "red")

plot(density(dane_stroke_mean$bmi, na.rm = TRUE), 
     main = "Średnia", xlab = "BMI", col = "green")

plot(density(dane_stroke_med$bmi, na.rm = TRUE), 
     main = "Mediana", xlab = "BMI", col = "orange")

plot(density(dane_stroke_knn$bmi, na.rm = TRUE), 
     main = "KNN", xlab = "BMI", col = "purple")

plot(density(dane_stroke$bmi, na.rm = TRUE), 
     main = "Porównanie", xlab = "BMI", col = "blue", lwd=1.5, ylim= c(0,0.08))
lines(density(dane_stroke_omit$bmi, na.rm = TRUE), col = "red", lwd=1.5)
lines(density(dane_stroke_mean$bmi, na.rm = TRUE), col = "green", lwd=1.5)
lines(density(dane_stroke_med$bmi, na.rm = TRUE), col = "orange", lwd=1.5)
lines(density(dane_stroke_knn$bmi, na.rm = TRUE), col = "purple", lwd=1.5)

par(mfrow = c(1,1))  # przywróć układ domyślny


#Średni poziom glukozy
par(mfrow = c(2, 3))

plot(density(dane_stroke$avg_glucose_level, na.rm = TRUE), 
     main = "Oryginalne", xlab = "Średni poziom glukozy", col = "blue")

plot(density(dane_stroke_omit$avg_glucose_level, na.rm = TRUE), 
     main = "Usunięte braki", xlab = "Średni poziom glukozy", col = "red")

plot(density(dane_stroke_mean$avg_glucose_level, na.rm = TRUE), 
     main = "Średnia", xlab = "Średni poziom glukozy", col = "green")

plot(density(dane_stroke_med$avg_glucose_level, na.rm = TRUE), 
     main = "Mediana", xlab = "Średni poziom glukozy", col = "orange")

plot(density(dane_stroke_knn$avg_glucose_level, na.rm = TRUE), 
     main = "KNN", xlab = "Średni poziom glukozy", col = "purple")

plot(density(dane_stroke$avg_glucose_level, na.rm = TRUE), 
     main = "Porównanie", xlab = "Średni poziom glukozy", col = "blue", lwd = 1.5, ylim = c(0, 0.016))
lines(density(dane_stroke_omit$avg_glucose_level, na.rm = TRUE), col = "red", lwd = 1.5)
lines(density(dane_stroke_mean$avg_glucose_level, na.rm = TRUE), col = "green", lwd = 1.5)
lines(density(dane_stroke_med$avg_glucose_level, na.rm = TRUE), col = "orange", lwd = 1.5)
lines(density(dane_stroke_knn$avg_glucose_level, na.rm = TRUE),col = "purple", lwd = 1.5)

par(mfrow = c(1, 1))


#wiek
par(mfrow = c(2, 3))

plot(density(dane_stroke$age, na.rm = TRUE), 
     main = "Oryginalne", xlab = "Wiek", col = "blue")

plot(density(dane_stroke_omit$age, na.rm = TRUE), 
     main = "Usunięte braki", xlab = "Wiek", col = "red")

plot(density(dane_stroke_mean$age, na.rm = TRUE), 
     main = "Średnia", xlab = "Wiek", col = "green")

plot(density(dane_stroke_med$age, na.rm = TRUE), 
     main = "Mediana", xlab = "Wiek", col = "orange")

plot(density(dane_stroke_knn$age, na.rm = TRUE), 
     main = "KNN", xlab = "Wiek", col = "purple")

plot(density(dane_stroke$age, na.rm = TRUE), 
     main = "Porównanie", xlab = "Wiek", col = "blue", lwd = 1.5)
lines(density(dane_stroke_omit$age, na.rm = TRUE), col = "red", lwd = 1.5)
lines(density(dane_stroke_mean$age, na.rm = TRUE), col = "green", lwd = 1.5)
lines(density(dane_stroke_med$age, na.rm = TRUE), col = "orange", lwd = 1.5)
lines(density(dane_stroke_knn$age, na.rm = TRUE), col = "purple", lwd = 1.5)

par(mfrow = c(1, 1))


#HEART
#cholesterol
par(mfrow = c(2, 3))

plot(density(dane_heart$Cholesterol, na.rm = TRUE), 
     main = "Oryginalne", xlab = "Cholesterol", col = "blue")

plot(density(dane_heart_omit$Cholesterol, na.rm = TRUE), 
     main = "Usunięte braki", xlab = "Cholesterol", col = "red")

plot(density(dane_heart_mean$Cholesterol, na.rm = TRUE), 
     main = "Średnia", xlab = "Cholesterol", col = "green")

plot(density(dane_heart_med$Cholesterol, na.rm = TRUE), 
     main = "Mediana", xlab = "Cholesterol", col = "orange")

plot(density(dane_heart_knn$Cholesterol, na.rm = TRUE), 
     main = "KNN", xlab = "Cholesterol", col = "purple")

plot(density(dane_heart$Cholesterol, na.rm = TRUE), 
     main = "Porównanie", xlab = "Cholesterol", col = "blue", lwd = 1.5, ylim= c(0,0.01))
lines(density(dane_heart_omit$Cholesterol, na.rm = TRUE), col = "red", lwd = 1.5)
lines(density(dane_heart_mean$Cholesterol, na.rm = TRUE), col = "green", lwd = 1.5)
lines(density(dane_heart_med$Cholesterol, na.rm = TRUE), col = "orange", lwd = 1.5)
lines(density(dane_heart_knn$Cholesterol, na.rm = TRUE), col = "purple", lwd = 1.5)

par(mfrow = c(1, 1))


#Wiek
par(mfrow = c(2, 3))

plot(density(dane_heart$Age, na.rm = TRUE), 
     main = "Oryginalne", xlab = "Wiek", col = "blue")

plot(density(dane_heart_omit$Age, na.rm = TRUE), 
     main = "Usunięte braki", xlab = "Wiek", col = "red")

plot(density(dane_heart_mean$Age, na.rm = TRUE), 
     main = "Średnia", xlab = "Wiek", col = "green")

plot(density(dane_heart_med$Age, na.rm = TRUE), 
     main = "Mediana", xlab = "Wiek", col = "orange")

plot(density(dane_heart_knn$Age, na.rm = TRUE), 
     main = "KNN", xlab = "Wiek", col = "purple")

plot(density(dane_heart$Age, na.rm = TRUE), 
     main = "Porównanie", xlab = "Wiek", col = "blue", lwd = 1.5, ylim = c(0, 0.055))
lines(density(dane_heart_omit$Age, na.rm = TRUE), col = "red", lwd = 1.5)
lines(density(dane_heart_mean$Age, na.rm = TRUE), col = "green", lwd = 1.5)
lines(density(dane_heart_med$Age, na.rm = TRUE), col = "orange", lwd = 1.5)
lines(density(dane_heart_knn$Age, na.rm = TRUE), col = "purple", lwd = 1.5)

par(mfrow = c(1, 1))


#Oldpeak
par(mfrow = c(2, 3))

plot(density(dane_heart$Oldpeak, na.rm = TRUE), 
     main = "Oryginalne", xlab = "Oldpeak", col = "blue")

plot(density(dane_heart_omit$Oldpeak, na.rm = TRUE), 
     main = "Usunięte braki", xlab = "Oldpeak", col = "red")

plot(density(dane_heart_mean$Oldpeak, na.rm = TRUE), 
     main = "Średnia", xlab = "Oldpeak", col = "green")

plot(density(dane_heart_med$Oldpeak, na.rm = TRUE), 
     main = "Mediana", xlab = "Oldpeak", col = "orange")

plot(density(dane_heart_knn$Oldpeak, na.rm = TRUE), 
     main = "KNN", xlab = "Oldpeak", col = "purple")

plot(density(dane_heart$Oldpeak, na.rm = TRUE), 
     main = "Porównanie", xlab = "Oldpeak", col = "blue", lwd = 1.5, ylim = c(0, 0.8))
lines(density(dane_heart_omit$Oldpeak, na.rm = TRUE), col = "red", lwd = 1.5)
lines(density(dane_heart_mean$Oldpeak, na.rm = TRUE), col = "green", lwd = 1.5)
lines(density(dane_heart_med$Oldpeak, na.rm = TRUE), col = "orange", lwd = 1.5)
lines(density(dane_heart_knn$Oldpeak, na.rm = TRUE), col = "purple", lwd = 1.5)

par(mfrow = c(1, 1))

#========================
#zad 2

##stroke
#Podział na zbiór uczący i testowy
#usunięte
set.seed(123)
index_s_omit = sample(length(dane_stroke_omit$gender), size=0.2*nrow(dane_stroke_omit))
testowy_s_omit = dane_stroke_omit[index_s_omit,]
uczacy_s_omit = dane_stroke_omit[-index_s_omit,]

#standaryzacja
stand = c(2, 8, 9)
uczacy_s_omit[,stand] = scale(uczacy_s_omit[,stand])
testowy_s_omit[,stand] = scale(testowy_s_omit[,stand])

#średnia
index_s_mean = sample(length(dane_stroke_mean$gender), size=0.2*nrow(dane_stroke_mean))
testowy_s_mean = dane_stroke_mean[index_s_mean,]
uczacy_s_mean = dane_stroke_mean[-index_s_mean,]

#stand
uczacy_s_mean[,stand] = scale(uczacy_s_mean[,stand])
testowy_s_mean[,stand] = scale(testowy_s_mean[,stand])

#mediana
index_s_med = sample(length(dane_stroke_med$gender), size=0.2*nrow(dane_stroke_med))
testowy_s_med = dane_stroke_med[index_s_med,]
uczacy_s_med = dane_stroke_med[-index_s_med,]

#stand
uczacy_s_med[,stand] = scale(uczacy_s_med[,stand])
testowy_s_med[,stand] = scale(testowy_s_med[,stand])

#knn
index_s_knn = sample(length(dane_stroke_knn$gender), size=0.2*nrow(dane_stroke_knn))
testowy_s_knn = dane_stroke_knn[index_s_knn,]
uczacy_s_knn = dane_stroke_knn[-index_s_knn,]

#stand
uczacy_s_knn[,stand] = scale(uczacy_s_knn[,stand])
testowy_s_knn[,stand] = scale(testowy_s_knn[,stand])



#REGRESJA LOGISTYCZNA

#Usunięte
glm_model_s_omit = glm(stroke ~ ., data = uczacy_s_omit, family = "binomial")
summary(glm_model_s_omit)

#Predykcja na zbiorze uczącym
pred_uczacy_s_omit = predict(glm_model_s_omit, newdata = uczacy_s_omit, type = "response")
prog_odciecia = 0.5
klasy_uczacy_s_omit = ifelse(pred_uczacy_s_omit >= prog_odciecia, 1, 0)
confusionMatrix(factor(klasy_uczacy_s_omit), factor(uczacy_s_omit$stroke), positive = "1")

#Predykcja na zbiorze testowym
pred_testowy_s_omit = predict(glm_model_s_omit, newdata = testowy_s_omit, type = "response")
klasy_testowy_s_omit = ifelse(pred_testowy_s_omit >= prog_odciecia, 1, 0)
confusionMatrix(factor(klasy_testowy_s_omit), factor(testowy_s_omit$stroke), positive = "1")


#Średnia
glm_model_s_mean = glm(stroke ~ ., data = uczacy_s_mean, family = "binomial")
summary(glm_model_s_mean)

#Predykcja na zbiorze uczącym
pred_uczacy_s_mean = predict(glm_model_s_mean, newdata = uczacy_s_mean, type = "response")
klasy_uczacy_s_mean = ifelse(pred_uczacy_s_mean >= prog_odciecia, 1, 0)
confusionMatrix(factor(klasy_uczacy_s_mean), factor(uczacy_s_mean$stroke), positive = "1")

#Predykcja na zbiorze testowym
pred_testowy_s_mean = predict(glm_model_s_mean, newdata = testowy_s_mean, type = "response")
klasy_testowy_s_mean = ifelse(pred_testowy_s_mean >= prog_odciecia, 1, 0)
confusionMatrix(factor(klasy_testowy_s_mean), factor(testowy_s_mean$stroke), positive = "1")


#Mediana
#zapewnienie pozostania wszystkich unikalnych wartości work_type
glm_model_s_med = glm(stroke ~ ., data = uczacy_s_med, family = "binomial")
summary(glm_model_s_med)

#Predykcja na zbiorze uczącym
pred_uczacy_s_med = predict(glm_model_s_med, newdata = uczacy_s_med, type = "response")
klasy_uczacy_s_med = ifelse(pred_uczacy_s_med >= prog_odciecia, 1, 0)
confusionMatrix(factor(klasy_uczacy_s_med), factor(uczacy_s_med$stroke), positive = "1")

#Predykcja na zbiorze testowym
#zamiana Never Worked na Private, z uwagi na problem, gdzie glm() pomija w modelu wartość Never Worked, która nie występuje w zbiorze uczącym
testowy_s_med$work_type <- as.character(testowy_s_med$work_type)
testowy_s_med$work_type[testowy_s_med$work_type == "Never_worked"] <- "Private"
testowy_s_med$work_type <- factor(testowy_s_med$work_type,
                                  levels = levels(uczacy_s_med$work_type))
pred_testowy_s_med = predict(glm_model_s_med, newdata = testowy_s_med, type = "response")
klasy_testowy_s_med = ifelse(pred_testowy_s_med >= prog_odciecia, 1, 0)
confusionMatrix(factor(klasy_testowy_s_med), factor(testowy_s_med$stroke), positive = "1")


#KNN
glm_model_s_knn = glm(stroke ~ ., data = uczacy_s_knn, family = "binomial")
summary(glm_model_s_knn)

#Predykcja na zbiorze uczącym
pred_uczacy_s_knn = predict(glm_model_s_knn, newdata = uczacy_s_knn, type = "response")
klasy_uczacy_s_knn = ifelse(pred_uczacy_s_knn >= prog_odciecia, 1, 0)
confusionMatrix(factor(klasy_uczacy_s_knn), factor(uczacy_s_knn$stroke), positive = "1")

#Predykcja na zbiorze testowym
pred_testowy_s_knn = predict(glm_model_s_knn, newdata = testowy_s_knn, type = "response")
klasy_testowy_s_knn = ifelse(pred_testowy_s_knn >= prog_odciecia, 1, 0)
confusionMatrix(factor(klasy_testowy_s_knn), factor(testowy_s_knn$stroke), positive = "1")


#================
#Podsumowanie
get_metrics = function(pred, true) {
  cm = confusionMatrix(factor(pred), factor(true), positive = "1")
  data.frame(
    Accuracy    = cm$overall['Accuracy'],
    Sensitivity = cm$byClass['Sensitivity'],
    Specificity = cm$byClass['Specificity'],
    row.names = NULL
  )
}

# Tworzymy tabelę porównawczą
wyniki = bind_rows(
  get_metrics(klasy_testowy_s_omit, testowy_s_omit$stroke) %>% mutate(Model = "Usunięte"),
  get_metrics(klasy_testowy_s_mean, testowy_s_mean$stroke) %>% mutate(Model = "Średnia"),
  get_metrics(klasy_testowy_s_med,  testowy_s_med$stroke)  %>% mutate(Model = "Mediana"),
  get_metrics(klasy_testowy_s_knn,  testowy_s_knn$stroke)  %>% mutate(Model = "KNN")
)

# Porządkowanie kolumn
wyniki = wyniki %>% select(Model, Accuracy, Sensitivity, Specificity)
wyniki



##HEART


set.seed(123)

# PODZIAŁ NA ZBIORY UCZĄCY/TESTOWY
# dla każdego wariantu danych

# Usunięte
index_h_omit = sample(1:nrow(dane_heart_omit), size = 0.2 * nrow(dane_heart_omit))
testowy_h_omit = dane_heart_omit[index_h_omit, ]
uczacy_h_omit = dane_heart_omit[-index_h_omit, ]

# Średnia
index_h_mean = sample(1:nrow(dane_heart_mean), size = 0.2 * nrow(dane_heart_mean))
testowy_h_mean = dane_heart_mean[index_h_mean, ]
uczacy_h_mean = dane_heart_mean[-index_h_mean, ]

# Mediana
index_h_med = sample(1:nrow(dane_heart_med), size = 0.2 * nrow(dane_heart_med))
testowy_h_med = dane_heart_med[index_h_med, ]
uczacy_h_med = dane_heart_med[-index_h_med, ]

# KNN
index_h_knn = sample(1:nrow(dane_heart_knn), size = 0.2 * nrow(dane_heart_knn))
testowy_h_knn = dane_heart_knn[index_h_knn, ]
uczacy_h_knn = dane_heart_knn[-index_h_knn, ]


# STANDARYZACJA ZMIENNYCH NUMERYCZNYCH

kolumny_num = c("Age", "Cholesterol", "Oldpeak")

uczacy_h_omit[, kolumny_num] = scale(uczacy_h_omit[, kolumny_num])
testowy_h_omit[, kolumny_num] = scale(testowy_h_omit[, kolumny_num])

uczacy_h_mean[, kolumny_num] = scale(uczacy_h_mean[, kolumny_num])
testowy_h_mean[, kolumny_num] = scale(testowy_h_mean[, kolumny_num])

uczacy_h_med[, kolumny_num] = scale(uczacy_h_med[, kolumny_num])
testowy_h_med[, kolumny_num] = scale(testowy_h_med[, kolumny_num])

uczacy_h_knn[, kolumny_num] = scale(uczacy_h_knn[, kolumny_num])
testowy_h_knn[, kolumny_num] = scale(testowy_h_knn[, kolumny_num])

# REGRESJA LOGISTYCZNA

prog_odciecia = 0.5

# Usunięte
glm_model_h_omit = glm(HeartDisease ~ ., data = uczacy_h_omit, family = "binomial")
pred_testowy_h_omit = predict(glm_model_h_omit, newdata = testowy_h_omit, type = "response")
klasy_testowy_h_omit = ifelse(pred_testowy_h_omit >= prog_odciecia, 1, 0)
confusionMatrix(factor(klasy_testowy_h_omit), factor(testowy_h_omit$HeartDisease), positive = "1")

# Średnia
glm_model_h_mean = glm(HeartDisease ~ ., data = uczacy_h_mean, family = "binomial")
pred_testowy_h_mean = predict(glm_model_h_mean, newdata = testowy_h_mean, type = "response")
klasy_testowy_h_mean = ifelse(pred_testowy_h_mean >= prog_odciecia, 1, 0)
confusionMatrix(factor(klasy_testowy_h_mean), factor(testowy_h_mean$HeartDisease), positive = "1")

# Mediana
glm_model_h_med = glm(HeartDisease ~ ., data = uczacy_h_med, family = "binomial")
pred_testowy_h_med = predict(glm_model_h_med, newdata = testowy_h_med, type = "response")
klasy_testowy_h_med = ifelse(pred_testowy_h_med >= prog_odciecia, 1, 0)
confusionMatrix(factor(klasy_testowy_h_med), factor(testowy_h_med$HeartDisease), positive = "1")

# KNN
glm_model_h_knn = glm(HeartDisease ~ ., data = uczacy_h_knn, family = "binomial")
pred_testowy_h_knn = predict(glm_model_h_knn, newdata = testowy_h_knn, type = "response")
klasy_testowy_h_knn = ifelse(pred_testowy_h_knn >= prog_odciecia, 1, 0)
confusionMatrix(factor(klasy_testowy_h_knn), factor(testowy_h_knn$HeartDisease), positive = "1")

# PODSUMOWANIE WYNIKÓW
get_metrics = function(pred, true) {
  cm = confusionMatrix(factor(pred), factor(true), positive = "1")
  data.frame(
    Accuracy    = cm$overall['Accuracy'],
    Sensitivity = cm$byClass['Sensitivity'],
    Specificity = cm$byClass['Specificity'],
    row.names = NULL
  )
}

wyniki_heart = bind_rows(
  get_metrics(klasy_testowy_h_omit, testowy_h_omit$HeartDisease) %>% mutate(Model = "Usunięte"),
  get_metrics(klasy_testowy_h_mean, testowy_h_mean$HeartDisease) %>% mutate(Model = "Średnia"),
  get_metrics(klasy_testowy_h_med,  testowy_h_med$HeartDisease)  %>% mutate(Model = "Mediana"),
  get_metrics(klasy_testowy_h_knn,  testowy_h_knn$HeartDisease)  %>% mutate(Model = "KNN")
)

wyniki_heart = wyniki_heart %>% select(Model, Accuracy, Sensitivity, Specificity)
wyniki_heart

# WIZUALIZACJA WYNIKÓW

wyniki_long_heart = wyniki_heart %>%
  pivot_longer(cols = c(Accuracy, Sensitivity, Specificity),
               names_to = "Metryka",
               values_to = "Wartość")

ggplot(wyniki_long_heart, aes(x = Model, y = Wartość, fill = Metryka)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Porównanie jakości modeli dla danych Heart",
       y = "Wartość metryki", x = "Sposób eliminacji braków danych") +
  theme_minimal()


