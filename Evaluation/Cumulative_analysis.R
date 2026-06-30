install.packages("ordinal")
install.packages("readr")


library(ordinal)
df <- read.csv("long_cleaned_data_gpt_novelty.csv")
df$Rater <- as.factor(df$Rater)
df$RQ <- as.factor(df$RQ)
df$System <- as.factor(df$System)

df$Novelty <- ordered(df$Novelty, levels=c(1,2,3,4,5))

model <- clmm(
  Novelty ~ System +
    (1 | Rater) +
    (1 | RQ),
  data = df,
  link = "logit"
)

summary(model)



library(ordinal)
df <- read.csv("long_cleaned_data_gpt_usefulness.csv")
df$Rater <- as.factor(df$Rater)
df$RQ <- as.factor(df$RQ)
df$System <- as.factor(df$System)

df$Usefulness <- ordered(df$Usefulness, levels=c(1,2,3,4,5))

model <- clmm(
  Usefulness ~ System +
    (1 | Rater) +
    (1 | RQ),
  data = df,
  link = "logit"
)

summary(model)


library(ordinal)
df <- read.csv("long_cleaned_data_gpt_clarity.csv")
df$Rater <- as.factor(df$Rater)
df$RQ <- as.factor(df$RQ)
df$System <- as.factor(df$System)

df$Clarity <- ordered(df$Clarity, levels=c(1,2,3,4,5))

model <- clmm(
  Clarity ~ System +
    (1 | Rater) +
    (1 | RQ),
  data = df,
  link = "logit"
)

summary(model)

install.packages("lme4")
install.packages("lmerTest")   # optional



library(lme4)
data <- read.csv("combined_mixed_effects_data_1.csv")
data$system <- factor(data$system)
data$reviewer <- factor(data$reviewer)
data$requirement <- factor(data$requirement)

model <- glmer(
  rating ~ system +
    (1 | requirement) +
    (1 | reviewer),
  data = data,
  family = binomial(link = "logit")
)
summary(model)

exp(fixef(model))
anova(model)