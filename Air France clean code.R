
install.packages("fastDummies")
install.packages("worldcloud2")
install.packages("tm")

# Load Libraries
library(readr)
library(stringr)
library(fastDummies)
library(ggplot2)
library(plotly)
library(dplyr)
library(caret)
library(wordcloud)
library(RColorBrewer)
library(wordcloud2)
library(tm)
library(ROCR)


# load the stringr package
library("stringr")

airfrance_df <- read_csv("HULT/Spring Term 22/Data Science - R/99. Resources/Air France Case Spreadsheet Supplement_DoubleClick.csv")

# Transform csv file into df
airfrance_df <- data.frame(airfrance_df)

# kayak parameter
kayak_param <- c("Kayak1","Kayak", 0, "o","N/A","Unassigned", "Unassigned", "uncategorized","", 
                 "Unassigned", "Unavailable", 0.00, 2.839, 3567.13,  1123.53, 0,0,0,0,0, 233694, 3567.13, 208)

#merge kayak information
airfrance_df <- rbind(airfrance_df, kayak_param)

# transform every numeric variable into numberic type
airfrance_df$Amount <- as.numeric(airfrance_df$Amount)
airfrance_df$Total.Cost <- as.numeric(airfrance_df$Total.Cost)
airfrance_df$Search.Engine.Bid <- as.numeric(airfrance_df$Search.Engine.Bid)
airfrance_df$Clicks <- as.numeric(airfrance_df$Clicks)
airfrance_df$Click.Charges <- as.numeric(airfrance_df$Click.Charges)
airfrance_df$Avg..Cost.per.Click <- as.numeric(airfrance_df$Avg..Cost.per.Click)
airfrance_df$Impressions <- as.numeric(airfrance_df$Impressions)
airfrance_df$Engine.Click.Thru.. <- as.numeric(airfrance_df$Engine.Click.Thru..)
airfrance_df$Avg..Pos. <- as.numeric(airfrance_df$Avg..Pos.)
airfrance_df$Trans..Conv... <- as.numeric(airfrance_df$Trans..Conv...)
airfrance_df$Total.Cost..Trans. <- as.numeric(airfrance_df$Total.Cost..Trans.)
airfrance_df$Total.Volume.of.Bookings <- as.numeric(airfrance_df$Total.Volume.of.Bookings)


# Replace spaces by _
airfrance_df$Publisher.Name <- gsub(" ", "", airfrance_df$Publisher.Name)
airfrance_df$Publisher.Name <- gsub("-", "_", airfrance_df$Publisher.Name)
airfrance_df$Publisher.Name <- gsub("-", "_", airfrance_df$Publisher.Name)



# Create ROA column
airfrance_df$roa <- ifelse(airfrance_df$Total.Cost == 0,0,  airfrance_df$Amount/airfrance_df$Total.Cost)

# Create dummies 

airfrance_df <-  dummy_cols(airfrance_df, select_columns ='Publisher.Name')
airfrance_df <-  dummy_cols(airfrance_df, select_columns ='Status')
airfrance_df <-  dummy_cols(airfrance_df, select_columns ='Match.Type')


# Create a new column that is 1 when ROA is greater 8.53
airfrance_df$roa_num <- ifelse(airfrance_df$roa >2, 1, 0)

# First logistic regression
my_logit <- glm(roa_num ~ Clicks + Impressions + Publisher.Name_Google_US + 
                  Publisher.Name_Google_Global+ Publisher.Name_MSN_US + 
                  Publisher.Name_Overture_Global + 
                  airfrance_df$Publisher.Name_MSN_Global +
                  Publisher.Name_Overture_US + Engine.Click.Thru.. + 
                  Search.Engine.Bid + Trans..Conv... +Avg..Pos.,
                data=airfrance_df, family = "binomial")

summary(my_logit)


# businees insight of clicks
Clicks_odds < - (exp(1.878e-04)-1)*1000
Clicks_odds
# Understand impact of coefficients with normalization

# Normalization technique -> min-max re-scalling (=percentiles creation)
normalize <- function(var) {                  #opening the normalization UDF / input with a <variable>
  
  #creating a temporarily object: min_max
  min_max <- (var - min(var)) / (max(var) - min(var)) #(variable - Xmin) / (Xmax - Xmin)
  return(min_max)                                     #returning the <min_max> object
  
}  
#Normalize continous variables. There is no need to normalize dummies, as they are already between 0 & 1
airfrance_df$norm_clicks <- normalize(var =airfrance_df$Clicks )
airfrance_df$norm_imp <- normalize(var =airfrance_df$Impressions )
airfrance_df$norm_Engine.Click.Thru.. <- normalize(var =airfrance_df$Engine.Click.Thru.. )
airfrance_df$norm_Search.Engine.Bid <- normalize(var =airfrance_df$Search.Engine.Bid )
airfrance_df$norm_Avg..Pos. <- normalize(var =airfrance_df$Avg..Pos. )
airfrance_df$norm_Trans..Conv... <- normalize(var =airfrance_df$Trans..Conv... )



# Running normalized model
my_logit_norm <- glm(roa_num ~ norm_clicks + norm_imp + Publisher.Name_Google_US +
                       Publisher.Name_Google_Global + Publisher.Name_MSN_US + 
                       Publisher.Name_Overture_Global + 
                       airfrance_df$Publisher.Name_MSN_Global + 
                       Publisher.Name_Overture_US + norm_Engine.Click.Thru.. + 
                       norm_Search.Engine.Bid +norm_Avg..Pos. + norm_Trans..Conv... ,
                     data=airfrance_df, family = "binomial")

summary(my_logit_norm)


# Split stratified Training and Test dataset
# Random Statrified
set.seed(2021)
split <- rsample::initial_split(airfrance_df, prop = 0.8, strata = roa_num)

# Create Train and Test dataset
airfrance_df_train <- rsample::training(split)
airfrance_df_test <-rsample::testing(split)

# Train Logit Model
# Logit Model
logit_airfrance_df_train <- glm(roa_num ~ Clicks + Impressions + 
                                  Publisher.Name_Google_US + 
                                  Publisher.Name_Google_Global + 
                                  Publisher.Name_MSN_US + 
                                  Publisher.Name_Overture_Global + 
                                  Publisher.Name_MSN_Global +
                                  Publisher.Name_Overture_US + 
                                  Engine.Click.Thru.. + Search.Engine.Bid + 
                                  Trans..Conv... +Avg..Pos. , 
                                data= airfrance_df_train, family = "binomial")
summary(logit_airfrance_df_train)

# Predict test data
airfrance_predict <- predict(logit_airfrance_df_train, airfrance_df_test, type = "response")

# Create confussion matrix
confusionMatrix(data = as.factor(as.numeric(airfrance_predict>0.5)) ,
                reference = as.factor(as.numeric(airfrance_df_test$roa_num)))


# ROCR
airfrance_pred <- prediction(airfrance_predict, airfrance_df_test$roa_num)
per_logit <- performance(airfrance_pred, 'tpr', 'fpr')
plot(per_logit)


# Plots
unique_channel <- list(unique(c(airfrance_df$Publisher.Name)) )

# grouping and adding dataframe by publisher and campaing
airfrance_df_group_PName_Camp <- aggregate(x = airfrance_df$Total.Volume.of.Bookings, 
                                           by = list(airfrance_df$Publisher.Name, 
                                                     airfrance_df$Campaign), 
                                           FUN = sum)

#Ploting a bubble plot

airfrance_df_group_PName_Camp %>%
  #filter(x > 0) %>%
  ggplot(aes(x=Group.1, y=x, color = Group.2))  +
  geom_point(size = 5) + 
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1), legend.text = element_text(size=7)) + 
  ylab("# of Bookings") + 
  xlab("") +
  labs( color = "Campaign")

# group dataframe by number of bookings
airfrance_df_group_Publishername <- aggregate(x = airfrance_df$Total.Volume.of.Bookings, 
                                              by = list(airfrance_df$Publisher.Name), 
                                              FUN = sum)

airfrance_df_group_Publishername  %>%
  ggplot(aes(x= reorder(Group.1, -x), y=x, fill = Group.1))  +
  geom_bar(stat = 'identity') +xlab("")+
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) + 
  ylab("# of Bookings") +labs( fill = "Publisher")



# Word Cloud 
# Global

#Create a vector containing only the text
keywords <- airfrance_df$Keyword

# Create a corpus  
docs <- Corpus(VectorSource(keywords))

# Create TermDocumentMatrix object
dtm <- TermDocumentMatrix(docs) 
matrix <- as.matrix(dtm) 
words <- sort(rowSums(matrix),decreasing=TRUE) 
# Create Dataframe with each keyword and its frequency
df <- data.frame(word = names(words),freq=words)
view(df)

# Set seed for reproducibility
set.seed(1234) 

# Using wordcloud2 package to create the visualization
wordcloud2(data=df, size = 0.4, shape = 'circle')

Group keywords by ROA
airfrance_df_group_keyword<- aggregate(x = airfrance_df$roa, by = list(airfrance_df$Keyword), FUN = mean)



airfrance_df_group_keyword_filter <- filter(airfrance_df_group_keyword,airfrance_df_group_keyword$x > 0)


airfrance_df_group_keyword_filter %>%
  ggplot(aes(x = reorder(Group.1, -x), y = x, fill = -x))  +
  geom_bar(stat = 'identity') + theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) + ylab("ROA") + xlab("")
labs( color = "Frequency")
# Scatter plot Avg cost per click vs Avg cost per conversion
cpc_avg <- aggregate(Publisher.Name ~ cpc,x = airfrance_df$Avg..Cost.per.Click, 
                     by = list(Publisher.Name = airfrance_df$Publisher.Name), 
                     FUN = mean)

cpconv<- aggregate(x = airfrance_df$Total.Cost..Trans., 
                   by = list(Publisher.Name = airfrance_df$Publisher.Name), 
                   FUN = mean)

imp_size<- aggregate(x = airfrance_df$Impressions, 
                     by = list(Publisher.Name = airfrance_df$Publisher.Name), 
                     FUN = sum)

df_1 <- merge(cpc_avg, cpconv, by=c("Publisher.Name"),all.x=TRUE)
df_2 <- merge(df_1, imp_size, by=c("Publisher.Name"),all.x=TRUE)

ggplot(df_2, aes(x=x.x, y=x.y, size = x, color = Publisher.Name)) +
  geom_point(alpha=0.7) +xlab("Average Cost per Click") + ylab("Average Cost per Conversion") +labs(size = "Number of impressions", color = "SEM Channel")

